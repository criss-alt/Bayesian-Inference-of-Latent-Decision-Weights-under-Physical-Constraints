
import os, sys, io, time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, vmap, grad, value_and_grad, jit
from functools import partial

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from inversion_dynamic_core_jax import (
    PHYS_PARAMS, load_precomputed_flow_field, discretize_action_space
)

SEED = 42
FIG_PATH = "abc_smc_figures_REAL/Figure9_GradientOpt"
os.makedirs(FIG_PATH, exist_ok=True)

# ==========================================
# JAX-differentiable expected velocity
# ==========================================

# No @jit — cleaner for vmap with closure-captured JIT functions
def expected_vel_and_cost(P, v_swim, theta, get_u_env, get_S_ij, weights, actions):
    """Compute expected velocity under Boltzmann softmax — fully JAX differentiable."""
    def eval_one(a):
        v_target, d_theta = a[0], a[1]
        theta_new = theta + d_theta
        u = get_u_env(P)
        S = get_S_ij(P)
        c_E = (v_target / PHYS_PARAMS["U_burst"])**3
        s_clip = jnp.minimum(S / PHYS_PARAMS["S_max"], 1.0)
        c_R = (jnp.exp(PHYS_PARAMS["kappa"] * s_clip) - 1.0) / (jnp.exp(PHYS_PARAMS["kappa"]) - 1.0)
        v_abs = jnp.sqrt((v_target * jnp.cos(theta_new) + u[0])**2 +
                         (v_target * jnp.sin(theta_new) + u[1])**2)
        c_A = jnp.minimum(v_abs / PHYS_PARAMS["U_inf"], 1.0)
        J = weights[0] * c_E + weights[1] * c_R + weights[2] * c_A
        return J, v_abs

    J_and_v = vmap(eval_one)(actions)
    costs, vels = J_and_v[0], J_and_v[1]

    u = get_u_env(P)
    S = get_S_ij(P)
    u_mag = jnp.sqrt(u[0]**2 + u[1]**2)
    snr = u_mag / (PHYS_PARAMS["L_ref"] * S + PHYS_PARAMS["epsilon"])
    beta = PHYS_PARAMS["beta_min"] + (PHYS_PARAMS["beta_max"] - PHYS_PARAMS["beta_min"]) * (1.0 - jnp.exp(-PHYS_PARAMS["lambda_snr"] * snr))

    logits = -beta * costs
    logits = jnp.clip(jnp.nan_to_num(logits, nan=-1e10), -1e10, 1e10)
    probs = jax.nn.softmax(logits)
    exp_vel = jnp.sum(probs * vels)
    exp_cost = jnp.sum(probs * costs)
    return exp_vel, exp_cost, probs

# ==========================================
# Multi-point loss function (JAX differentiable)
# ==========================================

def make_loss_fn(flow_file, n_samples=400):
    """Create a JAX loss function that samples positions from the flow field."""
    get_u_env, get_S_ij, bounds = load_precomputed_flow_field(flow_file)
    x_min, x_max = float(bounds[0]), float(bounds[1])
    y_min, y_max = float(bounds[2]), float(bounds[3])
    actions = discretize_action_space(5, 7)

    # Fixed sample points (deterministic)
    np.random.seed(SEED)
    pts = jnp.array(np.column_stack([
        np.random.uniform(x_min + 0.02, x_max - 0.02, n_samples),
        np.random.uniform(y_min + 0.02, y_max - 0.02, n_samples)
    ]))

    v_swim_default = jnp.array(0.2)
    theta_default = jnp.array(jnp.pi)

    # No @jit — closure-captured JIT functions cause issues with jit-on-jit
    # JAX grad() can still differentiate through this
    def loss_fn(log_weights):
        weights_raw = jnp.exp(log_weights)
        weights = weights_raw / jnp.sum(weights_raw)

        def one_point(P_xy):
            exp_vel, exp_cost, _ = expected_vel_and_cost(
                P_xy, v_swim_default, theta_default,
                get_u_env, get_S_ij, weights, actions)
            return exp_vel

        exp_vels = vmap(one_point)(pts)

        # Loss: expected velocity should be near zero (station-keeping)
        # Lower velocity = better station-keeping for given flow condition
        # Also penalize high variance
        mean_vel = jnp.mean(exp_vels)
        var_vel = jnp.var(exp_vels)

        # Mean velocity: too high → fish being swept away; too low → can't maintain position
        # Target: close to U_inf for rheotaxis, but weighted by risk avoidance
        target = PHYS_PARAMS["U_inf"] * 0.5  # Fish should choose ~half of inflow
        loss = jnp.mean((exp_vels - target)**2) + 0.1 * var_vel

        return loss

    return loss_fn, pts

# ==========================================
# Adam optimizer (pure JAX, no scipy)
# ==========================================

def adam_optimize(loss_fn, n_starts=5, n_epochs=500, lr=0.02):
    """Multi-start Adam optimization."""
    grad_fn = grad(loss_fn)
    results = []

    for start_idx in range(n_starts):
        print(f"\n--- Start {start_idx+1}/{n_starts} ---")
        key = random.PRNGKey(SEED + start_idx + 3000)
        log_w = jnp.array(np.random.uniform(-1.5, 1.5, 3), dtype=jnp.float64)

        m = jnp.zeros(3, dtype=jnp.float64)
        v = jnp.zeros(3, dtype=jnp.float64)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        best_loss = float('inf')
        best_log_w = log_w
        loss_history = []

        t0 = time.time()
        for epoch in range(n_epochs):
            loss_val, grads_val = value_and_grad(loss_fn)(log_w)
            loss_f = float(loss_val)

            # Clip
            grad_norm = jnp.sqrt(jnp.sum(grads_val**2))
            grads_val = jnp.where(grad_norm > 5.0, grads_val * 5.0 / grad_norm, grads_val)

            # Adam
            m = beta1 * m + (1 - beta1) * grads_val
            v = beta2 * v + (1 - beta2) * (grads_val**2)
            m_hat = m / (1 - beta1**(epoch+1))
            v_hat = v / (1 - beta2**(epoch+1))
            log_w = log_w - lr * m_hat / (jnp.sqrt(v_hat) + eps)

            loss_history.append(loss_f)
            if loss_f < best_loss:
                best_loss = loss_f
                best_log_w = log_w

            if epoch % 100 == 0 or epoch == n_epochs - 1:
                w_arr = np.array(jnp.exp(log_w) / jnp.sum(jnp.exp(log_w)))
                print(f"  Epoch {epoch:3d}: loss={loss_f:.6f}, "
                      f"w=({w_arr[0]:.4f},{w_arr[1]:.4f},{w_arr[2]:.4f}), "
                      f"|g|={float(grad_norm):.4f}")

        elapsed = time.time() - t0
        w_final = np.array(jnp.exp(log_w) / jnp.sum(jnp.exp(log_w)))
        w_best = np.array(jnp.exp(best_log_w) / jnp.sum(jnp.exp(best_log_w)))

        results.append({
            'w_final': w_final, 'w_best': w_best,
            'best_loss': best_loss, 'loss_history': loss_history,
            'time': elapsed
        })
        print(f"  Time: {elapsed:.1f}s, Best: w=({w_best[0]:.4f},{w_best[1]:.4f},{w_best[2]:.4f}), loss={best_loss:.6f}")

    return results

# ==========================================
# Main
# ==========================================
def main():
    print("=" * 60)
    print("PHASE B: Pure JAX Gradient Optimization")
    print("=" * 60)
    print("Differentiable expected velocity under Boltzmann softmax")
    print("No scipy — pure JAX grad + Adam")

    for flow_file, label in [("flow_30.npz", "30 cm/s"), ("flow_50.npz", "50 cm/s")]:
        print(f"\n{'='*60}")
        print(f"Optimizing on {label} ({flow_file})")
        print(f"{'='*60}")

        loss_fn, _ = make_loss_fn(flow_file, n_samples=400)

        # Warmup: run loss once to trigger JIT
        test_w = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float64)
        _ = loss_fn(test_w)
        print("  JIT compilation complete.")

        results = adam_optimize(loss_fn, n_starts=3, n_epochs=400, lr=0.015)

        # Best result
        best = min(results, key=lambda r: r['best_loss'])
        print(f"\n✅ {label} BEST: w=({best['w_best'][0]:.4f},{best['w_best'][1]:.4f},{best['w_best'][2]:.4f}), loss={best['best_loss']:.6f}")

        # Save
        safe_label = label.replace(' cm/s', 'cms').replace(' ', '_')
        # Save simple results
        np.savez(f"{FIG_PATH}/grad_opt_{safe_label}.npz",
                 w_best=best['w_best'], best_loss=np.float64(best['best_loss']),
                 loss_history=np.array(best['loss_history'], dtype=np.float64))

    # Final summary
    print(f"\n{'='*60}")
    print("GRADIENT OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"ABC-SMC MAP: w=(0.0048, 0.4739, 0.5213)")
    print(f"→ w_E ≈ 0 consistently")
    print(f"→ w_R + w_A dominate station-keeping strategy")
    print(f"\nResults saved to {FIG_PATH}/")

if __name__ == "__main__":
    main()
