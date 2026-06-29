# gradient_optimization.py
# Differentiable gradient-based optimization of behavior weights (w_E, w_R, w_A)
# Uses Gumbel-Softmax STE to replace ABC-SMC's sampling-based search

import os, sys, io, time, gc
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, vmap, jit, grad, value_and_grad
from functools import partial
from scipy.optimize import minimize
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from inversion_dynamic_core_jax import (
    PHYS_PARAMS, load_precomputed_flow_field,
    discretize_action_space
)
# Add Gumbel-Softmax functions
from inversion_dynamic_core_jax_diff import (
    sample_gumbel, compute_loss_single_speed as _jit_loss
)

SEED = 42
KEY = random.PRNGKey(SEED)
np.random.seed(SEED)

FIG_PATH = "abc_smc_figures_REAL/Figure9_GradientOpt"
os.makedirs(FIG_PATH, exist_ok=True)

# ==========================================
# 1. Pragmatic differentiable loss (hybrid JAX+NumPy)
# ==========================================

def compute_loss_hybrid(log_weights, flow_file, exp_file,
                         n_fish=50, t_max=300, burn_in=60, temperature=1.0):
    """
    Hybrid loss: JAX simulation + NumPy histogram.
    Uses Gumbel-Softmax for gradient through Boltzmann policy.
    """
    get_u_env, get_S_ij, bounds = load_precomputed_flow_field(flow_file)
    exp_data = np.load(exp_file, allow_pickle=True)
    exp_vel = exp_data["velocity_pdf"]
    v_edges = exp_data["velocity_edges"]
    exp_spatial = exp_data["spatial_pdf"]
    x_edges = exp_data["spatial_x_edges"]
    y_edges = exp_data["spatial_y_edges"]

    # Normalize weights
    weights_raw = np.exp(log_weights)
    weights_norm = weights_raw / np.sum(weights_raw)
    w_vec = jnp.array(weights_norm)

    actions = discretize_action_space(5, 7)

    # Run N fish
    fish_keys = random.split(random.PRNGKey(SEED + hash(str(log_weights.tobytes())) % 10000), n_fish)

    all_valid_v = []
    all_valid_P = []

    for i in range(n_fish):
        # Use Gumbel-Softmax for this fish
        P = jnp.array([np.random.uniform(0.12, 0.22), np.random.uniform(0.06, 0.14)])
        v_swim = jnp.array(0.2)
        theta = jnp.array(np.pi)
        step_keys = random.split(fish_keys[i], t_max)

        traj_v_list = []
        for t in range(t_max):
            # Evaluate costs for all actions
            costs_list = []
            for a_idx in range(len(actions)):
                a = actions[a_idx]
                # Simplified cost: energy + risk + anchor (subset of full model)
                v_target, d_theta = a[0], a[1]
                theta_new = theta + d_theta
                u_curr = get_u_env(P)
                S_curr = get_S_ij(P)

                # Energy cost
                c_E = (float(v_target) / PHYS_PARAMS["U_burst"])**3
                # Risk cost
                s_clip = jnp.minimum(float(S_curr) / PHYS_PARAMS["S_max"], 1.0)
                c_R = (jnp.exp(PHYS_PARAMS["kappa"] * s_clip) - 1.0) / (jnp.exp(PHYS_PARAMS["kappa"]) - 1.0)
                # Anchor cost
                v_abs_pred = jnp.sqrt((v_target * jnp.cos(theta_new) + float(u_curr[0]))**2 +
                                      (v_target * jnp.sin(theta_new) + float(u_curr[1]))**2)
                c_A = jnp.minimum(v_abs_pred / PHYS_PARAMS["U_inf"], 1.0)

                J = w_vec[0] * c_E + w_vec[1] * c_R + w_vec[2] * c_A
                costs_list.append(float(J))

            costs_arr = jnp.array(costs_list)
            snr = float(v_swim) / (PHYS_PARAMS["L_ref"] * float(S_curr) + PHYS_PARAMS["epsilon"])
            beta_state = PHYS_PARAMS["beta_min"] + (PHYS_PARAMS["beta_max"] - PHYS_PARAMS["beta_min"]) * (1.0 - jnp.exp(-PHYS_PARAMS["lambda_snr"] * snr))
            logits = jnp.nan_to_num(-beta_state * costs_arr, nan=-1e8)

            # Gumbel-Softmax STE
            gumbel = sample_gumbel(step_keys[t], logits.shape)
            y_soft = jax.nn.softmax((logits + gumbel) / temperature)
            action_idx = int(jnp.argmax(y_soft))
            action = actions[action_idx]
            # Store softmax probs for gradient computation
            y_soft_val = np.array(y_soft)

            v_new, d_theta = action[0], action[1]
            theta_new = theta + d_theta
            u_env = get_u_env(P)
            v_abs_x = v_new * jnp.cos(theta_new) + u_env[0]
            v_abs_y = v_new * jnp.sin(theta_new) + u_env[1]
            P_new = P + jnp.array([v_abs_x, v_abs_y]) * PHYS_PARAMS["Delta_t"]
            P_new = jnp.clip(P_new, jnp.array([0.0, 0.0]), jnp.array([0.7, 0.2]))

            traj_v_list.append(float(v_swim))
            P, v_swim, theta = P_new, v_new, theta_new

        traj_v_arr = np.array(traj_v_list[burn_in:])
        all_valid_v.append(traj_v_arr)

    # Concatenate
    all_v = np.concatenate(all_valid_v)

    # Velocity histogram
    v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])
    sim_vel_counts, _ = np.histogram(all_v, bins=v_edges, density=True)
    sim_vel_pmf = sim_vel_counts / (sim_vel_counts.sum() + 1e-10)

    # W1 distance (NumPy)
    sim_cdf = np.cumsum(sim_vel_pmf)
    exp_vel_norm = exp_vel / (exp_vel.sum() + 1e-10)
    exp_cdf = np.cumsum(exp_vel_norm)
    w1 = np.trapz(np.abs(sim_cdf - exp_cdf), v_centers)

    return float(w1), sim_vel_pmf, weights_norm, all_v

# ==========================================
# 2. Gradient computation via JAX
# ==========================================

def compute_w1_loss_for_grad(log_weights_jax, flow_file, exp_file,
                              n_fish=30, t_max=200, burn_in=40, temperature=1.0):
    """
    JAX-traced loss for gradient computation.
    Simplified to be JIT-compatible (fixed sizing).
    """
    get_u_env, get_S_ij, bounds = load_precomputed_flow_field(flow_file)
    exp_data = np.load(exp_file, allow_pickle=True)
    exp_vel_arr = np.asarray(exp_data["velocity_pdf"], dtype=np.float64)
    exp_vel_arr = exp_vel_arr / (exp_vel_arr.sum() + 1e-10)
    v_edges_arr = np.asarray(exp_data["velocity_edges"], dtype=np.float64)

    actions = discretize_action_space(5, 7)

    # Normalize
    weights_raw = jnp.exp(log_weights_jax)
    weights = weights_raw / jnp.sum(weights_raw)
    w_vec = weights

    # Key for reproducibility
    key = random.PRNGKey(SEED)

    fish_keys = random.split(key, n_fish)
    P_init_arr = np.column_stack([
        np.random.uniform(0.12, 0.22, n_fish),
        np.random.uniform(0.06, 0.14, n_fish)
    ])

    all_v_list = []

    for i in range(n_fish):
        P = jnp.array(P_init_arr[i])
        v_swim = jnp.array(0.2)
        theta = jnp.array(np.pi)
        step_keys = random.split(fish_keys[i], t_max)

        for t in range(t_max):
            # Vectorized cost evaluation
            def single_cost(a):
                v_target, d_theta = a[0], a[1]
                theta_new = theta + d_theta
                u_curr = get_u_env(P)
                S_curr = get_S_ij(P)
                c_E = (v_target / PHYS_PARAMS["U_burst"])**3
                s_clip = jnp.minimum(S_curr / PHYS_PARAMS["S_max"], 1.0)
                c_R = (jnp.exp(PHYS_PARAMS["kappa"] * s_clip) - 1.0) / (jnp.exp(PHYS_PARAMS["kappa"]) - 1.0)
                v_abs_pred = jnp.sqrt((v_target * jnp.cos(theta_new) + u_curr[0])**2 +
                                      (v_target * jnp.sin(theta_new) + u_curr[1])**2)
                c_A = jnp.minimum(v_abs_pred / PHYS_PARAMS["U_inf"], 1.0)
                return w_vec[0] * c_E + w_vec[1] * c_R + w_vec[2] * c_A

            costs = vmap(single_cost)(actions)
            snr = v_swim / (PHYS_PARAMS["L_ref"] * get_S_ij(P) + PHYS_PARAMS["epsilon"])
            beta_state = PHYS_PARAMS["beta_min"] + (PHYS_PARAMS["beta_max"] - PHYS_PARAMS["beta_min"]) * (1.0 - jnp.exp(-PHYS_PARAMS["lambda_snr"] * snr))
            logits = jnp.nan_to_num(-beta_state * costs, nan=-1e8)

            gumbel = sample_gumbel(step_keys[t], logits.shape)
            y_soft = jax.nn.softmax((logits + gumbel) / jnp.array(temperature))
            action_idx = jnp.argmax(y_soft)
            action = actions[action_idx]

            v_new, d_theta = action[0], action[1]
            theta_new = theta + d_theta
            u_env = get_u_env(P)
            P_new = P + jnp.array([v_new * jnp.cos(theta_new) + u_env[0],
                                   v_new * jnp.sin(theta_new) + u_env[1]]) * PHYS_PARAMS["Delta_t"]
            P_new = jnp.clip(P_new, jnp.array([0.0, 0.0]), jnp.array([0.7, 0.2]))

            all_v_list.append(v_swim)
            P, v_swim, theta = P_new, v_new, theta_new

    # Stack and burn-in
    all_v_jax = jnp.array([float(v) for v in all_v_list[burn_in * n_fish:]])

    # Histogram using jnp.histogram
    counts = jnp.zeros(len(v_edges_arr) - 1)
    for j in range(len(v_edges_arr) - 1):
        in_bin = (all_v_jax >= v_edges_arr[j]) & (all_v_jax < v_edges_arr[j+1])
        counts = counts.at[j].set(jnp.sum(in_bin))

    sim_pmf = counts / (jnp.sum(counts) + 1e-10)

    # W₁ (manual trapezoidal integration — jnp.trapz not in all JAX versions)
    sim_cdf = jnp.cumsum(sim_pmf)
    exp_cdf = jnp.cumsum(jnp.array(exp_vel_arr))
    v_centers = 0.5 * (v_edges_arr[:-1] + v_edges_arr[1:])
    diff = jnp.abs(sim_cdf - exp_cdf)
    # Manual trapz: sum(0.5*(y[i+1]+y[i])*(x[i+1]-x[i]))
    dx = v_centers[1:] - v_centers[:-1]
    w1 = jnp.sum(0.5 * (diff[1:] + diff[:-1]) * dx)

    return w1

# ==========================================
# 3. Multi-start gradient optimization
# ==========================================

def optimize_weights_gradient(flow_file="flow_30.npz", exp_file="target_pdf_30.0cms.npz",
                               n_starts=5, n_epochs=200, lr=0.05):
    """
    Multi-start gradient optimization of behavior weights.

    Uses Adam-like update with projection onto simplex.
    """
    print(f"\n{'='*60}")
    print(f"Gradient Optimization on {flow_file}")
    print(f"{'='*60}")

    loss_fn = partial(compute_w1_loss_for_grad, flow_file=flow_file, exp_file=exp_file)
    grad_fn = grad(loss_fn)

    all_trajectories = []
    all_final = []

    for start_idx in range(n_starts):
        print(f"\n--- Start {start_idx+1}/{n_starts} ---")

        # Random initialization in log-space
        np.random.seed(SEED + start_idx + 1000)
        init_log_w = np.random.uniform(-2, 2, 3)
        init_log_w = init_log_w.astype(np.float64)

        log_w = jnp.array(init_log_w)
        trajectory = [np.exp(init_log_w) / np.sum(np.exp(init_log_w))]

        # Adam parameters
        m = jnp.zeros(3)
        v = jnp.zeros(3)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        losses = []
        best_loss = float('inf')
        best_weights = None

        for epoch in range(n_epochs):
            try:
                loss_val, grads_val = value_and_grad(loss_fn)(log_w)
            except Exception as e:
                print(f"  Epoch {epoch}: grad error: {e}")
                break

            loss_f = float(loss_val)
            grads_arr = np.array(grads_val)

            # Clip gradients
            grad_norm = np.sqrt(np.sum(grads_arr**2))
            if grad_norm > 10.0:
                grads_arr = grads_arr / grad_norm * 10.0

            # Adam update
            m = beta1 * m + (1 - beta1) * grads_arr
            v = beta2 * v + (1 - beta2) * (grads_arr**2)
            m_hat = m / (1 - beta1**(epoch+1))
            v_hat = v / (1 - beta2**(epoch+1))

            log_w = log_w - lr * m_hat / (jnp.sqrt(v_hat) + eps)

            losses.append(loss_f)

            # Track best
            if loss_f < best_loss:
                best_loss = loss_f
                best_weights = np.exp(np.array(log_w)) / np.sum(np.exp(np.array(log_w)))

            # Record trajectory every 10 epochs
            if epoch % 10 == 0:
                w_norm = np.exp(np.array(log_w)) / np.sum(np.exp(np.array(log_w)))
                trajectory.append(w_norm)
                if epoch % 50 == 0:
                    print(f"  Epoch {epoch:3d}: loss={loss_f:.4f}, "
                          f"w=({w_norm[0]:.3f},{w_norm[1]:.3f},{w_norm[2]:.3f})")

        # Final weights
        w_final = np.exp(np.array(log_w)) / np.sum(np.exp(np.array(log_w)))
        trajectory.append(w_final)
        all_trajectories.append(np.array(trajectory))
        all_final.append(w_final)
        if losses:
            print(f"  Final: loss={losses[-1]:.4f}, w=({w_final[0]:.3f},{w_final[1]:.3f},{w_final[2]:.3f})")
        if best_weights is not None:
            print(f"  Best:  loss={best_loss:.4f}, w=({best_weights[0]:.3f},{best_weights[1]:.3f},{best_weights[2]:.3f})")

    return all_trajectories, all_final, best_weights, best_loss

# ==========================================
# 4. Main execution
# ==========================================
def main():
    print("=" * 60)
    print("PHASE B: Differentiable Gradient Optimization")
    print("=" * 60)

    # Optimize on 30 cm/s
    trajectories_30, finals_30, best_w_30, best_loss_30 = optimize_weights_gradient(
        "flow_30.npz", "target_pdf_30.0cms.npz", n_starts=3, n_epochs=200, lr=0.03)

    # Optimize on 50 cm/s
    trajectories_50, finals_50, best_w_50, best_loss_50 = optimize_weights_gradient(
        "flow_50.npz", "target_pdf_50.0cms.npz", n_starts=3, n_epochs=200, lr=0.03)

    # Save results
    results = {
        "trajectories_30": trajectories_30,
        "finals_30": np.array(finals_30),
        "best_w_30": best_w_30,
        "best_loss_30": best_loss_30,
        "trajectories_50": trajectories_50,
        "finals_50": np.array(finals_50),
        "best_w_50": best_w_50,
        "best_loss_50": best_loss_50,
    }
    np.savez(f"{FIG_PATH}/gradient_opt_results.npz",
             **{k: v for k, v in results.items() if isinstance(v, np.ndarray)},
             best_w_30=best_w_30, best_loss_30=best_loss_30,
             best_w_50=best_w_50, best_loss_50=best_loss_50,
             allow_pickle=True)

    print(f"\n{'='*60}")
    print("GRADIENT OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    print(f"\n30 cm/s best: w=({best_w_30[0]:.4f},{best_w_30[1]:.4f},{best_w_30[2]:.4f}), loss={best_loss_30:.4f}")
    print(f"50 cm/s best: w=({best_w_50[0]:.4f},{best_w_50[1]:.4f},{best_w_50[2]:.4f}), loss={best_loss_50:.4f}")
    print(f"\nABC-SMC MAP:   w=(0.0048, 0.4739, 0.5213)")
    print(f"Results saved to {FIG_PATH}/")

    return results

if __name__ == "__main__":
    results = main()
