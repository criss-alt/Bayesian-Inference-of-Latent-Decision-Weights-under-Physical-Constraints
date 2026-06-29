

import jax
import jax.numpy as jnp
from jax import random, vmap, jit
from functools import partial
import numpy as np

# Import base model
from inversion_dynamic_core_jax import (
    PHYS_PARAMS, load_precomputed_flow_field,
    discretize_action_space, interp_field
)


def sample_gumbel(key, shape, dtype=jnp.float64):
    """Sample from Gumbel(0, 1) distribution."""
    u = random.uniform(key, shape=shape, dtype=dtype, minval=1e-10, maxval=1.0 - 1e-10)
    return -jnp.log(-jnp.log(u))

@partial(jit, static_argnums=(4, 5))
def boltzmann_policy_gumbel_softmax(key, P, v_swim, theta, get_u_env,
                                     get_S_ij, weights, actions, temperature=1.0):
    """
    Differentiable Boltzmann policy using Gumbel-Softmax reparameterization.

    Forward pass: hard argmax (same behavior as original)
    Backward pass: softmax gradient through Gumbel-Softmax

    Args:
        temperature: controls discreteness (low T → closer to argmax)
    """
    # Evaluate all actions (same as original)
    results = vmap(lambda a: boltzmann_policy_fwd_cost(a, P, v_swim, theta,
                                                        get_u_env, get_S_ij, weights))(actions)
    costs, _ = results[0], results[1]
    beta_state = results[1][0]

    logits = jnp.nan_to_num(-beta_state * costs, nan=-1e10)
    logits = jnp.clip(logits, -1e10, 1e10)

  
    gumbel_key = random.fold_in(key, 42)
    gumbel_noise = sample_gumbel(gumbel_key, logits.shape, logits.dtype)

  
    y_soft = jax.nn.softmax((logits + gumbel_noise) / temperature)

  
    hard_idx = jnp.argmax(y_soft)
    y_hard = jax.nn.one_hot(hard_idx, len(actions), dtype=logits.dtype)

    
    y_ste = jax.lax.stop_gradient(y_hard - y_soft) + y_soft

    
    action_selected = actions[hard_idx]
    return action_selected, y_ste, y_soft, costs, beta_state

@partial(jit, static_argnums=(4, 5))
def boltzmann_policy_fwd_cost(action, P_curr, v_swim_curr, theta_curr,
                               get_u_env, get_S_ij, weights):
    """Compute cost for a single action (same as original total_cost_and_beta)."""
    from inversion_dynamic_core_jax import total_cost_and_beta
    return total_cost_and_beta(action, P_curr, v_swim_curr, theta_curr,
                               get_u_env, get_S_ij, weights)



def single_fish_simulation_diff(key, weights, get_u_env, get_S_ij,
                                 actions, P_init, v_init, theta_init, T_max,
                                 temperature=1.0):
    """Single fish simulation with Gumbel-Softmax (supports gradients)."""
    keys = random.split(key, T_max)

    traj_P = []
    traj_v = []
    traj_probs = []  

    P, v_swim, theta = P_init, v_init, theta_init

    for t in range(T_max):
        action, y_ste, y_soft, costs, beta = boltzmann_policy_gumbel_softmax(
            keys[t], P, v_swim, theta, get_u_env, get_S_ij,
            weights, actions, temperature)

        v_new, d_theta = action[0], action[1]
        theta_new = theta + d_theta

        u_env = get_u_env(P)
        v_abs_x = v_new * jnp.cos(theta_new) + u_env[0]
        v_abs_y = v_new * jnp.sin(theta_new) + u_env[1]
        P_new = P + jnp.array([v_abs_x, v_abs_y]) * PHYS_PARAMS["Delta_t"]
        P_new = jnp.clip(P_new, jnp.array([0.0, 0.0]), jnp.array([0.7, 0.2]))

        traj_P.append(P)
        traj_v.append(v_swim)
        traj_probs.append(y_soft)

        P, v_swim, theta = P_new, v_new, theta_new

    return (jnp.stack(traj_P), jnp.stack(traj_v),
            jnp.stack(traj_probs))



@jit
def wasserstein1d_jax(p_sim, p_exp, v_centers):
    """
    Differentiable 1D Wasserstein-1 distance.
    W₁ = ∫|CDF_sim(v) - CDF_exp(v)| dv

    Args:
        p_sim: simulated velocity PMF [n_bins]
        p_exp: experimental velocity PMF [n_bins]
        v_centers: bin centers [n_bins]
    """
    cdf_sim = jnp.cumsum(jnp.abs(p_sim) / (jnp.sum(jnp.abs(p_sim)) + 1e-10))
    cdf_exp = jnp.cumsum(jnp.abs(p_exp) / (jnp.sum(jnp.abs(p_exp)) + 1e-10))
    # Trapezoidal integration
    diff = jnp.abs(cdf_sim - cdf_exp)
    w1 = jnp.trapz(diff, v_centers)
    return w1

@jit
def tvd_loss(p_sim, p_exp):
    """Total Variation Distance (differentiable)."""
    p_sim_norm = jnp.abs(p_sim) / (jnp.sum(jnp.abs(p_sim)) + 1e-10)
    p_exp_norm = jnp.abs(p_exp) / (jnp.sum(jnp.abs(p_exp)) + 1e-10)
    return 0.5 * jnp.sum(jnp.abs(p_sim_norm - p_exp_norm))



def compute_differentiable_loss(weights_log, keys, get_u_env, get_S_ij,
                                 actions, exp_spatial_flat, exp_vel,
                                 v_edges, x_edges, y_edges,
                                 n_fish, t_max, burn_in, temperature):
    """
    End-to-end differentiable loss: weights → simulation → PDF → W₁ + TVD.

    Args:
        weights_log: unconstrained log-weights [wE_log, wR_log, wA_log]
        (other args as in original)

    Returns:
        scalar loss = W₁(velocity) + TVD(spatial)
    """
   
    weights_raw = jnp.exp(weights_log)
    weights = weights_raw / jnp.sum(weights_raw)

   
    fish_keys = random.split(keys[0], n_fish)
    P_init = jnp.column_stack([
        jnp.array(np.random.uniform(0.12, 0.22, n_fish)),
        jnp.array(np.random.uniform(0.06, 0.14, n_fish))
    ])
    v_init = jnp.full((n_fish,), 0.2)
    theta_init = jnp.full((n_fish,), jnp.pi)

    
    traj_P_all, traj_v_all, _ = vmap(
        lambda k, pi, vi, ti: single_fish_simulation_diff(
            k, weights, get_u_env, get_S_ij, actions, pi, vi, ti, t_max, temperature),
        in_axes=(0, 0, 0, 0)
    )(fish_keys, P_init, v_init, theta_init)

    
    traj_P_burn = traj_P_all[:, burn_in:, :].reshape(-1, 2)
    traj_v_burn = traj_v_all[:, burn_in:].reshape(-1)

    
    x_mask = (traj_P_burn[:, 0] >= x_edges[0]) & (traj_P_burn[:, 0] <= x_edges[-1])
    y_mask = (traj_P_burn[:, 1] >= y_edges[0]) & (traj_P_burn[:, 1] <= y_edges[-1])
    valid = x_mask & y_mask

    valid_P = traj_P_burn[valid]
    valid_v = traj_v_burn[valid]

    
    sim_vel_pdf = jnp.zeros(len(v_edges) - 1)
    for i in range(len(v_edges) - 1):
        in_bin = (valid_v >= v_edges[i]) & (valid_v < v_edges[i+1])
        sim_vel_pdf = sim_vel_pdf.at[i].set(jnp.sum(in_bin))

    sim_vel_pdf = sim_vel_pdf / (jnp.sum(sim_vel_pdf) + 1e-10)

    
    n_x, n_y = len(x_edges) - 1, len(y_edges) - 1
    sim_spatial_pdf = jnp.zeros((n_x, n_y))
    for i in range(n_x):
        x_in = (valid_P[:, 0] >= x_edges[i]) & (valid_P[:, 0] < x_edges[i+1])
        for j in range(n_y):
            y_in = (valid_P[:, 1] >= y_edges[j]) & (valid_P[:, 1] < y_edges[j+1])
            count = jnp.sum(x_in & y_in)
            sim_spatial_pdf = sim_spatial_pdf.at[i, j].set(count)

    sim_spatial_pdf = sim_spatial_pdf / (jnp.sum(sim_spatial_pdf) + 1e-10)
    sim_spatial_flat = sim_spatial_pdf.reshape(-1)

    
    v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])
    w1 = wasserstein1d_jax(sim_vel_pdf, exp_vel, v_centers)
    tvd = tvd_loss(sim_spatial_flat, exp_spatial_flat)

    return w1 + tvd, weights


@partial(jit, static_argnums=(4, 5, 6, 7, 8))
def compute_loss_single_speed(weights_log, key, get_u_env, get_S_ij,
                               actions, exp_vel, v_edges,
                               n_fish, t_max, burn_in):
    """
    Simplified differentiable loss for ONE flow speed.
    Uses JIT-compatible Gumbel-Softmax simulation.

    Returns: (W₁_loss, normalized_weights)
    """
    weights_raw = jnp.exp(weights_log)
    weights = weights_raw / jnp.sum(weights_raw)

    
    fish_keys = random.split(key, n_fish)
    P_init = jnp.column_stack([
        jax.random.uniform(random.fold_in(key, 100), (n_fish,), minval=0.12, maxval=0.22),
        jax.random.uniform(random.fold_in(key, 200), (n_fish,), minval=0.06, maxval=0.14)
    ])
    v_init = jnp.full((n_fish,), 0.2)
    theta_init = jnp.full((n_fish,), jnp.pi)

    def run_one_fish(fish_key, pi, vi, ti):
        keys_step = random.split(fish_key, t_max)
        traj_P = []
        traj_v = []
        P, v_swim, theta = pi, vi, ti

        for t in range(t_max):
            # Evaluate costs
            def eval_cost(a):
                from inversion_dynamic_core_jax import total_cost_and_beta
                J, _ = total_cost_and_beta(a, P, v_swim, theta,
                                           get_u_env, get_S_ij, weights)
                return J
            costs = vmap(eval_cost)(actions)

            _, betas = vmap(lambda a: total_cost_and_beta_helper(
                a, P, v_swim, theta, get_u_env, get_S_ij, weights))(actions)
            beta_state = betas[0]
            logits = jnp.nan_to_num(-beta_state * costs, nan=-1e8)

            # Gumbel-Softmax STE
            gumbel = sample_gumbel(keys_step[t], logits.shape)
            y_soft = jax.nn.softmax((logits + gumbel) / jnp.array(1.0))
            hard_idx = jnp.argmax(y_soft)
            action = actions[hard_idx]

            v_new, d_theta = action[0], action[1]
            theta_new = theta + d_theta
            u_env = get_u_env(P)
            P_new = P + jnp.array([v_new * jnp.cos(theta_new) + u_env[0],
                                   v_new * jnp.sin(theta_new) + u_env[1]]) * PHYS_PARAMS["Delta_t"]
            P_new = jnp.clip(P_new, jnp.array([0.0, 0.0]), jnp.array([0.7, 0.2]))

            traj_P.append(P)
            traj_v.append(v_swim)
            P, v_swim, theta = P_new, v_new, theta_new

        return jnp.stack(traj_P), jnp.stack(traj_v)

    traj_P_all, traj_v_all = vmap(run_one_fish)(fish_keys, P_init, v_init, theta_init)

    
    valid_v = traj_v_all[:, burn_in:].reshape(-1)
    n_bins = len(v_edges) - 1
    sim_vel_pdf = jnp.zeros(n_bins)
    for i in range(n_bins):
        sim_vel_pdf = sim_vel_pdf.at[i].set(
            jnp.sum((valid_v >= v_edges[i]) & (valid_v < v_edges[i+1])))
    sim_vel_pdf = sim_vel_pdf / (jnp.sum(sim_vel_pdf) + 1e-10)

    
    v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])
    cdf_sim = jnp.cumsum(sim_vel_pdf)
    cdf_exp = jnp.cumsum(exp_vel)
    w1 = jnp.trapz(jnp.abs(cdf_sim - cdf_exp), v_centers)

    return w1, weights


@partial(jit, static_argnums=(4, 5))
def total_cost_and_beta_helper(action, P_curr, v_swim_curr, theta_curr,
                                get_u_env, get_S_ij, weights):
    """Wrapper for jit compatibility."""
    from inversion_dynamic_core_jax import total_cost_and_beta as _fn
    return _fn(action, P_curr, v_swim_curr, theta_curr, get_u_env, get_S_ij, weights)


print("✅ inversion_dynamic_core_jax_diff.py loaded.")
print("   - Gumbel-Softmax Boltzmann policy (differentiable)")
print("   - Differentiable W₁ loss (JAX)")
print("   - compute_loss_single_speed (JIT-compatible)")
