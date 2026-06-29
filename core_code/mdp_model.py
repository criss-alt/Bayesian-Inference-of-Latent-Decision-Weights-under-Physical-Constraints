"""
mdp_model.py -- Cognitive MDP model for fish station-holding under hydrodynamic risk.

Implements Methods Section 2.2:
  2.2.1  Galilean kinematic mapping and state-action space
  2.2.2  Biomechanically grounded cost functional (C_E, C_R, C_A, J)
  2.2.3  SNR-gated cognitive phase transition (beta, alpha)
  2.2.4  Boltzmann exploration policy and Gumbel-Softmax reparameterization

All parameters from manuscript; no real data hardcoded.
"""

import numpy as np
from utils import (U_BURST, S_MAX, L_REF, ETA, KAPPA, GAMMA,
                   BETA_MAX, BETA_MIN, LAMBDA_SNR, R_MAX)

# ---------------------------------------------------------------------------
# 2.2.1  State and Action Space
# ---------------------------------------------------------------------------
def galilean_kinematic_update(position, v_swim_target, delta_theta,
                              heading, u_env):
    """Eq. (2): Galilean transformation for one MDP step.

    Parameters
    ----------
    position : ndarray (2,)        Current centroid (x, y) in m
    v_swim_target : float          Target relative swimming speed (m/s)
    delta_theta : float            Turning angle increment (rad)
    heading : float                Current heading angle (rad)
    u_env : ndarray (2,)           Local flow velocity (u, v) in m/s

    Returns
    -------
    new_position : ndarray (2,)
    new_heading : float
    v_abs : ndarray (2,)           Absolute ground velocity
    """
    new_heading = heading + delta_theta
    v_swim = np.array([v_swim_target * np.cos(new_heading),
                        v_swim_target * np.sin(new_heading)])
    v_abs = v_swim + u_env
    new_position = position + v_abs  # single-step Euler integration
    return new_position, new_heading, v_abs


def build_action_space(n_v=15, n_theta=31):
    """Discretize continuous action space into a Cartesian grid.

    Parameters
    ----------
    n_v : int     Number of speed bins in [0, U_BURST]
    n_theta : int Number of angle bins in [-DeltaTheta_max, +DeltaTheta_max]

    Returns
    -------
    actions : ndarray (n_v * n_theta, 2)  Each row: (v_swim_target, delta_theta)
    """
    v_grid = np.linspace(0, U_BURST, n_v)
    dt_max = 0.802  # rad, maximum sustained turning angle (Methods 2.2.1)
    theta_grid = np.linspace(-dt_max, dt_max, n_theta)
    VV, TT = np.meshgrid(v_grid, theta_grid)
    return np.column_stack([VV.ravel(), TT.ravel()])


# ---------------------------------------------------------------------------
# 2.2.2  Cost Functional (Eqs. 4-7)
# ---------------------------------------------------------------------------
def energy_cost(v_swim_target, delta_theta, dt_max=0.802, epsilon=1e-3):
    """Eq. (4): Normalized energy expenditure.

    Cubic thrust scaling + quadratic turning penalty + basal metabolism.
    Clamped to [0, 1] by min(..., 1.0).
    """
    v_norm = v_swim_target / U_BURST
    turn_norm = abs(delta_theta) / dt_max
    cost = (v_norm ** 3 +
            ETA * (v_norm ** 2) * turn_norm +
            epsilon * turn_norm)
    return min(cost, 1.0)


def risk_cost(S_local):
    """Eq. (5): Exponential shear-risk penalty.

    Anchored at S_max = 42 s^-1. Uses Weber-Fechner nonlinearity (kappa=4.0).
    """
    S_clipped = min(S_local, S_MAX)
    return (np.exp(KAPPA * S_clipped / S_MAX) - 1.0) / (np.exp(KAPPA) - 1.0)


def anchoring_cost(v_abs_target_mag, U_inf, theta_target):
    """Eq. (6): Displacement penalty + rheotactic posture reward.

    gamma=0.7 balances absolute position retention vs. upstream alignment.
    """
    disp_penalty = min(abs(v_abs_target_mag), U_inf) / U_inf
    posture_reward = (1.0 + np.cos(theta_target)) / 2.0
    return GAMMA * disp_penalty + (1.0 - GAMMA) * posture_reward


def integrated_cost(v_swim_target, delta_theta, S_local,
                    v_abs_target_mag, U_inf, theta_target,
                    weights, dt_max=0.802):
    """Eq. (7): Simplex-weighted sum of three sub-costs.

    Parameters
    ----------
    weights : tuple (w_E, w_R, w_A)  Sums to 1.
    """
    w_E, w_R, w_A = weights
    c_e = energy_cost(v_swim_target, delta_theta, dt_max)
    c_r = risk_cost(S_local)
    c_a = anchoring_cost(v_abs_target_mag, U_inf, theta_target)
    return w_E * c_e + w_R * c_r + w_A * c_a


def collision_penalty(position, cylinder_center=(0.20, 0.10),
                      r_exc=0.035, c_coll=1000.0):
    """Hard-wall collision check (Supplementary Text S4).

    Returns c_coll if inside the exclusion zone, else 0.
    """
    dist = np.sqrt((position[0] - cylinder_center[0])**2 +
                   (position[1] - cylinder_center[1])**2)
    return c_coll if dist < r_exc else 0.0


# ---------------------------------------------------------------------------
# 2.2.3  SNR-Gated Cognitive Phase Transition (Eqs. 8-11)
# ---------------------------------------------------------------------------
def compute_snr(v_swim_inst, S_local, eps_snr=1e-3):
    """Eq. (8): Instantaneous signal-to-noise ratio.

    SNR_inst = v_swim / (L_ref * |S_ij| + epsilon_snr)
    Uses the fish's instantaneous relative speed, not U_burst.
    """
    return v_swim_inst / (L_REF * S_local + eps_snr)


def compute_snr_phys_crit():
    """Eq. (9): Absolute biophysical survival boundary.

    SNR_phys_crit = U_burst / (L_ref * S_max) ~ 0.21
    """
    return U_BURST / (L_REF * S_MAX)


def beta_rationality(snr_inst):
    """Eq. (10): SNR-gated inverse temperature.

    beta -> BETA_MAX at high SNR (deterministic exploitation)
    beta -> BETA_MIN at low SNR  (heuristic panic exploration)
    """
    return BETA_MIN + (BETA_MAX - BETA_MIN) * (1.0 - np.exp(-LAMBDA_SNR * snr_inst))


def alpha_sensory(snr_inst):
    """Eq. (11): SNR-gated sensory fusion weight.

    Controls capacity to resolve far-field hydrodynamic structure.
    """
    return 1.0 - np.exp(-LAMBDA_SNR * snr_inst)


def compute_sensory_horizon(snr_inst):
    """Effective sensory radius: R_sense = alpha(SNR) * R_max."""
    return alpha_sensory(snr_inst) * R_MAX


def snr_beta_half():
    """Cognitive half-activation threshold: SNR such that beta = (BETA_MAX+BETA_MIN)/2.

    SNR_beta_half = ln(2) / lambda_snr ~ 1.39 (manuscript Section 2.2.3)
    """
    return np.log(2.0) / LAMBDA_SNR


# ---------------------------------------------------------------------------
# 2.2.4  Boltzmann Policy and Gumbel-Softmax STE
# ---------------------------------------------------------------------------
def boltzmann_probabilities(costs, beta):
    """Eq. (12): Softmax over action costs.

    Parameters
    ----------
    costs : ndarray (n_actions,)  Integrated cost J for each candidate action.
    beta  : float                 SNR-gated rationality.

    Returns
    -------
    probs : ndarray (n_actions,)  Probability mass over actions.
    """
    logits = -beta * costs
    logits -= logits.max()  # numerical stability
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum()


def gumbel_softmax_sample(log_probs, tau=1.0):
    """Gumbel-Softmax Straight-Through Estimator (Methods 2.2.4).

    Forward pass: discrete action via Gumbel-Max trick.
    Backward-compatible: returns both the discrete index and the
    softmax-relaxed probability vector for gradient computation.

    Parameters
    ----------
    log_probs : ndarray (n_actions,)  log(pi) for each action
    tau : float  Softmax temperature (fixed at 1.0 per manuscript)

    Returns
    -------
    action_idx : int                   Discrete action index (forward)
    softmax_probs : ndarray (n_actions,)  Relaxed probabilities (backward)
    """
    n = len(log_probs)
    gumbel_noise = -np.log(-np.log(np.random.uniform(1e-10, 1.0, n)))
    logits = (log_probs + gumbel_noise) / tau
    softmax_probs = np.exp(logits - logits.max())
    softmax_probs /= softmax_probs.sum()
    action_idx = np.argmax(log_probs + gumbel_noise)  # discrete forward
    return action_idx, softmax_probs


# ---------------------------------------------------------------------------
# MDP trajectory rollout
# ---------------------------------------------------------------------------
def rollout_trajectory(weights, flow_field, U_inf, n_steps=300,
                       dt=1.0, domain_bounds=(0.0, 0.7, 0.0, 0.2),
                       cylinder_center=(0.20, 0.10), rng=None):
    """Simulate one fish trajectory under the cognitive MDP.

    Parameters
    ----------
    weights : tuple (w_E, w_R, w_A)
    flow_field : tuple (x, y, u_mean, v_mean, S_mean)
    U_inf : float
    n_steps : int
    dt : float       Time-step scaling (1.0 = unit step)
    domain_bounds : tuple (xmin, xmax, ymin, ymax)
    cylinder_center : tuple (cx, cy)
    rng : np.random.Generator

    Returns
    -------
    positions : ndarray (n_steps, 2)
    v_abs_history : ndarray (n_steps, 2)
    snr_history : ndarray (n_steps,)
    beta_history : ndarray (n_steps,)
    """
    if rng is None:
        rng = np.random.default_rng()

    x_grid, y_grid, u_field, v_field, S_field = flow_field
    xmin, xmax, ymin, ymax = domain_bounds
    nx, ny = len(x_grid), len(y_grid)

    actions = build_action_space()
    positions = np.zeros((n_steps, 2))
    v_abs_hist = np.zeros((n_steps, 2))
    snr_hist = np.zeros(n_steps)
    beta_hist = np.zeros(n_steps)

    # Initialize: random position in upstream region
    pos = np.array([rng.uniform(0.14, 0.20),
                     rng.uniform(0.08, 0.12)])
    heading = 0.0

    for t in range(n_steps):
        # Interpolate local flow
        ix = np.searchsorted(x_grid, pos[0])
        iy = np.searchsorted(y_grid, pos[1])
        ix = np.clip(ix, 1, nx-2)
        iy = np.clip(iy, 1, ny-2)
        u_local = u_field[iy, ix]
        v_local = v_field[iy, ix]
        S_local = S_field[iy, ix]
        u_env = np.array([u_local, v_local])

        # Current SNR and beta
        v_swim_inst = U_BURST * 0.3  # representative instantaneous speed
        snr = compute_snr(v_swim_inst, max(S_local, 1e-6))
        beta = beta_rationality(snr)

        # Evaluate all actions
        costs = np.zeros(len(actions))
        for j, (v_tgt, dth) in enumerate(actions):
            new_pos, new_hd, v_abs = galilean_kinematic_update(
                pos, v_tgt, dth, heading, u_env)
            costs[j] = integrated_cost(v_tgt, dth, S_local,
                                       np.linalg.norm(v_abs), U_inf,
                                       new_hd, weights)
            costs[j] += collision_penalty(new_pos, cylinder_center)

        # Boltzmann selection
        if np.random.random() < 0.01:
            log_p = np.log(np.ones(len(actions)) / len(actions))
        else:
            log_p = np.log(boltzmann_probabilities(costs, beta) + 1e-12)
        action_idx, _ = gumbel_softmax_sample(log_p)

        v_tgt, dth = actions[action_idx]
        pos, heading, v_abs = galilean_kinematic_update(
            pos, v_tgt, dth, heading, u_env)

        # Domain clamping
        pos[0] = np.clip(pos[0], xmin + 0.01, xmax - 0.01)
        pos[1] = np.clip(pos[1], ymin + 0.01, ymax - 0.01)

        positions[t] = pos
        v_abs_hist[t] = v_abs
        snr_hist[t] = snr
        beta_hist[t] = beta

    return positions, v_abs_hist, snr_hist, beta_hist
