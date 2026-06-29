"""
gradient_opt.py -- Gradient-based surrogate optimization for cognitive weights.

Implements Methods Section 2.3.2:
  - Surrogate loss L(w) = expected single-step velocity deviation
  - Adam optimizer with log-weight-space mapping
  - Gumbel-Softmax STE integration

The surrogate loss is structurally blind to rare tail events;
this module is intended as a methodological comparison against ABC-SMC.
"""

import numpy as np
from utils import (U_BURST, S_MAX, LR, BETA1, BETA2, GRAD_CLIP, N_EPOCHS,
                   load_flow_field)
from mdp_model import (build_action_space, integrated_cost,
                        boltzmann_probabilities, beta_rationality,
                        compute_snr, collision_penalty)


def log_weight_to_simplex(z):
    """Map unconstrained log-weights to 2-simplex via softmax.

    Methods Section 2.3.2 / Supplementary Text S6.
    """
    exp_z = np.exp(z - z.max())
    return exp_z / exp_z.sum()


def surrogate_loss(weights, flow_field, U_inf, v_target,
                    n_positions=400, rng=None):
    """Eq. (14): Expected single-step absolute velocity deviation.

    L(w) = E_P[(E[|v_abs| | P, w] - v_target)^2] + 0.1 * Var_P[...]

    Evaluated over n_positions uniformly sampled from the fluid domain.

    Parameters
    ----------
    weights : ndarray (3,)   Cognitive weights (w_E, w_R, w_A)
    flow_field : tuple       (x, y, u_mean, v_mean, S_mean)
    U_inf : float            Inflow velocity
    v_target : float         Target holding speed (0.5 * U_inf)
    n_positions : int        Number of spatial sampling positions
    rng : np.random.Generator

    Returns
    -------
    loss : float
    """
    if rng is None:
        rng = np.random.default_rng()

    x_grid, y_grid, u_field, v_field, S_field = flow_field
    nx, ny = len(x_grid), len(y_grid)

    actions = build_action_space()
    expected_velocities = np.zeros(n_positions)

    for i in range(n_positions):
        # Sample random position in domain
        ix = rng.integers(1, nx-2)
        iy = rng.integers(1, ny-2)
        S_local = S_field[iy, ix]
        u_local = u_field[iy, ix]
        v_local = v_field[iy, ix]
        u_env = np.array([u_local, v_local])

        # Evaluate expected |v_abs| under Boltzmann policy
        costs = np.zeros(len(actions))
        for j, (v_tgt, dth) in enumerate(actions):
            # Simplified: compute cost without full position update
            costs[j] = integrated_cost(v_tgt, dth, S_local,
                                        np.sqrt(v_tgt**2 + U_inf**2),
                                        U_inf, 0.0, tuple(weights))

        snr = compute_snr(U_BURST * 0.3, max(S_local, 1e-6))
        beta = beta_rationality(snr)
        probs = boltzmann_probabilities(costs, beta)

        # Expected absolute velocity
        v_abs_expected = np.sum(probs * actions[:, 0])  # weighted v_swim
        expected_velocities[i] = v_abs_expected

    # Eq. (14): squared bias + variance penalty
    bias_term = np.mean((expected_velocities - v_target) ** 2)
    var_term = np.var(expected_velocities)
    return bias_term + 0.1 * var_term


def adam_optimize(flow_field, U_inf, n_epochs=N_EPOCHS,
                  lr=LR, beta1=BETA1, beta2=BETA2,
                  grad_clip=GRAD_CLIP, n_restarts=5, seed=42):
    """Adam optimization in log-weight space (Methods Section 2.3.2).

    Parameters
    ----------
    flow_field : tuple    (x, y, u, v, S) for target velocity
    U_inf : float
    n_epochs : int
    lr, beta1, beta2 : float
    grad_clip : float     L2-norm clipping threshold
    n_restarts : int      Independent random initializations
    seed : int

    Returns
    -------
    best_weights : ndarray (3,)
    best_loss : float
    all_trajectories : list of (loss_history, weight_history)
    """
    rng = np.random.default_rng(seed)
    v_target = 0.5 * U_inf  # target holding speed

    best_weights = None
    best_loss = np.inf
    all_trajectories = []

    for restart in range(n_restarts):
        # Initialize in log-weight space
        z = rng.normal(0, 1, 3)
        w = log_weight_to_simplex(z)

        m = np.zeros(3)  # Adam first moment
        v = np.zeros(3)  # Adam second moment
        t = 0

        loss_history = []
        weight_history = [w.copy()]

        for epoch in range(n_epochs):
            t += 1

            # Finite-difference gradient (simplified; production uses autodiff)
            grad = np.zeros(3)
            eps_fd = 1e-4
            for dim in range(3):
                z_plus = z.copy()
                z_plus[dim] += eps_fd
                w_plus = log_weight_to_simplex(z_plus)

                z_minus = z.copy()
                z_minus[dim] -= eps_fd
                w_minus = log_weight_to_simplex(z_minus)

                loss_plus = surrogate_loss(w_plus, flow_field, U_inf, v_target, rng=rng)
                loss_minus = surrogate_loss(w_minus, flow_field, U_inf, v_target, rng=rng)
                grad[dim] = (loss_plus - loss_minus) / (2 * eps_fd)

            # Gradient clipping
            grad_norm = np.linalg.norm(grad)
            if grad_norm > grad_clip:
                grad *= grad_clip / grad_norm

            # Adam update
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            z -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

            w = log_weight_to_simplex(z)
            loss = surrogate_loss(w, flow_field, U_inf, v_target, rng=rng)
            loss_history.append(loss)
            weight_history.append(w.copy())

        all_trajectories.append((loss_history, weight_history))

        if loss_history[-1] < best_loss:
            best_loss = loss_history[-1]
            best_weights = weight_history[-1].copy()

    return best_weights, best_loss, all_trajectories
