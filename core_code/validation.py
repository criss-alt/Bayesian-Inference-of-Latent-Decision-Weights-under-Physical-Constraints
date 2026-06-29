"""
validation.py -- Model validation protocols.

Implements Methods Section 2.4:
  2.4.1  LOOCV cross-validation for out-of-sample generalization
  2.4.2  Global cost landscape closure test
"""

import numpy as np
from utils import (VELOCITIES, S_MAX, D, GAMMA, KAPPA, U_BURST,
                   load_flow_field, total_variation_distance,
                   wasserstein_1d, composite_distance)
from mdp_model import (energy_cost, risk_cost, anchoring_cost, integrated_cost)


def loocv_protocol(map_weights, data_dir="../03data",
                   train_velocities=None, test_velocities=None):
    """Leave-One-Out Cross-Validation (Methods Section 2.4.1).

    Trains ABC-SMC on train_velocities, evaluates MAP weights in
    zero-shot forward simulations at test_velocities.

    Parameters
    ----------
    map_weights : ndarray (3,)  MAP-inferred cognitive weights
    data_dir : str
    train_velocities : list     e.g., [0.10, 0.30, 0.50]
    test_velocities : list      e.g., [0.20, 0.40]

    Returns
    -------
    results : dict  {test_U: {'W1': float, 'TVD': float}}
    """
    if train_velocities is None:
        train_velocities = [0.10, 0.30, 0.50]
    if test_velocities is None:
        test_velocities = [0.20, 0.40]

    # Load experimental PDFs for comparison (placeholder)
    # In production: run forward MDP simulations at test velocities
    # using the MAP weights, then compare against experimental targets.

    results = {}
    for U_test in test_velocities:
        # Placeholder: simulate trajectories and compute distances
        flow = load_flow_field(U_test, data_dir)
        # ... forward simulation with map_weights ...
        # ... compute TVD and W1 against experimental PDFs ...
        results[U_test] = {'W1': None, 'TVD': None}

    return results


def compute_cost_landscape(weights, flow_field, U_inf,
                           cylinder_center=(0.20, 0.10)):
    """Eq. (15): Eulerian global cost landscape.

    Projects MAP weights onto the static LBM grid to construct the
    theoretical expected cost field J_global(x, y).

    Parameters
    ----------
    weights : ndarray (3,)   (w_E, w_R, w_A)
    flow_field : tuple       (x, y, u_mean, v_mean, S_mean)
    U_inf : float
    cylinder_center : tuple  (cx, cy)

    Returns
    -------
    C_E : ndarray (ny, nx)
    C_R : ndarray (ny, nx)
    C_A : ndarray (ny, nx)
    J_global : ndarray (ny, nx)
    """
    x_grid, y_grid, u_field, v_field, S_field = flow_field
    w_E, w_R, w_A = weights
    nx, ny = len(x_grid), len(y_grid)

    # Velocity magnitude
    v_mag = np.sqrt(u_field**2 + v_field**2)

    # C_E: normalized energy cost (stationary holding: v_swim = U_inf opposing flow)
    C_E = np.minimum((v_mag / (0.5 * U_BURST)) ** 1.5, 1.0)

    # C_R: exponential shear penalty (Eq. 5)
    S_clipped = np.minimum(S_field, S_MAX)
    C_R = (np.exp(KAPPA * S_clipped / S_MAX) - 1.0) / (np.exp(KAPPA) - 1.0)

    # C_A: displacement from target + rheotactic posture
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    dist_from_target = np.sqrt((X - 0.17)**2 + (Y - 0.10)**2)
    C_A = np.minimum(dist_from_target / 0.3, 1.0)
    # Down-weight near cylinder boundary
    near_cyl = np.sqrt((X - cylinder_center[0])**2 +
                        (Y - cylinder_center[1])**2) < D
    C_A[near_cyl] = 1.0

    # Global cost landscape
    J_global = w_E * C_E + w_R * C_R + w_A * C_A

    return C_E, C_R, C_A, J_global


def closure_test(weights, flow_field, U_inf, exp_spatial_pdf):
    """Cost landscape closure test (Methods Section 2.4.2).

    Verifies that argmin(J_global) coincides with the peak of the
    experimental spatial occupancy PDF.

    Parameters
    ----------
    weights : ndarray (3,)
    flow_field : tuple
    U_inf : float
    exp_spatial_pdf : ndarray (ny, nx)

    Returns
    -------
    closure : dict
        'j_min_position': (x, y) of global cost minimum
        'exp_peak_position': (x, y) of experimental occupancy peak
        'spatial_error': Euclidean distance between them (m)
        'is_closed': bool  True if error < grid resolution tolerance
    """
    x_grid, y_grid, _, _, _ = flow_field
    C_E, C_R, C_A, J_global = compute_cost_landscape(
        weights, flow_field, U_inf)

    # J_global minimum
    min_idx = np.unravel_index(np.argmin(J_global), J_global.shape)
    j_min_x = x_grid[min_idx[0]]
    j_min_y = y_grid[min_idx[1]]

    # Experimental peak
    exp_peak_idx = np.unravel_index(np.argmax(exp_spatial_pdf),
                                     exp_spatial_pdf.shape)
    exp_peak_x = x_grid[exp_peak_idx[0]]
    exp_peak_y = y_grid[exp_peak_idx[1]]

    spatial_error = np.sqrt((j_min_x - exp_peak_x)**2 +
                             (j_min_y - exp_peak_y)**2)

    # Grid resolution: ~0.01 m (typical LBM grid spacing)
    is_closed = spatial_error < 0.02

    return {
        'j_min_position': (j_min_x, j_min_y),
        'exp_peak_position': (exp_peak_x, exp_peak_y),
        'spatial_error': spatial_error,
        'is_closed': is_closed,
    }


def compute_cognitive_degradation(flow_fields, U_list):
    """Compute SNR, beta, and R_sense statistics across velocities.

    Parameters
    ----------
    flow_fields : list of tuples
    U_list : list of float

    Returns
    -------
    stats : dict
    """
    from mdp_model import (compute_snr, beta_rationality, compute_sensory_horizon)
    import numpy as np

    n_velocities = len(U_list)
    snr_medians = np.zeros(n_velocities)
    beta_medians = np.zeros(n_velocities)
    rsense_medians = np.zeros(n_velocities)
    exceedance_fracs = np.zeros(n_velocities)

    for i, (flow, U) in enumerate(zip(flow_fields, U_list)):
        _, _, _, _, S_field = flow
        S_flat = S_field.ravel()
        S_flat = S_flat[S_flat > 0]  # exclude cylinder interior

        # SNR sampled across domain
        snr_samples = compute_snr(U_BURST * 0.3, np.clip(S_flat, 1e-6, None))
        snr_medians[i] = np.median(snr_samples)
        beta_medians[i] = np.median(beta_rationality(snr_samples))
        rsense_medians[i] = np.median(compute_sensory_horizon(snr_samples))
        exceedance_fracs[i] = np.mean(S_flat > S_MAX)

    return {
        'velocities': np.array(U_list),
        'SNR_median': snr_medians,
        'beta_median': beta_medians,
        'R_sense_median': rsense_medians,
        'S_ij_exceedance_frac': exceedance_fracs,
    }
