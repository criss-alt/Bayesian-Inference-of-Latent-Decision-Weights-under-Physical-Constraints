"""
utils.py -- Shared utilities for data I/O, PDF construction, and distance metrics.

All functions use pure NumPy/SciPy. No external data is hardcoded.
"""

import numpy as np
from scipy.stats import wasserstein_distance
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# Physical constants (manuscript Table of Parameters)
# ---------------------------------------------------------------------------
D =               # Cylinder diameter (m)
L_REF =           # Fish body length (m)
U_BURST =           # Maximum burst speed (m/s)
S_MAX =            # Shear instability threshold (s^-1)
VELOCITIES = []  # Inflow speeds (m/s)

# MDP hyperparameters (Methods Section 2.2)
ETA =               # Turning resistance coefficient
KAPPA =             # Risk kurtosis
GAMMA =             # Anchoring balance: displacement vs. posture
BETA_MAX =         # Maximum decision rationality
BETA_MIN =          # Minimum decision rationality (panic floor)
LAMBDA_SNR =       # Sensory decay rate
R_MAX =     # Maximum perceptual range (m)

# Gradient optimization (Methods Section 2.3.2)
LR = 
BETA1, BETA2 = 
GRAD_CLIP = 
N_EPOCHS = 

# ABC-SMC (Methods Section 2.3.1)
N_PARTICLES = 
N_GENERATIONS = 
EPS_INIT = 
EPS_FLOOR = 

# Simulation protocol (Supplementary Text S7)
N_FISH = 
N_STEPS = 
BURN_IN = 
SUBSAMPLE = 

# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------
def load_flow_field(velocity, data_dir="../03data"):
    """Load time-averaged LBM flow field for a given inflow velocity.

    Parameters
    ----------
    velocity : float
        Inflow velocity in m/s (one of {0.10, 0.20, 0.30, 0.40, 0.50}).
    data_dir : str
        Path to the data directory.

    Returns
    -------
    x : ndarray (nx,)
    y : ndarray (ny,)
    u_mean : ndarray (ny, nx)
    v_mean : ndarray (ny, nx)
    S_mean : ndarray (ny, nx)
    """
    import os
    fname = os.path.join(data_dir, "flow_fields",
                         f"flow_{int(velocity*100):02d}.npz")
    data = np.load(fname)
    return (data['x'], data['y'], data['u_mean'],
            data['v_mean'], data['S_mean'])


def load_experimental_pdfs(data_dir="../03data"):
    """Load experimental spatial and velocity PDFs.

    Returns
    -------
    spatial_pdfs : dict  {velocity_index: 2D ndarray}
    velocity_pdfs : dict {velocity_index: 2D ndarray}
    v_bins : ndarray
    x_edges, y_edges : ndarray
    """
    import os
    sp = np.load(os.path.join(data_dir, "experimental", "spatial_pdf.npz"))
    vp = np.load(os.path.join(data_dir, "experimental", "velocity_pdf.npz"))

    spatial_pdfs = {}
    for U in VELOCITIES:
        key = f"pdf_{int(U*100):02d}"
        spatial_pdfs[U] = sp[key]

    velocity_pdfs = {}
    for U in VELOCITIES:
        key = f"pdf_{int(U*100):02d}"
        velocity_pdfs[U] = vp[key]

    return (spatial_pdfs, velocity_pdfs, vp['v_bins'],
            sp['x_edges'], sp['y_edges'])


# ---------------------------------------------------------------------------
# PDF construction from trajectory ensembles
# ---------------------------------------------------------------------------
def build_spatial_histogram(trajectories, x_edges, y_edges):
    """Build 2D occupancy histogram from trajectory ensemble.

    Parameters
    ----------
    trajectories : list of ndarray (T_i, 2)
        List of (x, y) position arrays, one per fish.
    x_edges, y_edges : ndarray
        Bin edges for spatial grid.

    Returns
    -------
    pdf : ndarray (ny-1, nx-1) normalized to unit sum.
    """
    all_positions = np.vstack([t for t in trajectories if len(t) > 0])
    hist, _, _ = np.histogram2d(all_positions[:, 0], all_positions[:, 1],
                                 bins=[x_edges, y_edges])
    pdf = hist / hist.sum()
    return pdf.T  # (ny-1, nx-1)


def build_velocity_pdf(trajectories, v_bins):
    """Build 1D relative-velocity PDF from trajectory ensemble.

    Parameters
    ----------
    trajectories : list of ndarray
        List of (v_rel,) arrays, one per fish.
    v_bins : ndarray
        Velocity bin centers.

    Returns
    -------
    pdf : ndarray normalized to unit integral.
    """
    all_v = np.concatenate([t for t in trajectories if len(t) > 0])
    hist, _ = np.histogram(all_v, bins=v_bins)
    pdf = hist.astype(float)
    pdf /= (pdf.sum() * (v_bins[1] - v_bins[0]))
    return pdf


# ---------------------------------------------------------------------------
# Distance metrics (Methods Eq. 13)
# ---------------------------------------------------------------------------
def total_variation_distance(p_sim, p_exp):
    """TVD between two spatial PDFs (2D histograms)."""
    return 0.5 * np.sum(np.abs(p_sim - p_exp))


def wasserstein_1d(p_sim, p_exp, bins):
    """1-Wasserstein distance between two 1D velocity distributions.

    Uses empirical CDF representation for stability when supports differ.
    """
    # Normalize to probability mass functions
    p_sim_norm = p_sim / p_sim.sum()
    p_exp_norm = p_exp / p_exp.sum()
    return wasserstein_distance(bins, bins, p_sim_norm, p_exp_norm)


def composite_distance(p_sim, p_exp, v_sim, v_exp, v_bins):
    """Composite distance (Eq. 13): TVD(spatial) + W1(velocity)."""
    tvd = total_variation_distance(p_sim, p_exp)
    w1 = wasserstein_1d(v_sim, v_exp, v_bins)
    return tvd + w1
