# Core Code: Cognitive MDP + ABC-SMC + Gradient Optimization

Python implementation of the inference framework described in:

> *Fish Station-Keeping in Bluff-Body Wakes Is Governed by a Safety-First Principle: Bayesian Inference of Latent Decision Weights under Physical Constraints*

## Files

| File | Content | Manuscript Section |
|------|---------|-------------------|
| `utils.py` | Constants, data I/O, PDF construction, distance metrics | -- |
| `mdp_model.py` | Cost functions (C_E, C_R, C_A, J), SNR-gating, Boltzmann policy, Gumbel-Softmax STE | Methods 2.2 |
| `abc_smc.py` | ABC-SMC inference: prior, composite distance, particle propagation, MAP/phenotype extraction | Methods 2.3.1 |
| `gradient_opt.py` | Surrogate loss, Adam optimizer in log-weight space | Methods 2.3.2 |
| `validation.py` | LOOCV protocol, cost landscape closure test, cognitive degradation | Methods 2.4 |
| `run_inference.py` | Master pipeline script | -- |

## Dependencies

```
numpy >= 1.24
scipy >= 1.10
```



## Usage

```bash
# Full pipeline (requires pre-computed posterior or runs ABC-SMC)
python run_inference.py --data_dir ../03data --seed 42

# Individual modules
python -c "from mdp_model import *; print(compute_snr_phys_crit())"
python -c "from abc_smc import sample_simplex_prior; print(sample_simplex_prior(5))"
```

## Key Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| S_max | 42 s^-1 | Eq. (1), experimental calibration |
| U_burst | 1.1 m/s | Sprint trials (Supplementary Text S2) |
| kappa | 4.0 | Weber-Fechner risk kurtosis |
| gamma | 0.7 | Anchoring balance coefficient |
| lambda_snr | 0.5 | Sensory decay rate |
| SNR_phys_crit | 0.21 | Eq. (9), absolute biophysical limit |
| SNR_beta_half | 1.39 | Cognitive half-activation threshold |

## Notes

- The LBM flow solver is not included; this package assumes pre-computed flow fields.
- ABC-SMC full trajectory inference requires significant computation (~20,000 forward simulations). The skeleton in `abc_smc.py` shows the workflow; production use should couple with `mdp_model.rollout_trajectory()`.
- The gradient optimization module uses finite-difference gradients for simplicity. Production use should employ automatic differentiation (JAX).
- All random seeds are configurable for reproducibility.
