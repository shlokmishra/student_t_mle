# Testing & Validation Notebooks

## Main notebooks

| Notebook | Purpose |
|----------|---------|
| **shlok.ipynb** | Gibbs vs KDE validation: trace/density comparison (k=1, m=100), variance sweep across (k, m). Confirms Gibbs ≈ KDE for m ≥ 20. |
| **kl_study.ipynb** | KL divergence study: how KL(Gibbs \|\| KDE) varies with m (5 to 100) for k=1 (Cauchy). |
| **posterior_analysis.ipynb** | Posterior variance table by (k, m), posterior predictive sampling, DP framing. |

## Other notebooks

- **x_kernel_diagnosis.ipynb** – x-transition kernel diagnostics (extreme values near z≈0)

## Modules

- `validation.py` – run_single_gibbs_kde, run_variance_sweep
- `analysis.py` – kl_divergence_estimate, run_kl_vs_m_study, posterior_predictive_samples

Run from project root or from `testing/` (path setup is automatic).
