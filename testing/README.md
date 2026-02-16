# Testing & Validation Notebooks

## Main notebooks

| Notebook | Purpose |
|----------|---------|
| **shlok.ipynb** | Gibbs vs KDE validation: trace/density comparison (k=1, m=100), variance sweep across (k, m). Confirms Gibbs ≈ KDE for m ≥ 20. |
| **kl_study.ipynb** | KL divergence study: KL(Gibbs \|\| KDE) vs m for k=1 (Cauchy), plus multi-k comparison (k=1,2,3,5). |
| **posterior_analysis.ipynb** | Posterior variance table, information loss ratio Var(μ\|μ*)/Var(μ\|x), posterior predictive, DP framing. |

## Other notebooks

- **x_kernel_diagnosis.ipynb** – x-transition kernel diagnostics (extreme values near z≈0)

## Modules

- `validation.py` – run_single_gibbs_kde, run_variance_sweep
- `analysis.py` – kl_divergence_estimate, run_kl_vs_m_study, run_kl_vs_m_multi_k, run_info_loss_sweep, posterior_predictive_samples

Run from project root or from `testing/` (path setup is automatic).
