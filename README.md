# Gibbs for MLE (location models)

Simulate from the Bayesian posterior of theta (location) given the MLE for several location families. Two-step augmented Gibbs; verification via the KDE trick.

## Models

- **Student-t** (df k, scale 1)
- **Laplace** (scale b)
- **Logistic** (scale 1)

## Layout

- `models/` - active model implementations, Gibbs samplers, and RATTLE geometry code.
- `kde_ref/` - KDE/reference-posterior helpers.
- `diagnostics/` - shared run registry and cost-ledger utilities.
- `reporting/diagnostics/` - active audit and postprocessing scripts.
- `reporting/diagnostic_outputs/model_reference_audit/` - canonical raw weighted-MC/KDE reference artifacts.
- `reporting/diagnostic_outputs/kde_reference_audit/mle_sample_cache/` - reusable KDE MLE-sample cache.
- `dashboard/` - Streamlit dashboard app, pages, and dashboard-cache scripts.
- `scripts/` - active Grace/postprocessing entrypoints.
- `tests/` - lightweight correctness tests for model/reference conventions.
- `docs/presentation_notes.md` - compact notes for the eventual collaborator deck.
- `results/` - compact current audit outputs and fresh run directories.
- `_archive_repo_cleanup_20260608/` - reversible archive for old notebooks, legacy scripts, stale caches, NN experiments, and Python cache files.

The dashboard now lives entirely under `dashboard/`:

```bash
streamlit run dashboard/app.py
```

Generated dashboard cache files are written to `results/dashboard_cache/` when rebuilt.

## Setup

```bash
pip install -r requirements.txt
```
