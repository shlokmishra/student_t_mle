# Diagnostic Workflow

This folder contains the active diagnostics used for the Student-t MLE-conditional posterior audit. Keep this workflow focused: raw weighted-MC moments are the primary reference candidate for posterior moments of `mu`; KDE rows are sensitivity checks, not a reference standard.

## Active Scripts

- `audit_kde_reference.py`: simulate centered MLE errors and compute raw weighted-MC posterior moments; optionally add KDE grid/quad sensitivity rows.
- `summarize_kde_reference_audit.py`: summarize raw weighted-MC, KDE backend, and integration sensitivity.
- `run_rattle_tuning_grid.py`: run compact RATTLE grids with optional `--mu-star-reference-csv`.
- `run_rattle_long_selected_configs.py`: run longer selected Student-2 configs with batch-means uncertainty for posterior mean and variance.
- `summarize_rattle_against_references.py`: compare RATTLE rows primarily against raw weighted-MC moments, with optional Gibbs/KDE context.
- `compare_latent_geometry.py`: compare Gibbs and RATTLE latent states via Gram, tail, branch, and constraint-residual diagnostics.
- `audit_rattle_target.py`: inspect the Student-t constraint, Gram term, and potential convention for a concrete state.
- `student_t_geometry.py`: shared helper functions for Student-t geometry summaries.

## Current Reproducible Path

1. Build or refresh raw weighted-MC references with `audit_kde_reference.py`.
2. Run compact RATTLE grids or selected long runs using the same observed `mu_star` values via `--mu-star-reference-csv`.
3. Summarize RATTLE against raw weighted-MC moments using `summarize_rattle_against_references.py`.
4. Use `compare_latent_geometry.py` only for selected configs, not as a broad tuning campaign.

See `reports/student2_rattle_audit_summary.md` for the current Student-2 and exploratory Cauchy conclusions.
