# Diagnostic Workflow

This folder contains the active postprocessing and audit scripts for the sampler/KDE comparison.

## Current Pipeline

- `audit_reference_all_models.py` and `summarize_reference_all_models.py`: build and summarize canonical raw weighted-MC/KDE references.
- `audit_kde_correctness_all_models.py`: audit KDE backend correctness and dashboard-default rules.
- `audit_sampler_correctness.py`: audit Gibbs/RATTLE correctness diagnostics and verdicts.
- `analyze_efficiency.py`: summarize ESS, timing, and method winners from existing run outputs.
- `analyze_geometry.py`: summarize Gibbs/RATTLE geometry diagnostics from existing run outputs.
- `analyze_release_information.py`: Step 4 audit for MLE-release information loss and latent privacy leakage.
- `analyze_targeted_validation.py`: summarize targeted validation runsets.
- `refresh_analysis_from_runset.py`: refresh correctness/efficiency/geometry outputs from a named runset without overwriting baseline outputs.
- `reconcile_baseline_targeted.py`: reconcile production-length baseline runs with shorter targeted diagnostics.

Production sampler correctness command:

```bash
python reporting/diagnostics/audit_sampler_correctness.py \
  --runset-dir results/final_production_v1 \
  --out-dir results/sampler_correctness_audit
```

The script reads per-case files under `results/final_production_v1/case_*`, gates on
missing/failed cases, and writes both detailed diagnostics and
`final_sampler_verdict_table.csv` for downstream efficiency/geometry/dashboard steps.

Production efficiency command:

```bash
python reporting/diagnostics/analyze_efficiency.py \
  --runset-dir results/final_production_v1 \
  --correctness-dir results/sampler_correctness_audit \
  --out-dir results/efficiency_audit
```

Production geometry command:

```bash
python reporting/diagnostics/analyze_geometry.py \
  --runsets final_production_v1 \
  --correctness-dir results/sampler_correctness_audit \
  --efficiency-dir results/efficiency_audit \
  --out-dir results/geometry_audit
```

Efficiency uses the final production chains for posterior and timing summaries.
Geometry uses thinned diagnostics from the same runset for constraint, branch,
tail-state, RATTLE energy, and movement explanations.

Step 4 MLE-release information command:

```bash
python reporting/diagnostics/analyze_release_information.py \
  --mle-runset-dir results/final_production_v1 \
  --release-runset-dir results/release_information_runs \
  --out-dir results/release_information_audit
```

The release-information runset is optional until the Step 4 simulations exist.
When present, each `results/release_information_runs/case_*` directory may
contain `observed_data.csv`, `full_data_chain_samples.csv`,
`full_data_posterior_summaries.csv`, `mle_only_chain_samples.csv`,
`mle_only_posterior_summaries.csv`, and `latent_diagnostics.csv` or
`geometry_diagnostics.csv`. The script compares MLE-only Bayes to full-data
Bayes across matched simulated datasets and separately estimates latent
outlier-belief leakage from constrained latent diagnostics.

## Supporting Utilities

- `student_t_geometry.py`: shared Student-t geometry helpers used by tests and selected diagnostics.
- `diagnose_student_score_vs_mle.py`: focused Student k=1,n=10 diagnostic helper.
- `audit_kde_reference.py`, `summarize_kde_reference_audit.py`, `check_kde_reference_step1.py`: older/reference-cache utilities. Their reusable MLE sample cache is kept under `reporting/diagnostic_outputs/kde_reference_audit/mle_sample_cache/`.
- `run_rattle_tuning_grid.py`, `run_rattle_long_selected_configs.py`, `summarize_rattle_against_references.py`, `compare_latent_geometry.py`, `audit_rattle_target.py`: targeted RATTLE/geometry investigation helpers, not dashboard defaults.
