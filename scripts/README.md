# Current experiment and validation drivers

The old dashboard, production-grid, timing-grid, repair/package, profile and
Slurm submission drivers have been removed. This directory retains:

- `run_checkpoint_validation.py`: bounded tests, both examples and current CLI
  smoke checks. It does not launch an experiment campaign.
- `compare_location_only_performance.py`: frozen ten-case location comparison,
  saved parameter/latent chains, raw reference checks and cost accounting.
  Replaying sampling requires new paths in the frozen configuration; `--analyze`
  reuses saved results. See the corresponding performance report first.
- `analyze_location_chains.py`: reads saved chains, verifies hashes, and produces
  labelled/ordered latent diagnostics, quantiles and the trace gallery. No sampling.
- `build_location_review_figures.py`: two compact Student n=10 comparison figures
  for the collaborator note. Reads saved evidence, checks hashes/annotations and
  writes PNG/PDF plus provenance to a new `--out` directory. No sampler imports.
- `validate_location_only_correctness.py`: two 512-step dispersed chains per
  method across ten Student/Logistic/Laplace cases; independent references and
  numerical failure records. This is a correctness smoke study, not a mixing or
  speed ranking. Requires a new `--output` directory.
- `validate_selected_mle_reference.py`, `followup_shifted_student_reference.py`:
  independent reference construction, uncertainty and shifted-case diagnostics.
- `validate_student_selected_mle_gibbs.py`, `validate_cheap_selected_mle_gibbs.py`,
  `validate_student_selected_mle_rattle.py`, `validate_student_independence_mh.py`:
  current sampler validation and reference comparisons.
- `validate_student_mle_hardening.py`, `selected_mle_reference_oracle.py`:
  hardened-selector validation and independent rational oracle.
- `validate_student_n3_fast_path.py`, `audit_student_n3_fast_path.py`:
  guarded endpoint certification and matched-chain checks.
- `study_student_n3_selected_geometry.py`, `study_student_higher_n_geometry.py`,
  `study_student_n5_boundary.py`, `verify_student_n5_irreducibility.py`:
  reproducible geometry/proof calculations.
- `verify_student_independence_evidence.py`: source and raw-array integrity/oracle
  checks for the saved independence study.
- `validate_laplace_facet_rattle*.py`: retained comparator/prototype validation.
- `study_student_latent_esjd.py`, `tune_student_acceptance.py`: legacy comparator
  drivers whose movement/selection helpers are still imported by focused tests.
  Their names do not designate corrected selected-MLE tuning.
- `grace_setup_env.sh`: environment setup convenience on Grace; no job grid.

Current frozen configurations live in `docs/research/evidence/*/config.json`.
Consult the corresponding report before replay: several drivers deliberately
refuse to overwrite saved runs, and new runs need fresh destinations. Merely
running `--help` or checkpoint validation does not establish new sampling results.
