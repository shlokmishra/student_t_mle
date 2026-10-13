# Shared diagnostics

- `cost_ledger.py` records sampler work, projection failures and timing. Existing
  model kernels import it when cost collection is enabled.
- `source_provenance.py` verifies historical source hashes against the explicit
  model-layout migration. Current selected-MLE reference validation uses it.

These are live shared helpers. The old `run_registry.py` and its obsolete
runset/configuration machinery have been retired. Mathematical geometry tools
and model-specific diagnostics live under `reporting/diagnostics/`.
