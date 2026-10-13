# Reference calculations: current status

This package is retained research code, not dashboard infrastructure. Its name
predates the current distinction between authoritative raw Monte Carlo and
supplementary smoothed density displays.

| Module | Role | Status |
|---|---|---|
| `moments.py` | Prior-weighted posterior summaries from centered selected-MLE errors; optional KDE summaries | Raw-weighted path tested. Gaussian-mixture helper now uses the full mixture CDF and log normalization. |
| `numba_kernels.py` | Compiled Student kernel densities and Normal log weights | Covered by independent density comparisons. |
| `posterior.py` | KDE backends, correct location shift `release - mu`, numerical normalization | Approximate density display; smoothing and finite integration bounds can change summaries. |
| `reference_adapter.py` | Reference-table/array adapters, simulation and cache convenience | Legacy cache validation and resume keys are insufficient for authoritative provenance. |

The canonical selected-MLE comparisons use fresh, hashed standardized data/error
banks from the retained October 10 studies and the October 12 location comparison,
with raw prior weights, ESS, uncertainty and selector provenance in
`docs/research/evidence/`. Use the exact paths and hashes in the
[performance study](../docs/research/location_only_performance_comparison.md)
and [handoff inventory](../docs/research/evidence/location_review_handoff_2026-10-12/handoff_inventory.csv).
Do not substitute an arbitrary historical KDE cache for those banks.

The two known `gaussian_kde_gaussian_prior_moments` defects are fixed in the
location-only correctness update. Quantiles solve the full mixture CDF, including
component variance. Log-sum-exp preserves the normalization constant; the added
`log_normalization_constant` remains useful if the ordinary constant underflows.
Independent quadrature tests cover centered, shifted and narrow kernels. The
old [cleanup diagnostic](../docs/research/evidence/repository_cleanup_2026-10-12/kde_status.json)
records the pre-fix behavior and remains unchanged. See the
[collaborator review note](../docs/research/location_only_meeting_brief.md).

**This does not certify every KDE/cache path.** The raw weighted calculation and
sampler acceptance rules were unaffected by these two helper defects.

The `selected_global_v1` cache name separates old score-root draws but does not
identify selector version/tolerances, validate stored array metadata, or guarantee
freshness. Cache-write disabling still permits existing reads. The legacy audit's
resume keys also omit release/prior. No unchecked cache is promoted by cleanup.
`SJ_transform` may fall back to a Silverman bandwidth while retaining that
requested backend label; it is not guaranteed to run actual Sheather–Jones.

Use raw-weighted reference summaries as the primary numerical comparison and
KDE only with explicit approximation/provenance labels. The finite-grid and
quadrature helpers retain their truncation/integration limitations. Cache
provenance and bandwidth selection need separate work before broader use.
