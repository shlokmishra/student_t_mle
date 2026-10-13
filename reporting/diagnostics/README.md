# Retained mathematical and numerical diagnostics

Current tools cover Student score/MLE comparisons, the independent global-selector
checks, n=3 component/chamber construction and selected support, higher-n geometry,
and the exact n=5 boundary. They are imported by the mathematical tests and
validation drivers in `scripts/`.

`audit_kde_reference.py` and `audit_reference_all_models.py` remain because the
reference adapter and comparator tests use their numerical machinery. Legacy KDE
cache/resume limitations are recorded in `../../kde_ref/README.md`.

`prepare_meeting_evidence_figures.py` and `run_release_information_full_data.py`
remain for manuscript inputs and their numerical reconstruction functions.
The former now has only retained release-information inputs available by default;
retired efficiency/geometry report inputs are not recreated or invented. Original
privacy plotting source remains in the compact pre-cleanup source snapshot.

Old dashboards, general production-report orchestration, performance-summary
wrappers and presentation builders have been retired. Historical methods are
not automatically selected-MLE-correct because they use the current selector.
