# Executed repository cleanup: archive, dashboard and obsolete workflows

Date: 2026-10-12. This follows the completed model-layout migration and the
user's explicit instruction to remove the recovery archive, dashboard and old
results/scripts. No commit, push, new research campaign or numerical algorithm
change was made in this follow-up.

## Removed from the checkout

**19,962 files and 16,943,595,465 logical bytes (15.78 GiB)** were deleted in
this pass. The [per-file/directory deletion record](evidence/repository_cleanup_2026-10-12/deletions.json)
is the exact scope; counts exclude the empty `.aws` directory. Logical bytes
are not a filesystem quota/reclamation measurement. The earlier model-layout
cleanup separately deleted 607 MiB; its record remains unchanged. The measured
checkout is approximately **2.74 GiB including `.git` and `.venv`**; these
account for most remaining space. `results/` is 114 MiB and `tmp/` 67 MiB.
The [storage record](evidence/repository_cleanup_2026-10-12/storage.json) states
the measurement and distinguishes deletion from relocation.

| Item | What it was and action |
|---|---|
| `_archive_repo_cleanup_20260920_checkpoint/` | Removed the entire 9,051,650,359-byte September recovery package, including old tar/bundle. Compact metadata is retained, not its payload. |
| `.aws/` | Empty AWS-named directory, unrelated to the research; removed. Its origin could not be established. No credential contents were present. |
| `dashboard/` | Entire historical Streamlit UI, pages, cache preparation and associated launchers removed. Its dependency declaration was removed; installed packages were not uninstalled. |
| `configs/` | Eight old score-benchmark/tuning grids removed. Current selected-MLE configurations remain beside canonical evidence under `docs/research/evidence/*/config.json`. |
| `hpc/` | Twenty-two old Slurm job/configuration files removed, together with old `scripts/*.sbatch` and submit-all launcher. `squeue` reported no running user jobs. |
| `analysis.py` | Retired KDE variance/KL and simple ESS helper module. No live callers after retirement of its workflows; removed. |
| `diagnostics/run_registry.py` | Obsolete runset/grid loader coupled to old configs; removed. |
| Old Python drivers/reporting | Fifty-seven obsolete Python files removed after tracing retained entrypoints, imports, tests and dynamic/subprocess dependencies. Exact names are in the deletion record and import-closure record. |
| `reporting/presentation/` | Remaining five event-workflow numerical helpers removed after verifying their compact original-source recovery. No presentation builder remains active. |
| Old `results/` contents | Old score production grids, profile/tuning/repair runs, dashboard caches and redundant exports removed. Exact necessary inputs are enumerated by retained hashes. |
| Two temporary dependency trees | `tmp/presentation_diagnostic_deps/` and `tmp/selected_mle_validation_deps/` removed; no current entrypoint/import/test used them. The existing environment and separate oracle dependency installation are untouched. |

Three **unique historical controls**, totaling **141,058,990 bytes**, were moved
outside the checkout with read-back SHA-256 verification:

- `tmp/presentation_long_chains/`: original historical score-chain evidence.
- `tmp/selected_mle_reference_validation_20261003/`: initial selector/reference
  study, including its unfavorable precision result.
- `tmp/cheap_selected_mle_gibbs_20261010/`: first failed block-proposal study.

Their external location and every original/new path/hash are in the
[relocation manifest](evidence/repository_cleanup_2026-10-12/retired_raw_controls.json).
This is a same-filesystem move: it reduces checkout clutter, **not total disk
use or disaster-recovery risk**. Historical reports/manifests keep their original
paths; consult this map to retrieve/replay those retired controls. Current study
scripts need fresh output locations rather than overwriting saved evidence.

## What remains and why

`models/` retains the reorganized current implementations, score comparators
needed by corrected samplers, and explicitly experimental joint/comparator work.
This pass did not change their numerical source. See the [model guide](../../models/README.md).

`diagnostics/` contains only shared **cost accounting** and **source provenance**
helpers, both imported by current models/reference validation. Model-specific
geometry and diagnostic calculations remain in `reporting/diagnostics/`.

`examples/` contains two small demonstrations: the legacy score-conditioned
comparator and a new selected-global-MLE block example with an explicit valid
initial dataset. The latter runs 40 n=3/k=2 iterations and verifies shapes,
finiteness and endpoint constraints. Neither example establishes convergence.

`scripts/` contains 21 Python/shell drivers: current reference, selector, sampler
and geometry validation; the independent rational oracle; the checkpoint runner;
Laplace prototype validation; and an environment-setup convenience. Two comparator
drivers remain because tests directly import their movement/tuning helpers.
The [script index](../../scripts/README.md) gives the purpose of each group.
There is no current cluster job grid. `grace_setup_env.sh` is retained but was
not run, so no packages were installed or upgraded.

`results/` retains **348 immutable scientific files plus its updated index**:
geometry/component evidence, Laplace prototype validation, and manuscript inputs.
In particular, 300 paired draw CSVs and small summary tables are required by the
information-loss manuscript figure helper; these are not the full retired grid.
The manuscript still includes three PNGs under `docs/presentation/Images/final/`.
Their location does not make them disposable event assets. The old privacy
figure's numerical builder is preserved in the original-source checkpoint;
that figure has frozen-source replay rather than a current production CLI.
See the [retained-results index](../../results/README.md).

The seven current selected-MLE raw-study roots retain **153 files byte-for-byte**:
shifted follow-up, corrected pair Gibbs, corrected RATTLE, repaired block Gibbs,
n=3 geometry, guarded n=3 fast path, and independence/higher-n geometry. Their
raw weighted banks, seeds, chains, configuration and source hashes were not
regenerated. Small old score/MLE counterexamples and the joint diagnostic cited
in the target audit also remain. They provide specific scientific evidence,
not a dashboard or broad historical experiment collection.

## KDE/reference status: not uniformly clean

`kde_ref/` remains needed for raw-weighted summaries, reference adapters and
supplementary density display. Its numerical code was not modified by cleanup.
The current [package status](../../kde_ref/README.md) records known limitations:

1. `gaussian_kde_gaussian_prior_moments` takes quantiles of mixture component
   means instead of the complete Gaussian mixture. Its normalizer also loses
   the original maximum log weight. Mean/variance include component variance.
2. The `selected_global_v1` convenience cache lacks full selector/tolerance,
   metadata and finiteness validation. Disabling cache writes still permits
   existing reads. Legacy resume keys omit prior/release.
3. The `SJ_transform` backend can silently use a Silverman fallback under the
   requested label. KDE/grid smoothing/truncation remain approximations.

A deterministic two-point Gaussian-KDE probe confirms the first defects:
reported 2.5%/50%/97.5% quantiles are approximately
`[-0.72520, -0.72520, 0.65268]`, whereas the full mixture CDF gives
`[-2.46185, 0, 2.46185]`. Reported normalization is `1`, versus `0.15514589`.
The [probe record](evidence/repository_cleanup_2026-10-12/kde_status.json)
retains the values. `raw_weighted_posterior_moments` is a separate path and is
unaffected. Use validated fresh raw-weighted banks with recorded uncertainty
as the primary selected-MLE reference, not unchecked convenience caches.
A passing suite does not certify these known-defective optional KDE helpers.

## Verification and preservation

- **358 tests passed**, with no timeout or collection failure. Both examples and
  the reference, selected-Gibbs and component CLI probes passed. Exact commands,
  ceilings, environment and logs are under
  [validation](evidence/repository_cleanup_2026-10-12/validation/validation.json).
- Eleven implementations reproduced **bit-identical full arrays and acceptance
  decisions** against the saved pre-migration fixtures. This is a small seeded
  regression comparison, not a mixing experiment.
- All twenty moved numerical module files still have their exact post-migration
  hashes. Current selected-MLE raw files and retained scientific results match
  the pre-deletion hashes. Only the results directory's index changed.
- Independent reference preflight verified the 32,768-particle n=3 and two
  131,072-particle n=10 banks, original RATTLE/Gibbs comparison hashes and eight
  source records. See [reference checks](evidence/repository_cleanup_2026-10-12/reference_checks.json)
  and [preservation checks](evidence/repository_cleanup_2026-10-12/preservation_check.json).
- HEAD and the staged binary patch remain unchanged. Uncommitted retired source
  was saved before deletion: 94 original source/config/Slurm files in a verified
  **208,726-byte external archive**, with original modes/hashes and Git state.
  The [recovery pointer](evidence/repository_cleanup_2026-10-12/retired_source_recovery.json)
  identifies it. This is compact source recovery, not another repository copy.
- The prior model-layout source snapshot remains available for provenance and
  older figure helpers. The old nine-GB recovery payload and obsolete bulk
  result grids were deliberately deleted; this new source archive cannot
  recover those data. Existing canonical reports do not imply those old raw
  paths still exist.

The environment remains Python 3.13.5 in `.venv`, with existing SymPy/mpmath
supplied through `/tmp/student-mle-oracle-deps-20261003` for these checks.
No fresh environment installation, dependency upgrade, expensive simulation or
new statistical validation campaign was undertaken. Numerical fixes to KDE,
portable environment locking and further algorithm work are separate tasks.

For source recovery, extract only needed members into an isolated directory,
verify their manifest hashes, then reconcile with current source. Do not overlay
an old archive blindly. For retired historical raw controls, use the relocation
map; the canonical current reference banks continue to use their original paths.
