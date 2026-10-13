# Completed model-layout cleanup

**Historical checkpoint:** the subsequent [cleanup](../../research/repository_cleanup_followup.md) removed the old archive, dashboard and obsolete workflows mentioned below. This page records the preceding migration stage.

Date: 2026-10-12. Implemented after explicit approval to clean and reorganize
the repository. No commit, push, dependency upgrade or new research campaign.

## What changed

Twenty numerical modules moved into `models/loc/`, `models/loc_scale/`, and
`models/shared/`. Location modules are grouped by Student, Logistic and Laplace.
`models/scale/` explicitly reserves scale-only work; there is no existing
standalone implementation to move there. Use the Python name `loc_scale`.
The [model guide](../../../models/README.md) explains each implementation's target
and validation status. Flat numerical files were removed, rather than duplicated
under the new layout. One lazy compatibility table supports their old imports.

Active callers, including tests, examples, numerical references, experiment
drivers and mathematical diagnostics, use canonical imports. Future source
fingerprints use canonical physical paths. Nine documentation files received
updated clickable source-link destinations; historical claims and frozen
provenance records were not rewritten.

The comparator registry now explicitly marks Student and Logistic score kernels
as having a different target from selected-MLE references. It describes the
hardened selector's exact-tie convention correctly. These are metadata changes,
not algorithm changes. Importing the root package no longer eagerly loads joint
models or JAX; implementations load when requested.

**All twenty numerical module bodies are unchanged apart from imports.**
Arithmetic, branch laws, MH corrections, support checks, failures, RNG routines,
sampler parameters/defaults and outputs were preserved. Score Gibbs/RATTLE remain
because current corrected samplers depend on their numerical machinery. The
transparent selected pair implementation remains the correctness baseline.
Joint models and facet RATTLE remain explicitly experimental.

## Deletions

Executed 81 audited low-risk DELETE units and 16 event-only builder/dependency
files, plus caches regenerated during baseline checks. The exact
[deletion record](deletions.json) lists every removed member.

**1,515 files, 636,581,054 logical bytes (607.09 MiB)** were removed. This includes
obsolete MSc L.T. Jordan / Beyond Borders poster/PPT/PDF exports, temporary
render revisions, redundant diagnostic logs, presentation builders and
interpreter/JIT/pytest/font caches. New verification evidence costs about 1 MiB;
logical removal is not a quota-level disk-reclamation measurement.

Five historical numerical presentation helpers, the manuscript's three figure
inputs, original scientific chains, current weighted banks, proof certificates
and failed-run evidence remain. No ARCHIVE/REVIEW unit was removed. The historical
dashboard and its coupled launchers, large unique studies and September recovery
tar are not covered by this removal: their broader archival/dependency
prerequisites still apply. This is a completed model-code reorganization and
audited low-risk deletion, not a claim that every old result has disappeared.

## Numerical and operational verification

- Before changes: **329 tests passed**, plus the score example and three CLI
  probes. [Baseline logs](baseline/validation.json) record commands and ceilings.
- Final layout: **358 tests passed**, including 29 new layout/provenance tests;
  the example and CLI probes passed. [Final logs](final/validation.json).
- Eleven implementations produced **bit-identical full saved arrays and
  acceptance decisions** before/after: legacy Student Gibbs/RATTLE, selected
  pair/block/RATTLE, independence MH, affine, Logistic and Laplace Gibbs, and
  joint Student Gibbs/RATTLE. Each uses 12 iterations, fixed explicit settings
  and two-word seed `[0, 20261012]`. This is migration verification, not a mixing
  study. Both saved NPZ files have SHA-256
  `5fb4b2c320209aa84fb20faa976f45f0a0efa0b217d912be7f1eb55153d6eec0`.
- The twenty module ASTs are identical after removing import nodes. The hardened
  selector and n=3 certificate moved byte-for-byte.
- Old and new imports refer to the identical module, including private compiled
  helpers, in either import order. Frozen n=3 block-source imports still resolve.
- [Reference checks](reference_checks.json) verified eight historical source
  records, the 32,768-particle n=3 and 131,072-particle n=10 banks, and RATTLE's
  reference preflight: original summaries and comparison-chain hashes agree.
  Raw banks were never regenerated or overwritten.

The first post-move suite had 22 failures in the **new compatibility tests**:
Python overwrote canonical module `__spec__` with the old alias. Numerical tests
passed. Restoring the canonical spec in the alias loader fixed it; failed logs
remain in [first_attempt/](first_attempt/). There were no test timeouts or
collection failures. Warmed/cold timing differs across runs; no performance
improvement is inferred from shorter final test time.

The environment is Python 3.13.5 in `.venv`, with existing SymPy/mpmath supplied
by `/tmp/student-mle-oracle-deps-20261003`; dependencies were not installed or
upgraded. CPU/thread limits and exact commands are in validation logs. An
environment providing the same required dependencies can omit that PYTHONPATH.
`git diff --check` still flags pre-existing CRLF whitespace in two unchanged HPC
TSV files; this cleanup did not rewrite those user-owned files.

## Preservation and provenance

The [migration map](migration.json) records original/current paths and both
hashes. The 488-KiB `source_before.tar.gz` contains original active Python source
for historical integrity checks; it is explicitly unignored in Git. This is a
compact source checkpoint, not a duplicate repository/evidence backup.
`diagnostics/source_provenance.py` accepts an old hash only when archived original
bytes match it **and** current canonical bytes match this exact migration. A
later change or an unknown old hash raises; it never reports a new hash as old.

An externally located, read-back-verified source recovery package is at:

```
/scratch/user/shlokmishra08/codex/backups/student_t_mle/20261012-093023-model-layout/
```

It contains pre-change tracked/untracked source/configuration/docs, original Git
status, HEAD, staged/unstaged binary patches and index. Its compressed source
payload is 1,839,849 bytes. It is on the same filesystem and does **not** claim to
back up all ignored scientific arrays. Those remain unchanged in the checkout.
The [preservation check](preservation_check.json) confirms 659 unrelated files
from the earlier complete inventory are byte-identical, with no unexpected
changes. HEAD and staged patch are unchanged. Existing uncommitted numerical work
was moved intact, not replaced with committed older versions.

For rollback, extract only required members from the external package into an
isolated directory; verify them against its source manifest, then reverse the
documented path/import changes. Do not restore the whole package blindly over
later work. Obsolete generated exports were deliberately deleted; the old
September archive was not removed.

To repeat bounded verification from the repository root:

```bash
PYTHONPATH=/tmp/student-mle-oracle-deps-20261003 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/tmp/student-model-cleanup-numba-cache \
  PYTEST_ADDOPTS='-p no:cacheprovider' \
  .venv/bin/python scripts/run_checkpoint_validation.py --out /tmp/student-layout-new-check

PYTHONPATH=/tmp/student-mle-oracle-deps-20261003 PYTHONDONTWRITEBYTECODE=1 \
  NUMBA_CACHE_DIR=/tmp/student-model-cleanup-numba-cache \
  .venv/bin/python docs/checkpoints/2026-10-12-model-layout/verify_seeded.py
```

Use a fresh output directory. The seeded verifier only reads the saved evidence
and compares regenerated small chains; it does not overwrite it. Larger cleanup
of archived datasets or scientific algorithm changes is separate from this
verified source migration.
