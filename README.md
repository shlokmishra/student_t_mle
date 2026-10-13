# Student-t MLE conditioning research

Research on Bayesian inference from a released location estimator, with Student-t
as the main model and Logistic/Laplace comparators.

**Current work:** known-scale Student location inference given the selected
**global** MLE. Corrected pair Gibbs, reversible-block Gibbs, endpoint-corrected
RATTLE, independence MH, and weighted/rejection references are available.
The score-conditioned Student and Logistic comparators have a different target.
Selected Logistic Gibbs/RATTLE and Laplace median Gibbs/reflective HMC are also
available. A bounded ten-case comparison and expanded latent diagnostics are
complete. Student RATTLE has slow latent tails; Logistic RATTLE has n=10 label
mixing warnings. Block/median Gibbs passes the tested checks, with one borderline
Logistic case. Start with the
[collaborator review note](docs/research/location_only_meeting_brief.md), also
available as a [LaTeX document](docs/research/location_only_meeting_brief.tex).
Joint location-scale implementations remain experimental.

Then read [the performance comparison](docs/research/location_only_performance_comparison.md)
and [the chain/latent-quantile review](docs/research/location_chain_convergence.md).
For implementation and theory, use [the model guide](models/README.md), the
[selected-MLE target derivation](docs/research/selected_mle_target_audit.md), and
[the block-Gibbs results](docs/research/cheap_selected_mle_gibbs.md).
The [research reports](docs/research/) document geometry, independent reference
checks, mixing limitations and failures. A correct invariant target does not
establish adequate exploration for every sampler or dataset.

## Code layout

```text
models/
  loc/
    student/       selected-MLE samplers, score comparators, selector, n=3 certificate
    logistic/      selected-MLE methods and score-conditioned comparators
    laplace/       selected median methods and legacy facet comparators
    affine.py      multi-family local affine-normalization sampler
  loc_scale/       experimental joint Student/Laplace models
  scale/           reserved; no standalone scale-only implementation yet
  shared/          shared numerical utilities
```

Use imports such as `from models.loc.student import selected_block` and
`from models.loc.student.mle import global_mle`. Historical flat imports still
resolve to the same implementation through one compatibility module. New source
provenance uses canonical paths; old evidence hashes remain unchanged and are
checked against the [verified migration](docs/checkpoints/2026-10-12-model-layout/migration.json).

## Setup

On Grace, confirm the prerequisites (`module spider` only queries metadata),
then set up the project environment:

```bash
module spider Python/3.13.5
PYTHON_MODULE="GCCcore/14.3.0 Python/3.13.5" bash scripts/grace_setup_env.sh
```

For an existing environment, run from the repository root:

```bash
.venv/bin/python --version
.venv/bin/python -m pytest tests -q
.venv/bin/python examples/selected_mle_student.py
```

The 2026-10-12 collaborator check passed 403 tests in 637.11 seconds;
[its logs and bounded examples](docs/research/evidence/location_review_handoff_2026-10-12/validation/)
record the current verification. The historical 2026-09-20 baseline used
Python 3.13.5 and passed 113 tests; exact installed
versions are in [environment-freeze.txt](docs/checkpoints/2026-09-20/environment-freeze.txt).
This captures the existing environment, including historical UI dependencies;
it is not yet a minimal or cross-platform lockfile. The default system Python
on the inspected host is 3.6.8 and cannot run current project code. A fresh
dependency installation was not performed for this checkpoint.

On other systems, use Python >=3.10, create `.venv`, then install
`requirements.txt`. The loose dependency bounds have not been validated across
that whole version range. Packaging and dependency separation belong to the
next cleanup checkpoint.

## Small example

`examples/selected_mle_student.py` runs 40 n=3, k=2 selected-MLE block updates
with an explicit valid dataset, seed 20261012 and the optional guarded endpoint
check enabled. It checks shapes, finiteness, score residuals and released-MLE
accuracy. This is a runnable smoke example, not convergence evidence.
`examples/score_conditioned_student.py` provides the separate score comparator.
Initial execution may compile Numba kernels.

For bounded repeatable validation with logs and exit codes:

```bash
.venv/bin/python scripts/run_checkpoint_validation.py --out tmp/checkpoint-validation-new
```

Use a fresh output directory for each run. The runner limits numerical threads
and CPU affinity to one CPU and imposes a 900-second test ceiling. `pytest.ini`
keeps default discovery within `tests/`, excluding installed dependency tests
under temporary presentation directories.

## Repository map

| Path | Purpose |
| --- | --- |
| `models/` | Current numerical models, samplers and selectors; [guide](models/README.md) |
| `kde_ref/` | Raw weighted summaries and KDE utilities; [correctness status](kde_ref/README.md) |
| `diagnostics/` | Live cost accounting and historical source verification |
| `scripts/` | Current validation, geometry/oracle drivers, and a few tested comparator helpers; [index](scripts/README.md) |
| `reporting/diagnostics/` | Model-specific mathematical and numerical diagnostics, plus retained manuscript helpers |
| `tests/`, `examples/` | Correctness checks and two small runnable demonstrations |
| `docs/research/` | Canonical reports and compact evidence/configuration/provenance |
| `docs/paper/` | Earlier manuscript draft, not the current validation account; three historical figure inputs remain under `docs/presentation/Images/final/` |
| `results/` | Only directly required geometry, Laplace and manuscript inputs; [index](results/README.md) |
| `tmp/` | Current selected-MLE reference banks and raw validation chains, plus specifically retained older inputs |

The old root `configs/` held legacy experiment grids; those are retired. Current
selected-MLE configurations are versioned alongside their reports under
`docs/research/evidence/*/config.json`. Old `hpc/` job grids, the entire dashboard,
`analysis.py`, and the large September recovery archive have been removed.
The empty unrelated `.aws/` directory is also removed.

## Verification and cleanup record

The [model-layout migration](docs/checkpoints/2026-10-12-model-layout/README.md)
preserved numerical code and passed 358 tests with bit-identical seeded checks
for eleven implementations. The [subsequent cleanup](docs/research/repository_cleanup_followup.md)
records the exact additional deletions, retained dependencies and new validation.
Compact historical source snapshots remain for provenance; there is no nested
full-repository recovery archive. Current raw weighted reference banks and selected-MLE
validation trajectories are protected.

The [handoff inventory](docs/research/evidence/location_review_handoff_2026-10-12/handoff_inventory.csv)
records current file hashes and Git status. It includes untracked source and
ignored scientific banks/chains; these do not accompany a Git-only handoff.
The [review preservation record](docs/research/evidence/location_review_handoff_2026-10-12/verification.json)
checks that numerical code and previous evidence remain unchanged.

KDE is supplementary. Its Gaussian-mixture helper is fixed; cache issues remain.
Passing the test suite is not a claim that every KDE path is statistically correct.
Use the validated raw weighted reference and its source/uncertainty records.
