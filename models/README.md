# Model code

Code is organized by the parameters being inferred, then by distribution family.
The layout migration preserved numerical kernels. The later location-only
correctness work adds explicit selected-target methods and changes selected
Student numerical rules. Historical score comparators remain unchanged.

```
models/
  loc/                          # known-scale location inference
    student/
      selected_block.py         # efficient selected-global-MLE block Gibbs
      selected_gibbs.py         # transparent corrected pair Gibbs baseline
      selected_rattle.py        # corrected RATTLE, with known exploration limits
      independence.py           # selected-MLE independence MH in base-data coordinates
      mle.py                    # hardened global location selector
      n3_endpoint.py            # optional guarded n=3 endpoint certificate
      endpoint.py               # n=3 and small-diameter root certificates
      score_gibbs.py            # score-conditioned comparator and shared pair math
      score_rattle.py           # score comparator and shared constrained integrator
      mu_mh.py                  # legacy compiled parameter update
      selected_mu.py            # selected kernels: unfloored MH uniform
    logistic/
      selected_gibbs.py         # reversible score block with curvature MH
      selected_rattle.py        # pure score-guide RATTLE with curvature MH
      selected.py               # shared geometry, root brackets, compiled steps
      score_gibbs.py
      score_rattle.py
      _numba.py
    laplace/
      selected_gibbs.py         # odd median / even midpoint, exchangeable labels
      selected_facet_hmc.py     # reflective HMC; NOT smooth RATTLE
      selected.py               # ordered conditionals and wall dynamics
      median_gibbs.py
      facet_rattle.py           # fixed-facet experimental comparator
      _numba.py
    affine.py                   # multi-family local affine-normalization sampler
  loc_scale/                    # preserved experimental joint models
    student_gibbs.py
    student_rattle.py
    laplace_gibbs.py
  scale/                        # reserved; no standalone scale-only implementation
  shared/                       # numerical rules; old utilities preserved
  model_registry.py             # separate legacy and selected method metadata
  _compat.py                    # old import aliases; no duplicated algorithms
```

`loc_scale` uses an underscore because Python import names cannot contain a
hyphen. `scale` is an explicit placeholder, not a validated scale-only method.

For current Student location work:

```python
from models.loc.student import selected_block, independence
from models.loc.student.mle import global_mle
```

Selected-MLE pair, block and RATTLE kernels include global selection and curvature.
Score kernels condition on score zero and have a different invariant target.
They remain because corrected kernels reuse their numerical machinery and the
research still needs those comparisons. The pair kernel is the transparent
correctness baseline, not a disposable duplicate of block Gibbs.

RATTLE's correct endpoint target does not establish adequate component/tail
exploration. The affine sampler also has recorded shifted-case latent-mixing
limitations. Independence MH and block Gibbs are useful alternatives in their
tested regimes; there is no claim of universal mixing.

Logistic score conditioning is distinct from selected-MLE conditioning despite
its unique root. Use its new `selected_gibbs` or `selected_rattle` for the latter.
Laplace's new methods use the middle observation for odd n and the midpoint of
the two middle observations for even n. They restore exchangeable labels. The
legacy Laplace methods retain their old target and label restrictions. Joint
Student/Laplace models remain experimental.

`SELECTED_LOCATION_METHODS` lists explicit imports, entrypoints, targets and
validation status. The later bounded comparison and expanded diagnostics pass
all three Student block/independence cases, all three Logistic block cases (one
borderline), and all four Laplace cases for each selected method. Student RATTLE
has warnings in all three cases; Logistic RATTLE has n=10 label-mixing warnings.
These are finite-run findings, not universal guarantees. See the
[collaborator review note](../docs/research/location_only_meeting_brief.md) for derivations,
limitations, reference comparisons and reproduction commands.

Student block/RATTLE runners default to `mle_check='auto'`; choose `'selector'`
for the original endpoint check. The legacy `n3_endpoint_fast_path` option still
works when supplied alone. Never supply both controls. Certificates require a
root enclosure, not just a small score residual. The global selector is unchanged.

Raw weighted/rejection references remain in `kde_ref/` and their current
validation drivers. KDE is supplementary. Mathematical proofs and independent
oracles remain in `reporting/diagnostics/` and `scripts/`.

Historical `models.loc_student*`, `models.student_location_mle`, and other old
module imports still resolve to the identical canonical module through one lazy
alias table. They do not load a second kernel or introduce a second RNG stream.
Active callers use the new imports. Old physical filenames are retired; use the
[migration map](../docs/checkpoints/2026-10-12-model-layout/migration.json) for
source paths and historical integrity checks. Subsequent scientific changes have
separate before/after hashes. `verify_archived_source` can verify original bytes;
it does not assert that revised current code is numerically identical.
