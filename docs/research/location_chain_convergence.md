# Location-chain traces and latent quantiles

12 October 2026. Follow-up to the [performance comparison](location_only_performance_comparison.md).

**All 92 runs have full saved parameter and latent chains.** All file hashes
match the original records. The expanded checks pass for 18 of 23 method/case
settings. Student RATTLE still has slow latent-tail exploration. Two Logistic
RATTLE settings also show incomplete mixing of individual labelled observations.
Parameter diagnostics alone miss these problems.

Open the **[plot gallery](evidence/location_chain_diagnostics_2026-10-12/index.html)**
for every case and method. It contains 43 figures: ten parameter/latent-quantile
trace sheets, ten running-estimate/autocorrelation sheets, and 23 sheets showing
every labelled and ordered latent coordinate. All ten main trace sheets were
visually reviewed, together with detailed problem-coordinate and stability plots.

## What is stored

The [chain inventory](evidence/location_chain_diagnostics_2026-10-12/chain_inventory.csv)
lists each file, method, seed, array shape, retained range and SHA256 hash.
Files remain in `tmp/location_only_performance_20261012/`; no duplicate raw banks
or chains were created. The 92 compressed chain files occupy 50,665,592 bytes.
This directory is Git-ignored: the manifest identifies the raw evidence that
must accompany the source when sharing or preserving this study.

Each NPZ contains `mu` with shape `(12001,)` and `x` with shape `(12001, n)`.
Index zero is the initial state. We keep the original burn-in: analysis uses
indices 2001–12000, or 10,000 retained states per chain. Nothing is thinned.
The four dispersed starts, seeds and sampler settings remain unchanged.

From the repository root:

```python
import numpy as np

path = "tmp/location_only_performance_20261012/student_shifted_n10_block_chain0.npz"
with np.load(path, allow_pickle=False) as saved:
    mu_chain = saved["mu"]
    x_chain = saved["x"]

mu = mu_chain[2001:]
x = x_chain[2001:]
# Quantiles within each latent dataset, at each retained iteration:
latent_quantiles = np.quantile(x - 1.25, [.05, .25, .50, .75, .95], axis=1)
# Posterior quantiles of the first labelled observation, across iterations:
x1_posterior_quantiles = np.quantile(x[:, 0], [.01, .05, .50, .95, .99])
```

Direct rejection outputs and weighted reference banks also remain saved in the
original study. They are independent draws/weighted particles, not additional
Markov chains. They do not receive artificial MCMC R-hat values. Affine and
transparent pair-Gibbs historical runs are outside this fresh benchmark.

## Checks and interpretation

We examine two distinct quantiles:

- **Within-dataset quantiles:** the 5th, 25th, 50th, 75th and 95th percentiles of
  the `n` observations at each iteration. These measure changing latent shape.
  We use NumPy's linear interpolation, including at n=3.
- **Posterior quantiles:** the 1st, 5th, 25th, 50th, 75th, 95th and 99th
  percentiles across retained iterations, for each parameter/latent observable.
  We save pooled, per-chain and first-/second-half estimates. CDF-indicator ESS
  at each percentile assesses dependence relevant to that quantile.

The 493 scalar traces include the parameter, every labelled observation,
every order statistic, the five within-dataset quantiles, maximum absolute
residual, residual range and indicator of maximum residual above 6. Latent plots
use `x-t`; add the released value `t` to recover observation locations.

Each nonconstant trace receives rank-normalized split/folded R-hat, bulk ESS,
5%/95% tail ESS, raw-function ESS and split summaries. We reuse the existing
diagnostic estimators. The original parameter/maximum/tail-event values reproduce
to `1e-12` for all 23 settings. Plots show **every stored iteration**, with no
striding or line simplification. Grey shading marks burn-in. Student latent
axes use a symmetric log scale to retain both signs and extreme excursions.
Vertical limits differ between panels; compare persistence and numerical
summaries, not just the apparent width of the plotted band.

Dashed trace lines show the weighted reference's posterior 5th/95th percentiles.
Running-mean plots show its weighted mean. These use the verified raw banks,
not KDE. The lines are finite-bank estimates, **not confidence intervals**.
The new quantile comparisons are descriptive; no new formal reference-agreement
test or quantile confidence interval is claimed. Tail-sensitive running means
can move sharply after an extreme draw, even for a sound sampler.

The expanded diagnostic screen applies the existing thresholds, R-hat ≤1.01
and bulk/tail ESS ≥400, to all nonfixed observables. The 1%/99% indicator ESS is
reported separately. This is an exploratory extension, not a replacement of
the frozen benchmark criteria. It involves many dependent diagnostics; an
isolated threshold crossing is a warning, not proof of an incorrect kernel.
Passing also does not establish convergence of the whole joint distribution.

Laplace's median is fixed by the statistic, for both odd and midpoint-even
conventions. The odd-n middle order statistic is fixed too. These 12 observable
instances are marked `constraint_fixed`, with R-hat/ESS left undefined. Their
maximum residual is `5.16e-14`. Tiny even-n median movements visible on the
automatically magnified axes are rounding error, not posterior exploration.
A constant trace without such a mathematical constraint remains unresolved.

## Results

This table takes the worst R-hat and the smallest bulk/tail ESS over all
nonfixed observables. Different entries in a row can come from different
observables. Threshold decisions use unrounded values.

| Family / case | Method | Largest R-hat | Smallest bulk ESS | Smallest tail ESS | Expanded screen |
|---|---|---:|---:|---:|---|
| Student n3 centered | Block | 1.0056 | 986 | 2467 | Pass |
| Student n3 centered | RATTLE | 1.0128 | 685 | 508 | Check |
| Student n3 centered | Independence | 1.0001 | 33701 | 33495 | Pass |
| Student n10 centered | Block | 1.0037 | 1367 | 6583 | Pass |
| Student n10 centered | RATTLE | 1.0219 | 297 | 120 | Check |
| Student n10 centered | Independence | 1.0002 | 37774 | 36201 | Pass |
| Student n10 shifted | Block | 1.0048 | 1005 | 4707 | Pass |
| Student n10 shifted | RATTLE | 1.0098 | 432 | 176 | Check |
| Student n10 shifted | Independence | 1.0064 | 1038 | 652 | Pass |
| Logistic n3 centered | Block | 1.0095 | 710 | 2257 | Pass |
| Logistic n3 centered | RATTLE | 1.0073 | 691 | 750 | Pass |
| Logistic n10 centered | Block | 1.0093 | 746 | 3555 | Pass |
| Logistic n10 centered | RATTLE | 1.0158 | 572 | 729 | Check |
| Logistic n10 shifted | Block | 1.0100 | 568 | 3704 | Pass, borderline |
| Logistic n10 shifted | RATTLE | 1.0191 | 526 | 790 | Check |
| Laplace n3 centered | Gibbs | 1.0013 | 2278 | 2782 | Pass |
| Laplace n3 centered | Facet HMC | 1.0042 | 1555 | 1518 | Pass |
| Laplace n4 centered | Gibbs | 1.0014 | 3257 | 4193 | Pass |
| Laplace n4 centered | Facet HMC | 1.0048 | 1261 | 1294 | Pass |
| Laplace n3 shifted | Gibbs | 1.0015 | 3053 | 5343 | Pass |
| Laplace n3 shifted | Facet HMC | 1.0046 | 1620 | 1384 | Pass |
| Laplace n4 shifted | Gibbs | 1.0006 | 3471 | 5197 | Pass |
| Laplace n4 shifted | Facet HMC | 1.0033 | 1328 | 1223 | Pass |

All 23 **parameter** traces pass the bulk/5%/95% checks: largest parameter
R-hat is 1.00636 and smallest parameter tail ESS is 652. This does not suffice
for joint latent validation.

**Student:** the [centered n10 traces](evidence/location_chain_diagnostics_2026-10-12/student_centered_n10_traces.png)
show long positive-tail excursions for RATTLE. The within-dataset 95th percentile
has R-hat 1.02102 and tail ESS 120. The largest order statistic has tail ESS 120
too. Individual `x2` also mixes poorly. Per-chain probabilities of maximum
residual above 6 range from 0.1725 to 0.3218; the longest tail residence is 630
iterations. Block and independence proposals repeatedly leave these regions.

In the shifted Student case, RATTLE's **lower** latent tail is slow: the 5th
percentile has tail ESS 227, and `x4` has tail ESS 176. Every R-hat is below 1.01
in this setting, illustrating why R-hat alone is insufficient. At n=3, `x3`
and the tail-event indicator fail the R-hat screen, although the five
within-dataset quantiles pass. The RATTLE maximum-residual ACF decays much more
slowly than block Gibbs or centered independence MH.

Shifted Student independence MH passes the bulk/5%/95% checks but has less
precision at the outer 1%: indicator ESS is 323 for the parameter and 357 for
the fourth order statistic. Do not read its pass as precise far-tail inference.

**Logistic:** all five within-dataset quantiles and all order statistics pass.
However, [labelled traces](evidence/location_chain_diagnostics_2026-10-12/logistic_shifted_n10_rattle_coordinates.png)
show slower movement. Centered n10 RATTLE has `x8` R-hat 1.01585. Shifted n10
has `x5` 1.01141 and `x7` 1.01912. For shifted `x7-t`, the four chain means are
−0.454, −0.080, 0.276 and 0.054. This is evidence of incomplete finite-run label
exploration despite adequate symmetric summaries; it does not identify a new
target error. Shifted block Gibbs is borderline: `x8` R-hat is 1.009988.

**Laplace:** both methods pass the labelled and ordered checks. The trace
sheets show that facet HMC has more persistent latent excursions than Gibbs,
but the current run supplies adequate ESS for the measured quantities. Uniform
relabelling makes labelled observations move quickly, so the separate ordered
checks are essential. The fixed median alone contains no mixing information.

The earlier **20/23** result remains correct for its declared observables and
reference tests. The broader **18/23** result adds two Logistic RATTLE warnings.
Both records are retained. No failed run was extended, reseeded or retuned.

## Reproduction and checks

The [analysis script](../../scripts/analyze_location_chains.py) reads the frozen
study and refuses to overwrite an existing output directory:

```bash
env PYTHONPATH=/tmp/student-mle-oracle-deps-20261003 \
  PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/tmp/student-location-correctness-numba \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/analyze_location_chains.py \
  --out tmp/location_chain_diagnostics_reproduction
```

The `PYTHONPATH` points to this environment's existing isolated oracle dependency
directory, needed by the reused reference helpers. The output path must be new.
No new MLE oracle evaluation or MCMC iteration is performed by this analysis.

Four [focused tests](../../tests/test_location_chain_diagnostics.py) pass. They
check quantile axes and interpolation, labelled-versus-ordered observations,
fixed constraints versus stuck chains, serial dependence, chain disagreement,
and explicit nonfinite-data failure. The test run took 132.21 seconds including
cold imports; there was no timeout or collection failure. The full sampler
suite was not rerun because production code is unchanged.

Analysis and plotting took 130.81 seconds after imports. All 360 protected input
hashes matched before and after. Original study evidence, samplers, selectors,
and raw chains were unchanged. See the
[completion record](evidence/location_chain_diagnostics_2026-10-12/completion.json),
[scalar CSV](evidence/location_chain_diagnostics_2026-10-12/scalar_diagnostics.csv)
and [full per-chain/split diagnostics](evidence/location_chain_diagnostics_2026-10-12/diagnostics.json).

For subsequent comparisons, retain both arrays and these labelled, ordered and
quantile diagnostics as standard outputs. Use block Gibbs/direct methods for
the present practical comparisons. Pure Student RATTLE needs better latent-tail
exploration; Logistic RATTLE needs a focused check of label mixing before it is
ranked as a reliable full-latent sampler. No such follow-up is started here.
