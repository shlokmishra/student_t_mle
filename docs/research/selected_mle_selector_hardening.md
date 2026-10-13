# Student location-MLE selector hardening

Date: 2026-10-10. Scope: fixed-scale Student location estimation. This implements the focused selector work from the [target audit](selected_mle_target_audit.md) and [reference validation](selected_mle_reference_validation.md).

**Result:** the numerical likelihood-tie convention is removed; ambiguous comparisons use exact rational bounds. The corrected selector agrees with the independent oracle in 96 ordinary and 61 adversarial datasets. All 62 focused pytest checks passed across the final suite and one additional failure-reporting regression. No significant warmed runtime overhead was observed in the small benchmark. Fresh selected-MLE references are supported on the tested domains, with explicit failure accounting; universal numerical certification is not claimed.

## Changes and caller compatibility

The implementation is [models/student_location_mle.py](../../models/loc/student/mle.py). Its two model callers remain [loc_student.get_mle](../../models/loc/student/score_gibbs.py) and [loc_affine.global_location_mle](../../models/loc/affine.py). Those callers still receive a float on success and propagate numerical failures. No Gibbs, RATTLE, affine acceptance, KDE, model-registry, or historical-evidence code was changed.

The fast algorithm retains deterministic EM incumbents, interval score/derivative searches and likelihood pruning for global selection. Changes are:

- **Near ties:** a small likelihood gap triggers exact comparison; it never authorizes choosing the lower location. Exact symmetry or the analytic two-point argument can prove a genuine tie. The lower location is then selected.
- **Flat and degenerate maxima:** pairs and equal replication of two observations use exact rational arithmetic on the original binary inputs. A binary-grid square-root enclosure avoids cancellation at the transition. Other flat/ambiguous intervals use exact polynomial root isolation.
- **Complete-root requests:** the public stationary_points search disables incumbent pruning, so inferior extrema remain eligible. It includes degenerate roots through the exact fallback, and detects candidates that coalesce during polishing. An unrepresentable inferior root can cause enumeration to fail even when global selection succeeds.
- **Standardization:** selection sorts observations for deterministic permutation invariance and uses an observed order statistic as the center, avoiding an overflowing even-sample median. Returned roots are checked against original observations. Centering can also change the winning likelihood: a padded reconstruction-error budget widens likelihood pruning/comparison margins and triggers exact arithmetic for substantial data loss.
- **Failures:** StudentMLEError carries a reason such as output_precision, dynamic_range, exact_size_limit or selection_unresolved. Both public estimator functions validate positive finite tolerances and real, finite, nonempty one-dimensional data. Diagnostic score/log-likelihood functions reject nonfinite locations and use extended precision for extreme observations.

The existing default value of location_tol remains $2\times10^{-13}$, but its contract is now **absolute location accuracy in original data units**. Previously it controlled a relative bracket width after standardization. This deliberate tightening makes large-offset failures visible rather than allowing the tolerance to grow with location. A caller may explicitly request a looser representable tolerance.

An optional return_diagnostics argument on global_mle returns the estimate with its method and fallback information. Exact paths include an absolute enclosure-error bound; the fast path reports numerical diagnostics and does not claim such a proof.

SymPy is loaded lazily for ambiguous cases and is declared in [requirements.txt](../../requirements.txt). Validation used SymPy 1.14.0 from the existing isolated dependency directory; the project environment was not modified.

## Mathematical safeguards and limits

For original binary-float observations treated as exact rationals,

$$D(m)=\prod_i[k+(x_i-m)^2].$$

The Student likelihood is maximized exactly where $D$ is minimized. The production fallback independently builds the rational polynomial in coordinates shifted by an exact observed value, isolates every real root of $D'$, and bounds $D$ on each root interval. A unique possible minimum proves the global winner. A symmetric mirror pair proves equal objective values when all other candidates are excluded. This is separate code from the unchanged [validation oracle](../../scripts/selected_mle_reference_oracle.py).

The fallback refines intervals through 16, 32, 64 and 128 digits. Bounds that still overlap without a proved tie raise selection_unresolved. Genuine nonsymmetric ties are not guessed. Exact fallback is limited to $n\le20$; larger ambiguous datasets raise exact_size_limit. Constant samples and the analytic two-point/equal-replication cases do not require SymPy.

The two-point square root is enclosed using an integer square root on a rational binary grid. Before returning a float, both analytic and polynomial paths compare its exact rational value to the entire root enclosure. They raise output_precision if the requested absolute tolerance cannot be met. Global selection checks only its selected root; complete enumeration checks every root.

For the standardized reduced objective $-\sum_i\log(1+r_i^2)$, the absolute derivative in each observation is at most one. If standardization perturbs observations by total standardized amount $B$, each objective changes by at most $B$. Pruning and winner comparisons therefore include an extra $2B$ margin. The implementation pads the float reconstruction for its own roundoff and includes a rounded-square-root scale allowance. Original-data score/curvature checks address root accuracy separately. These are conservative numerical safeguards, not a formally verified directed-rounding library.

The ordinary interval path remains floating-point, with heuristic outward padding and Newton checks. Its search budget is 100,000 intervals; ambiguity or budget exhaustion invokes the bounded-size exact fallback. General samples whose standardized magnitude exceeds $10^{50}$ are rejected before searching. The analytic pair path can still handle much larger inputs.

Exact polynomial bounds have finite precision/size limits, but SymPy root isolation has no hard in-function wall-clock guarantee. Batch wrappers should retain a time ceiling. Extended-precision overflow diagnostics rely on the platform's long-double range; a nonfinite diagnostic raises explicitly. These limits must accompany any numerical-certification claim.

## Correctness evidence

The [validation driver](../../scripts/validate_student_mle_hardening.py) compares the preserved original selector with the corrected code and the unchanged independent oracle. It replays the prior 57-case audit, three recorded near-tie cases, and a centering counterexample. Each record stores data, root lists, winner location, likelihood gap and score residual separately.

| Global-selection checks | Original maximum location error | Corrected maximum location error | Corrected maximum likelihood gap | Corrected maximum reduced-score residual |
|---|---:|---:|---:|---:|
| 96 independent ordinary draws | $1.51\times10^{-13}$ | $4.44\times10^{-16}$ | $1.78\times10^{-15}$ | $4.30\times10^{-16}$ |
| 61 adversarial/replayed datasets | 7.48331 | $2.22\times10^{-16}$ | $7.11\times10^{-15}$ | $1.11\times10^{-15}$ |

All 157 global selections succeeded. Oracle-location comparisons are to its rounded float; exact-path enclosure checks additionally address real-root rounding. Likelihood gaps in this table are float diagnostics, so sub-roundoff differences require the exact product bounds or the previous high-precision loss calculation.

Specific regressions:

- The $k=2$ near tie made by adding $2^{-35}$ to one of five observations at $-4$ now selects $+3.74165738677$, rather than the negative mode. The original likelihood loss was only $4.76\times10^{-22}$, despite the 7.48331 location error.
- At $x=\pm\operatorname{float}(\sqrt2)$, the smaller split maximum near $-1.65357898604\times10^{-8}$ is retained. The exact degenerate case $k=1,x=(-1,1)$ retains its unique root zero.
- The corrected root enumerator returned complete, distinct oracle-matching lists on **57 successful requests**. Four other requests explicitly failed output_precision because distant inferior roots could not meet the absolute tolerance. There were no silently missing roots among successful requests. The original enumerator omitted 34 roots across 16 datasets in this expanded replay.
- At $k=2,x=(-10,-10,10,10,10)$, all three stationary roots are retained, including the inferior maximum and intervening minimum.
- At $k=2,x=(-10^{200},10^{200})$, selection is finite and equals $-10^{200}$; score and likelihood diagnostics are finite. General $x=(0,1,10^{200})$ now raises dynamic_range in about 0.10 ms instead of spending about 72 ms exhausting the original search budget.
- For $x=2^{40}+(0,0.25,1.5)$, default accuracy raises output_precision. An explicitly looser $10^{-4}$ tolerance returns the independently checked rounded winner.
- Original centering erased the difference between 0 and $10^{-4}$ in $(0,10^{-4},10^{14},2\,10^{14},3\,10^{14})$. The corrected result agrees with the oracle near $5\times10^{-5}$.
- A second competing-cluster case has a tiny score at the wrong mode: $x=(-a,-a,a,a,c,c,c,c+0.015625)$, $a=0.0078,c=10^{14}$. The corrected selector identifies the upper winning mode, reports default float accuracy as impossible, and returns $c$ under an explicit 0.01 tolerance with enclosure error 0.0039059. Checking stationarity alone would have missed this global-selection defect.

Tests cover exact versus near ties, multimodality, degeneracy, extreme ranges, large offsets, translation, permutation, invalid inputs/tolerances, and a tied $n=21$ sample that explicitly reports the exact-fallback size limit.

The final focused suite passed **61 tests in 31.44 seconds**; the subsequently added size-limit regression passed separately in 2.89 seconds. These include existing selector, affine-caller and weighted-reference tests. No full-repository test pass is claimed.

An earlier pass had 37 successes and two failures: permutation changed the last rounding unit, and a near-flat score assertion was below float summation precision. Sorting fixed the former; the latter test now uses a realistic score threshold while preserving independent location checks. A subsequent intermediate pass had 40 successes. One initial evidence-driver attempt stopped on a NumPy integer JSON-serialization error; that was corrected and its log/partial outputs preserved. No final pytest timeout or collection failure occurred.

## Small performance comparison

The [frozen configuration](evidence/selected_mle_selector_2026-10-10/config.json) specifies 128 datasets per group, three warmed timing repetitions, alternating original/corrected timing order, and fixed seeds. All benchmark processes use one CPU. First-call costs are separate: with different cache states, the final run's first selector calls took about 0.24 s original and 4.35 s corrected. These are not a controlled comparison of cold starts.

| $k,n$ | Original median microseconds/call | Corrected median microseconds/call | Change | Exact fallbacks / failures |
|---|---:|---:|---:|---|
| 2, 3 | 55.65 | 54.60 | -1.9% | 0 / 0 |
| 2, 10 | 71.58 | 69.97 | -2.2% | 0 / 0 |
| 3, 10 | 59.36 | 59.99 | +1.1% | 0 / 0 |
| 1, 50 | 197.62 | 202.74 | +2.6% | 0 / 0 |

There were **zero high-precision fallbacks or failures in 512 ordinary datasets**. These short timings do not establish a speedup; they show no material overhead against the declared 25% threshold. The Cauchy $n=50$ group is a throughput/failure stress check, not an independently certified $n=50$ oracle study.

The 61 adversarial global checks used 43 ordinary searches, nine analytic exact cases, and nine SymPy fallbacks. Successful polynomial fallbacks took roughly 6.8–33.5 ms once dependencies were imported. The fallback is deliberately expensive and selective.

## Readiness and reproduction

**Ready for fresh reference generation on the tested small-case domain**, with recorded selector hashes, exact-tie convention, tolerances and failures. Always propagate an unresolved calculation: discarding or repeatedly replacing failed draws biases the reference. Monitor fallback/error rates when moving beyond these cases, especially larger ambiguous samples or extreme offsets.

Legacy standardized-error caches must be regenerated or explicitly checked against the updated selector; their existing version labels were not changed in this task. This does not resolve the prior shifted-posterior importance-ESS/rare-tail problem, validate all $n=50$ selections, or change any sampler's conditioning target.

Run from the project root, with the isolated oracle dependencies available:

    export PYTHONPATH=/tmp/student-mle-oracle-deps-20261003
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 timeout 300s .venv/bin/python -m pytest tests/test_student_location_mle_hardening.py tests/test_student_global_mle_audit.py tests/test_loc_affine.py tests/test_selected_mle_reference_validation.py -q --assert=plain
    timeout 600s .venv/bin/python scripts/validate_student_mle_hardening.py --config docs/research/evidence/selected_mle_selector_2026-10-10/config.json

If that temporary dependency directory is unavailable, install project requirements or use the isolated SymPy installation command in the prior reference report. The driver has an additional 300-second post-warmup ceiling. Change the output path in a configuration copy to preserve this evidence when regenerating.

The [evidence directory](evidence/selected_mle_selector_2026-10-10/verification.json) contains complete numerical records, test logs, source hashes and preservation checks. The original selector is preserved as [student_location_mle_before.py](evidence/selected_mle_selector_2026-10-10/student_location_mle_before.py). Of the 460 pre-existing files hashed at the start, only the intended selector and dependency declaration changed. All existing sampler sources and experimental evidence were preserved. Work stops here; sampler refactoring has not begun.
