# Guarded n=3 selected-MLE endpoint validation

Date: 2026-10-10. Unit-scale Student location; k=1,2,3. **Outcome:** an optional, bracket-certified endpoint check preserves the tested block chains exactly while reducing warmed iteration cost by **1.58–2.05×**. All 68,247 endpoint checks reached in the matched study were resolved without the full selector. This accelerates the existing target; it does not change the block proposal or curvature correction.

Implementation remains **opt-in**, using `n3_endpoint_fast_path=True` in the block sampler's parameters. The default remains the original endpoint-validation path. No other existing sampler, global selector, reference bank, or historical result was changed.

## Mathematical review and acceptance certificate

Read the [geometry proof](student_n3_geometry_selected_mle.md), [block derivation](cheap_selected_mle_gibbs.md), [selector hardening](selected_mle_selector_hardening.md), and [pair-Gibbs validation](student_selected_mle_gibbs_validation.md). The n=3 argument assumes three **finite**, real observations, fixed positive scale, k>0, and location-only maximization. After dividing residuals by sigma sqrt(k), the Student exponent is positive and the likelihood maximizer minimizes the same sixth-degree denominator polynomial. The central-component quartic positivity argument establishes that every central stationary root is the unique global maximizer and has J>0.

For arbitrary data with median b, the score at b-sigma sqrt(k) is strictly positive: the median contributes the positive score maximum, another observation contributes a strictly positive score, and the remaining observation cannot contribute less than the negative score maximum. The opposite endpoint has strictly negative score. Every root strictly inside this interval is central and hence global. Existence plus the central-root result therefore establishes **exactly one root in this interval**, and a unique global MLE for every finite dataset. This does not assert that there are no stationary roots outside the interval. The exact-score-zero median criterion follows; it is not applied as an equality test to rounded observations.

At unit scale, write C(u;x)=sum_i (x_i-u)/(k+(x_i-u)^2). For the existing release tolerance epsilon, the guarded path constructs inward floating endpoints L,H and proves

\[
t-\epsilon\le L<t<H\le t+\epsilon,
\quad (b-L)^2<k,\quad(b-H)^2<k,
\quad C(L;x)>0>C(H;x).
\]

The intermediate value theorem and uniqueness in the median interval imply that the **selected global root** belongs to [L,H]. This is a root-location enclosure, not the estimate |C(t)/J(t)|. It works for approximate score-zero input without dropping the linear term from the likelihood polynomial. It certifies the numerical release tolerance, not literal real-number equality T(x)=t for arbitrary rounded data.

Score signs and squared distances use outward-rounded interval arithmetic on the **original binary input values**. Subtraction, squaring, positive-denominator addition, division, and summation are individually enclosed with `nextafter`; no approximate square root is used to certify the median interval. Interval division includes all four endpoint combinations. Overflow or an unusable denominator makes the check inconclusive. Exclusion uses an **outward** release window: a root interval disjoint from that entire window proves a support failure. Using the inward acceptance bracket for exclusion would incorrectly exclude its tiny omitted endpoint slivers.

Prerequisites and conservative fallback guards are:

- Three finite real observations, finite release, k>0, and positive finite location tolerance; the enclosing sampler already validates these and unit known scale.
- Representable strict L<t<H with outward displacement bounds <=epsilon. Large offsets can make this impossible.
- Existing score, Gram, and curvature checks still run before the endpoint test. The existing cheap nonpositive-J rejection is unchanged.
- J*k>sqrt(machine epsilon) and the original local displacement guard must pass to attempt certification. These are **fallback gates**, not proofs of root accuracy or new target cutoffs.
- Any squared residual within a relative 1e-8 neighborhood of the inverse fold requests fallback. This conservative chart-neighborhood guard does not exclude a valid fold from the target.
- Both bracket endpoints must be strictly inside the unique-root median interval, and the outward score bounds must have strictly opposite signs. An uncertain sign requests fallback.

The arithmetic enclosure argument assumes binary64 IEEE basic operations with gradual underflow, correctly rounded `nextafter`, and no unsafe reassociation/fastmath. The Numba kernels use no fastmath. Exact-rational tests exercise normal, subnormal, cancellation, and large-magnitude inputs. This is not a formal verification of the compiler or floating environment; an environment that flushes subnormals to zero is outside this contract.

## Implementation and unchanged target

Only [loc_student_selected_mle_block.py](../../models/loc/student/selected_block.py) was modified among pre-existing files. New [student_n3_endpoint.py](../../models/loc/student/n3_endpoint.py) contains independently testable interval/certificate machinery. The preserved original source and exact change patch are in the [evidence directory](evidence/student_n3_fast_path_2026-10-10/).

The fixed-mu reversible score proposal K^m, both inverse branches and Jacobian weights, parameter MH update, and outer log(J'/J) acceptance calculation are unchanged. Certification occurs **after** the existing cheap curvature MH draw. It consumes no random numbers. Exact-arithmetic selected-MLE invariance therefore follows from the previously justified block correction with the same support indicator and J. The Gram/coarea factor is not removed. The n>3 numerical path still uses the original selector; a matched n=10 test checks unchanged chains when the option is enabled.

An inconclusive check calls the unchanged `selected_state` validation path. That path invokes the hardened selector when its own numerical guards permit it; its guards can also raise **before** selection. In particular, the known near-flat score/curvature displacement failures remain explicit failures, not ordinary rejections and not silently accepted states. This task does not repair those older guards. Failed block computations retain the original partial-chain/iteration reporting.

Certified states store the release as a bracketed approximation with an explicit root enclosure and error bound, rather than claiming a freshly computed exact MLE. `max_selected_mle_residual` includes that bound on these states, and `selected_mle_residual_kind` identifies the changed diagnostic meaning. New counters distinguish fast attempts, certifications, exclusions, and fallback requests from full validation calls. Original initial-state validation is retained in both paths.

## Numerical correctness

Focused tests: **81 passed in 4.98 seconds**, covering the new endpoint tests, existing block tests, and n=3 geometry tests. [Final log](evidence/student_n3_fast_path_2026-10-10/pytest_focused.log). The first new-test run had 26 passes and one incorrect assertion about fallback priority: `(0,1,1e200)` at k=1 triggers its exact fold before overflow. The overflow test was separated from the fold; the initial failure log is preserved. The separate original block suite passed 18 tests in 117.83 seconds, including cold filesystem/import overhead. No collection error or test timeout occurred in this task.

Tests and the separate [endpoint audit](evidence/student_n3_fast_path_2026-10-10/endpoint_audit.json) include:

- All 25 chamber witnesses for each k=1,2,3, checked against the current hardened selector and saved independent rational-root intervals. All 39 central witnesses certify; no tail witness falsely certifies.
- All 39 additional archived fold/asymptote/near-flat/competing-mode/degenerate cases. Across all **114** archived cases, there are 54 certifications, 18 certified exclusions, and 42 inconclusive checks. The latter include negative/flat curvature cases that the sampler can reject or abort before calling this helper. The original validator still explicitly fails on the six previously recorded near-flat/degenerate cases.
- Six **fresh independent rational-oracle calculations** on shifted central and inferior strict-local-maximum witnesses, across k=1,2,3 with releases 1.25 and -0.75. Their certifications/exclusions agree with the oracle.
- Exact rational containment of 72 score intervals, including subnormals and observations up to 1e100; huge-square overflow requests fallback. Binary inputs are treated as exact rationals, not approximate algebraic score-zero points.
- Tiny nonzero score offsets, actual root displacement below/above tolerance, inward versus outward window boundaries, translation/permutation checks, fold proximity, unrepresentable release brackets at 2^40, and near-flat tiny-score states. Small score alone never authorizes certification.
- Injected selection failure after an ambiguous fold certificate propagates as an exception.
- Identical-seed short chains with nonzero releases, k=1,2,3, and the unchanged n=10 path.

In the performance study, **24 paired chains** (48 runs) have bit-identical full retained x and mu arrays. Acceptance/rejection counts and sampled proposal records also agree, as recorded in [matched_decisions_check.json](evidence/student_n3_fast_path_2026-10-10/matched_decisions_check.json). Maximum score residual is 1.15e-14; maximum certified release-location bound is 1e-10. Eighteen fresh oracle checks on retained/proposed datasets have zero errors and a maximum selector-versus-oracle location discrepancy of 2.23e-16. There were no false certifications or resolved false exclusions in these checks.

## Bounded experiment and reference comparison

The [pre-run configuration](evidence/student_n3_fast_path_2026-10-10/config.json) freezes four existing dispersed valid starts per case, 4,000 blocks per chain, 1,000 burn-in, m=4/16, k=1/2/3, release 0, prior N(0,2²), sigma_z=0.1, and sigma_mu=0.45. Seeds are `2026105100 + 100*k + 10*m + chain_index`. No tuning, additional seeds, or run extension was performed. The 192,000 recorded blocks include both paths; 32-block warmups for each path/chain are additional and excluded from timing. Each chain has a 60-second ceiling; the study has a 240-second ceiling and aborts on a mismatch/failure. It completed in 22.89 seconds after imports.

References are the saved raw weighted banks: 8,192 standardized draws for k=1/3 from the geometry study and 32,768 for k=2 from the pair-Gibbs study. Their hashes and recomputed summaries are checked before execution; previous bootstrap uncertainty is carried forward. No unchecked cache or fresh reference campaign is used. Reference importance ESS is approximately 7,637 / 32,009 / 8,081. Chamber uncertainty uses the independent-particle SNIS formula; chain uncertainty uses the larger of batch and between-chain estimates. The 4-combined-MCSE comparisons and R-hat<=1.01, bulk/tail ESS>=400 screens are finite-run diagnostics, not convergence proofs.

| k | m / reference | Mean | SD | q05 | q50 | q95 | Median max residual | P(max residual>6) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | reference | -0.01506 | 0.89866 | -1.47169 | -0.01465 | 1.42437 | 2.43291 | 0.23637 |
| 1 | 4 | -0.01096 | 0.92488 | -1.53848 | -0.00058 | 1.43478 | 2.46660 | 0.24983 |
| 1 | 16 | 0.05471 | 0.90585 | -1.32379 | 0.01457 | 1.58139 | 2.52328 | 0.25342 |
| 2 | reference | -0.00342 | 0.77107 | -1.24411 | -0.00958 | 1.23777 | 1.46790 | 0.06688 |
| 2 | 4 | 0.02084 | 0.78165 | -1.22125 | 0.00807 | 1.27003 | 1.45117 | 0.06933 |
| 2 | 16 | 0.00641 | 0.73482 | -1.18485 | 0.00815 | 1.20351 | 1.43855 | 0.06408 |
| 3 | reference | -0.00630 | 0.70741 | -1.14251 | -0.01629 | 1.13187 | 1.23332 | 0.02372 |
| 3 | 4 | -0.01247 | 0.70973 | -1.12993 | -0.01643 | 1.13272 | 1.29747 | 0.02550 |
| 3 | 16 | 0.00845 | 0.71371 | -1.18044 | 0.00600 | 1.17237 | 1.23522 | 0.02367 |

The complete summaries, max-residual q90/q95, MCSEs, diagnostics and chamber probabilities are in [comparisons.json](evidence/student_n3_fast_path_2026-10-10/comparisons.json). Both paths share these estimates because their chains are identical. All six posterior/latent summary screens pass; the largest summary difference is 1.99 combined MCSE. Every chain visits all 13 central chambers, with zero retained tail-component states. All chamber screens pass; the largest discrepancy is 2.91 combined MCSE. The independent reference latent-tail MCSEs are 0.00468, 0.00136, and 0.00172 for k=1,2,3; sampled nonzero tails and finite uncertainty are preserved.

Five cases pass all diagnostic screens. **k=1,m=4 does not:** parameter tail ESS is 360 (bulk ESS 421, R-hat 1.00795), below the fixed 400 threshold. Its latent diagnostics pass. This is a short-run precision limitation shared exactly by the original and guarded paths, not evidence of a changed kernel. No extra run was commissioned to make it pass.

## Fresh matched performance

Preserved pre-change source and guarded source are timed in the same process, on one pinned CPU with one numerical thread. Run order alternates between original-first and guarded-first across chains. Full-path warmups precede each pair. Timings measure the existing iteration-loop convention; setup, initial selector validation, imports and compilation are excluded for both paths. Selector counts below include the four initial validations in each row. These are preliminary measurements, not cross-environment comparisons to older reported times.

| k | m | Original us/block | Guarded us/block | Speedup | Original/full calls | Guarded/full calls | Guarded accepted blocks/sec |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 74.20 | 41.30 | 1.80× | 8,336 | 4 | 11,723 |
| 1 | 16 | 90.54 | 57.14 | 1.58× | 8,043 | 4 | 8,175 |
| 2 | 4 | 88.42 | 43.73 | 2.02× | 12,695 | 4 | 18,064 |
| 2 | 16 | 113.71 | 62.51 | 1.82× | 12,199 | 4 | 12,154 |
| 3 | 4 | 92.24 | 45.03 | 2.05× | 13,839 | 4 | 19,199 |
| 3 | 16 | 109.26 | 60.18 | 1.82× | 13,159 | 4 | 13,659 |

Totals: **68,271 -> 24** full validation calls, avoiding all **68,247 post-initialization calls**, or 99.965% including initial checks. Endpoint certification/exclusion succeeds in 100% of attempted ordinary checks, with **zero fallback requests** in the recorded chains. This empirical frequency does not bound all adversarial inputs. Positive-J endpoints resolved as non-global account for 585/565, 52/40, and 1/2 exclusions for k=1/2/3 at m=4/16. Overall block acceptance is unchanged: 48.42/46.71%, 78.99/75.97%, and 86.46/82.21% respectively. Certificate cost is about 6.8–7.3 us per reached endpoint, including Python validation and result construction.

| k | m | Parameter bulk ESS/sec, original -> guarded | Max-residual bulk ESS/sec | Tail>6 ESS/sec |
|---|---|---:|---:|---:|
| 1 | 4 | 355 -> 637 | 1,640 -> 2,947 | 1,875 -> 3,368 |
| 1 | 16 | 373 -> 591 | 1,406 -> 2,229 | 1,649 -> 2,612 |
| 2 | 4 | 393 -> 796 | 3,048 -> 6,163 | 5,190 -> 10,494 |
| 2 | 16 | 390 -> 710 | 2,530 -> 4,602 | 4,656 -> 8,471 |
| 3 | 4 | 474 -> 970 | 1,139 -> 2,333 | 6,680 -> 13,681 |
| 3 | 16 | 441 -> 801 | 2,755 -> 5,001 | 5,777 -> 10,488 |

ESS/sec includes burn-in iteration cost and uses retained functional ESS. The improvement comes entirely from saved validation time; this does not improve mixing per block. It is not a new claim that constrained MCMC beats independent rejection/weighted simulation, and no new direct-simulation benchmark was run.

## Limitations, recommendation, and reproduction

The guarded check is mathematically justified on its stated floating-point contract and passes the tested exact-oracle comparisons with no false certifications. It is not a universal certification of the complete floating sampler. Existing inverse-chart failures, score drift, Gram/curvature guards, tolerance-boundary ambiguity in the original selector path, and the inherited parameter-uniform guard remain. Near-flat and extreme cases can still fail explicitly; folds remain valid target states even when numerical proposal machinery cannot resolve them. Scaling generalization is proved geometrically but this implementation deliberately supports only the existing unit-scale block prototype. Nothing here extends the theorem or optimization to n>=10.

**Recommendation:** the guarded option is ready to become the n=3 unit-scale block sampler's default after review, retaining the original path and explicit fallback policy. It is currently opt-in so the default comparison remains available without changing callers. The evidence supports a roughly twofold ordinary-case speedup with identical exploration, not orders-of-magnitude acceleration or a fix for heavy-tailed parameter precision. No RATTLE work or broader stress campaign follows this task.

Exact executed commands:

```bash
PYTHONPATH=/tmp/student-mle-oracle-deps-20261003 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_student_n3_endpoint.py tests/test_student_selected_mle_block.py tests/test_student_n3_selected_geometry.py -q --assert=plain
PYTHONPATH=/tmp/student-mle-oracle-deps-20261003 .venv/bin/python scripts/validate_student_n3_fast_path.py
PYTHONPATH=/tmp/student-mle-oracle-deps-20261003 .venv/bin/python scripts/audit_student_n3_fast_path.py
```

The external dependency directory supplies the previously used SymPy/mpmath; an environment satisfying the repository requirements can instead provide these normally. The experiment refuses to overwrite existing raw trajectories. Replay in a fresh workspace with the saved configuration, original source snapshot, verified reference banks, and an empty study raw directory. Raw chains reside under `tmp/student_n3_fast_path_20261010/`; per-chain portable provenance contains their paths, SHA256 hashes, seeds, starts, source hashes, timings, and counters. The preservation check covers 866 pre-existing source/evidence/artifact files: only the authorized block module changed, and none disappeared.
