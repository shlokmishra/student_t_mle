# Student selected-MLE RATTLE: endpoint correction and bounded validation

Date: 2026-10-10. **The endpoint correction is mathematically justified on regular
projection branches and passes the focused numerical checks. The n=3 case passes
the declared small-run screens with weak tail precision. Both n=10 cases fail
latent mixing screens; neither is validated as a joint posterior simulation.**
The experiment was stopped at its frozen budget, without tuning or extending it.

This work builds on the [target audit](selected_mle_target_audit.md),
[reference validation](selected_mle_reference_validation.md),
[selector hardening](selected_mle_selector_hardening.md),
[shifted reference follow-up](shifted_student_reference_followup.md), and
[corrected Gibbs validation](student_selected_mle_gibbs_validation.md).
Implementation facts, mathematical statements, and experimental evidence are
separated below. All evidence is in
[evidence/student_selected_mle_rattle_2026-10-10/](evidence/student_selected_mle_rattle_2026-10-10/).

## 1. Target and endpoint acceptance: mathematical derivation

For fixed release t and unit scale, set y=x-t,
C(x)=sum psi(y_i), g=grad C, G=g'g, and J=sum psi'(y_i), where
psi(y)=y/(k+y²) and psi'(y)=(k-y²)/(k+y²)². On a regular branch of the selected
MLE statistic T, implicit differentiation gives grad T=g/J. The selected-MLE
conditional position density relative to Hausdorff surface measure is therefore

\[
 \rho_T(x\mid\mu)\ \propto\
 \mathbf1\{T(x)=t\}\,F_\mu(x)\,\frac{J(x)}{\sqrt{G(x)}}.
\]

Here J>0 at a nondegenerate selected maximum; negative J cannot be in regular
selected-MLE support. Positive J alone is insufficient: it also includes
inferior local maxima. Exact likelihood ties use the hardened selector's
smaller-location convention. Singular estimator branches and ties at which T
is not differentiable are outside this regular-region derivation; this work
does not provide an additional distribution on singular strata.

The shared integrator uses **identity mass** and tangent momenta p satisfying
g'p=0. Projecting an ambient N(0,I) vector gives the correctly normalized
N(0,I) law on this (n−1)-dimensional tangent fiber. With identity mass, the
cotangent Liouville volume is the product of position and tangent-fiber
Hausdorff measures. Thus the desired extended density is

\[
 \widetilde\rho_T(x,p\mid\mu)\propto
 \mathbf1\{T(x)=t\}\,J(x)\,\exp[-H_{score}(x,p)],
\quad
 H_{score}=-\log F_\mu(x)+\tfrac12\log G(x)+\tfrac12p'p.
\]

There is no additional momentum Gram determinant. A different constant mass
would require auditing its momentum and measure conventions; arbitrary mass
matrices are deliberately unsupported in the new wrapper.

Let Phi be one shared RATTLE step and S(x,p)=(x,−p). On a regular implicit
projection branch, RATTLE is symplectic and hence preserves canonical volume;
its defining equations obey S Phi S=Phi^{-1}. Therefore F=S Phi^L is a
volume-preserving involution on its admissible phase domain. For that proposal,

\[
 \log r=-[H_{score}(x',p')-H_{score}(x,p)]
             +\log J(x')-\log J(x)
\]

when both endpoints are selected-MLE valid, and acceptance is zero outside that
support. The final target Hamiltonian is H_T=H_score−log J on support. Guiding
with H_score rather than H_T changes proposals, not this MH proof. No curvature
gradient is needed. The momentum flip leaves kinetic energy unchanged.

Failure rejection also needs a symmetric admissible domain. In exact
arithmetic, require that forward and reverse solves succeed and that
F(F(x,p))=(x,p). Membership then holds for both paired phase points. A position
return alone does not establish this phase involution. Finite tolerances
approximate these identities; they do not constitute universal numerical
certification.

**Intermediate support:** Phi may visit score-manifold positions with J≤0 or
T(x)≠t. This does not invalidate a regular reversible endpoint proposal. Global
selection is an endpoint density factor, not a constraint on the guiding
trajectory. A zero Gram or singular projection branch is a geometric obstacle;
a nonselected intermediate maximum is not. No selected-MLE test is made at
integrator substeps. This is a mathematical permission, not a claim that this
small experiment observed a particular invalid-intermediate/valid-endpoint
trajectory.

**Cheap rejection:** draw one U~Uniform(0,1), compute a=min(1,exp(log r)) without
the indicator, and reject immediately if U≥a. The indicator can only reduce a.
If U<a, evaluate global selection and accept exactly when the endpoint is
valid. Use the same U, with no second MH test. An unresolved selector
calculation on a surviving candidate raises; it is not counted as a support
rejection. A selector calculation skipped after cheap rejection could not
change the outcome.

**Parameter block:** J and the selection indicator depend on x,t, not the
unknown mu. Consequently p(mu|x) remains proportional to prior(mu)F_mu(x).
The existing shared parameter MH function is reused unchanged.

## 2. Actual integration path and separate implementation

New files:

- [models/loc_student_selected_mle_rattle.py](../../models/loc/student/selected_rattle.py):
  `run_rattle`, `rattle_step`, `phase_proposal`, geometry and selection helpers.
- [tests/test_student_selected_mle_rattle.py](../../tests/test_student_selected_mle_rattle.py):
  independent energy, chart-volume, reversal, support and failure tests.
- [scripts/validate_student_selected_mle_rattle.py](../../scripts/validate_student_selected_mle_rattle.py):
  frozen chains, reference comparisons, oracle audit and matched benchmarks.

The unchanged [legacy RATTLE](../../models/loc/student/score_rattle.py) uses a half
potential kick, projection along the **old** constraint gradient, the associated
position multiplier divided by step size in the half momentum, a second kick,
and tangent projection along the **new** gradient. Those are standard RATTLE
implicit equations. The guiding force includes the score coarea +0.5 log G
term. Fixed-direction Newton damping/line search selects a root; it does not
change those equations at exact convergence.

The new Numba trajectory wrapper calls the legacy one-step kernel and audits
old/new gradient transversality |g_old'g_new|/sqrt(G_old G_new) at every completed
step. Nonzero transversality identifies a locally regular implicit position
root. The wrapper then checks the reverse trajectory in **both position and
momentum**, with tolerances 1e−8 each. Legacy checks only reverse position,
with default tolerance 5e−3; its behavior remains unchanged. A regression test
confirms that successful audited trajectories give the same phase arrays as the
unchanged multi-step legacy kernel.

At the endpoint, the new wrapper adds log J and applies hardened global
selection. It caches the current state's verified geometry/selector result,
with a dataset/release/k/scale token to prevent stale reuse. Selection is not
repeated for parameter updates or at substeps. Negative curvature is resolved
support failure; ambiguous zero curvature, nonfinite arithmetic, inadequate
score/curvature root-location resolution, and materially lossy physical-state
reconstruction raise explicit errors. The score/J displacement estimate is a
local accuracy diagnostic, not a rigorous root-error bound. Failed bounded
projection, nonregular projection branches and failed phase reversal are
proposal-domain rejections and are counted separately.

There is no endpoint curvature floor. A retained Gram below the shared
projection resolution raises. Although the shared potential has its historical
Gram floor, it is inactive in the completed regular trajectories tested here.
Known fixed scale is supported by integrating standardized residuals
(x−t)/scale, with identity mass there (physical mass scale^{-2}I). Constant
coordinate, likelihood and curvature factors cancel in MH ratios. The scale
case is checked in unit tests; the reference experiment is unit scale only.
Large offsets/scales that materially alter a standardized phase endpoint during
reconstruction fail explicitly.

The latent acceptance uniform is untruncated. The unchanged parameter update
retains its historical EPS_U=1e−12 uniform guard; this inherited finite precision
approximation and finite projection tolerances qualify literal numerical
exactness. No universal bound on cumulative stationary-law error is claimed.

A read-only mathematical reviewer independently checked the endpoint algebra,
momentum and scale conventions, implicit-root regularity and whole-phase
reversal argument. Its two numerical recommendations (reconstruction accuracy
and early constraint-location diagnostics) were incorporated before the final
tests and chains. No legacy Gibbs, RATTLE, affine, selector, KDE or location-scale
file was edited. The preservation audit checked all **626 pre-existing files**.

## 3. Numerical correctness and failure tests

The focused suite passed **101 tests in 21.80 seconds**. Coverage includes:

- Independent J/G calculations; selected-state checks at unit and nonunit scale.
- Endpoint ratio versus an independently written full selected-target energy;
  reverse-ratio antisymmetry, momentum-flip energy equality, and MH flux equality.
- Full forward/reverse phase return, score and tangent constraints, and exact
  array agreement with the shared successful integration path.
- Finite-difference canonical chart Jacobians at n=3 and n=5: determinant≈1 and
  D'ΩD≈Ω within absolute tolerance 2e−7. These local tests support the algebraic
  symplectic argument; they do not certify every numerical projection branch.
- A reverse solve that returns the correct position but wrong momentum is
  rejected. Bounded projection failures reject before invoking the selector.
- Positive-curvature, score-zero endpoints whose selected global MLE is elsewhere
  are rejected. Cheap MH rejection demonstrably skips global selection.
- Injected unresolved selector errors propagate; degenerate Gram/curvature,
  nonfinite energy/parameter density, stale caches, forbidden mass/reverse modes,
  flat-root accuracy failures and lossy large-offset reconstruction fail explicitly.
- Fixed seeds reproduce both chains; different seeds differ; NumPy's global RNG
  is unchanged. Every retained dataset in the small unit-test chains is checked
  with the hardened selector.

The initial run had 25 passes and one incorrect test expectation: the fixture
with all psi'=0 has zero Gram, so it raises `gram_degenerate` before curvature
classification. Separate regular-Gram/zero-curvature coverage was then added.
One command named a nonexistent test file and collected no tests; its log is
retained. The corrected focused command had no collection failure or timeout.

All 12 initializer oracle checks plus **252 proposed/retained checks** passed
against the independent rational polynomial-root oracle. Maximum production
selected-location discrepancy was 3.24e−16, likelihood gap 3.55e−15, and score
residual at the recomputed selected root 5.00e−16. The 252 checks comprise 144
sampled proposals and 108 retained states, including each chain's largest
retained residual. Sampling this subset does not certify every proposed root.

All 144,012 saved states (including starts) satisfy |C|≤1.00e−12. Largest cached
selected-MLE discrepancy from the release was 5.26e−12. Over all forward and
reverse trajectories, tangent residuals were ≤9.50e−16, position return error
≤1.61e−11 and momentum return error ≤4.65e−11. Minimum retained J was 0.03977,
minimum retained G 0.01882, and minimum projection transversality 0.6636. Two
selector high-precision fallbacks succeeded; there were **zero unresolved
numerical failures, projection failures or phase-reversal failures**.

## 4. Frozen experiment and independent reference provenance

After three 500-step health/cost pilots, freeze ε=0.1, L=10, parameter proposal
SD=0.45, projection tolerance 1e−12, maximum 50 Newton iterations and the full
phase checks above. Run four independent 12,000-step chains per case; discard
2,000 steps per chain, retaining 40,000 draws per case. Starts are exactly the
Gibbs study's weighted-reference maximum-residual q10/q50/q90 datasets plus a
normalized negative outlier of magnitude 12, with diverse initial parameters.
Seeds are 2026102200–2203, 2026102300–2303 and 2026102400–2403. Each chain has a 180-second
ceiling; no failure occurred. Runs were serial and each took about 2.3–2.7 seconds.

The declared screens are R-hat≤1.01 and both bulk/tail ESS≥400 for mu, maximum
residual and indicator(maximum>6), plus each summary within four combined
reference/chain MCSEs. Chain MCSE is the larger of a 500-step batch estimate and
the dispersion of four chain summaries. This is a diagnostic uncertainty
estimate, especially weak with long tail episodes. Zero constant event samples
have unknown event uncertainty, never zero MCSE. The shifted mu>2 event is
excluded from normal-uncertainty screening.

References are **raw weighted selected-MLE particles**, without KDE. The
independent identity is E~Student_t(k)^n, mu=t−T(E), X=E+mu, weight=prior(mu).
Raw file hashes, the unchanged hardened selector hash, and every reference
summary were rechecked. Original whole-particle bootstrap uncertainty is
carried forward unchanged; new reference draws are unnecessary for this task.

- n=3: the Gibbs study's fresh 32,768-dataset bank, importance ESS 32,009.
- centered n=10: the follow-up's two 65,536-dataset banks, reweighted for the
  centered prior, importance ESS 130,897.
- shifted n=10: the same validated banks, importance ESS 14,747; original
  shifted bootstrap and provenance are retained. The precise reference mean is
  0.493137±0.005301 and latent tail6 is 0.269247±0.003889.

The corrected Gibbs comparison carries forward its saved chains and diagnostics
rather than rerunning or editing Gibbs. Those raw chain hashes were checked too.

## 5. Posterior and latent comparisons

Below, R denotes raw weighted reference and H denotes corrected RATTLE. Saved
JSON/CSV includes every summary, reference bootstrap uncertainty, chain MCSE,
chain/split summaries, KS/Wasserstein distances and Gibbs comparison.

| Case | Mean R / H | SD R / H | q05 R / H | q50 R / H | q95 R / H |
|---|---:|---:|---:|---:|---:|
| centered_n3 | -0.00342 / 0.01501 | 0.77107 / 0.76516 | -1.24411 / -1.22651 | -0.00958 / 0.01167 | 1.23777 / 1.24484 |
| centered_n10 | -0.00122 / -0.01010 | 0.42048 / 0.42361 | -0.68545 / -0.69692 | -0.00029 / -0.00870 | 0.68395 / 0.67683 |
| shifted_n10 | 0.49314 / 0.49827 | 0.43861 / 0.43605 | -0.26500 / -0.27390 | 0.52972 / 0.53231 | 1.14689 / 1.16370 |

Latent maximum absolute residual summaries and tail probabilities:

| Case | Max q50 R / H | Max q90 R / H | Max q95 R / H | P(max>6) R / H |
|---|---:|---:|---:|---:|
| centered_n3 | 1.46790 / 1.39636 | 4.81761 / 4.14625 | 6.97662 / 5.73123 | 0.06688 / 0.04463 |
| centered_n10 | 3.60040 / 3.64507 | 9.67328 / 10.46347 | 13.79973 / 15.61063 | 0.23440 / 0.25155 |
| shifted_n10 | 3.95416 / 3.99204 | 10.34874 / 10.52404 | 14.68040 / 14.75507 | 0.26925 / 0.27210 |

Tail6 chain diagnostic MCSEs are 0.00759, 0.03669 and 0.02007, respectively,
versus reference bootstrap SEs 0.00136, 0.00109 and 0.00389. Thus central and
latent summary differences pass the deliberately coarse four-MCSE screens, but
large MCSEs do not establish adequate joint exploration. For n=3, tail6 is
2.9 combined MCSEs low and max q95 is 3.6 MCSEs low: its formal screen pass is
provisional evidence with weak tail precision, not a high-precision validation.

Parameter KS distances are 0.0145, 0.0118 and 0.0061. Latent-max KS distances are
0.0317, 0.0200 and 0.0123; corresponding Wasserstein distances are 0.377, 0.730
and 0.360. These descriptive distances have no independent-draw KS p-value
interpretation for autocorrelated chains.

## 6. Mixing, branches and components

| Case / observable | R-hat | Bulk ESS | Tail ESS | Declared screen |
|---|---:|---:|---:|---|
| centered_n3 / mu | 1.00285 | 1916 | 2258 | pass |
| centered_n3 / latent_max | 1.00453 | 1299 | 719 | pass |
| centered_n3 / latent_tail6 | 1.00556 | 683 | 683 | pass |
| centered_n10 / mu | 1.00078 | 5116 | 5616 | pass |
| centered_n10 / latent_max | 1.03200 | 133 | 25 | fail |
| centered_n10 / latent_tail6 | 1.02407 | 199 | 199 | fail |
| shifted_n10 / mu | 1.00093 | 3659 | 4728 | pass |
| shifted_n10 / latent_max | 1.01143 | 233 | 87 | fail |
| shifted_n10 / latent_tail6 | 1.00499 | 374 | 374 | fail |

**n=3:** all four chains visit all 13 central score-geometry chambers, with
2,887–3,075 chamber changes per chain. The raw weighted reference and retained
chains lie in central graph component C0. No component transitions are
observed, but these data identify no missing relevant selected-MLE component.
The other six two-chamber components belong to the historical *score* geometry;
using the old numerical classification as a theorem of global-MLE validity
would be unjustified. This experiment uses the hardened selector, not a chamber
classification, for support. Finite RATTLE projections may in principle choose
remote roots, so a continuous-path component argument alone does not establish
which components a discrete numerical proposal can connect.

Tail6 entries/exits occur 47–64 times per n=3 chain, and longest tail episodes
are 41–160 retained steps. Tail6 frequencies range 0.0320–0.0515, with split
frequencies ranging 0.0160–0.0742. Passing R-hat/ESS here does not erase the low
pooled tail estimate. The corrected Gibbs tail estimate is 0.0656±0.00240 and
its indicator ESS is 11,691, versus RATTLE's 683.

**centered n=10:** latent-max R-hat=1.0320, bulk ESS=133 and tail ESS=24.7;
tail6 R-hat=1.0241 and ESS=199. Chain tail6 frequencies are
0.3321/0.2904/0.2145/0.1692. Chain 0 spends one uninterrupted 1,510-step episode
above 6; its two half-chain tail probabilities are 0.2138 and 0.4504. Its largest
residual reaches 54.8. This is a failed latent-mixing validation even though
parameter R-hat=1.00078 and bulk ESS=5,116.

**shifted n=10:** latent-max R-hat=1.01143, bulk ESS=233 and tail ESS=86.6;
tail6 ESS=374, below the frozen minimum. Chain tail6 frequencies are
0.2981/0.3080/0.2346/0.2477; longest tail episodes are 925/677/224/210 steps.
The pooled tail probability happens to agree with the precise reference, but
this case still fails latent mixing. The parameter block passes its screen.
No mu>2 events were observed; its uncertainty is unavailable at this budget.

n=10 branch vectors change about 7,300 times per chain, so branch movement
alone does not diagnose maximum-residual mixing. With no projection/reversal
failures and only a small fraction of support rejections, these failures are
not explained by numerical MLE errors or insufficient reference importance
ESS. **Mechanistic interpretation, not a proof:** fixed identity mass and unit
trajectory length yield local displacement of remote observations; at large
|y| both the Student restoring force and the observation's constraint derivative
are weak. Long tail excursions therefore relax slowly. Topology of the n=10
selected-MLE support is not resolved by this experiment.

## 7. Acceptance, runtime and statistical efficiency

| Case | Acceptance | Negative-curvature rejection | Global-selection rejection | Cheap MH rejection |
|---|---:|---:|---:|---:|
| centered_n3 | 84.2708% | 0.0000% | 0.0000% | 15.7292% |
| centered_n10 | 85.7792% | 0.0500% | 0.0042% | 14.1667% |
| shifted_n10 | 83.4333% | 0.2312% | 0.0521% | 16.2833% |

Across 144,000 proposals, 135 negative-curvature endpoints and 27
positive-curvature nonselected global endpoints were rejected. About 14–17% of
selector calls were saved by cheap MH/curvature rejection. There were no
projection/reversal failures and two successful selector fallbacks. Acceptance
is not offered as evidence of latent convergence.

Warmed single-core benchmarks use three alternating repeats of 1,000 updates,
with the same ε,L for both RATTLE variants. Values are microseconds per update:

| Case | Legacy RATTLE | Selected RATTLE | Affine | Selected Gibbs |
|---|---:|---:|---:|---:|
| centered_n3 | 39.8 | 184.1 | 250.3 | 267.7 |
| centered_n10 | 47.9 | 192.9 | 312.0 | 1334.9 |
| shifted_n10 | 46.7 | 193.6 | 312.9 | 1377.0 |

The main diagnostic chains cost about 198/213/216 microseconds per iteration;
forward-or-reverse trajectories average 25.9/27.9/28.3 microseconds, measured
from the combined phase timer divided by the number of such trajectories.
These preliminary timers include Python bookkeeping and initialization where
applicable; benchmarks are warmed, not compilation timings. Legacy RATTLE
has a different target and weaker reversal checks; Gibbs has floor(n/2) pair
moves per sweep, whereas affine/RATTLE make one dataset proposal.

RATTLE is approximately four times the cost of legacy RATTLE and, at n=10,
approximately seven times faster per update than corrected Gibbs. This does
**not** make it more efficient for latent tails. Main-run bulk ESS/sec for
(mu, max residual, tail6) is respectively:

- n=3 RATTLE (201,136,72), carried-forward Gibbs (154,676,829).
- centered n=10 RATTLE (500,13,19), Gibbs (71,252,209).
- shifted n=10 RATTLE (353,22,36), Gibbs (49,187,209).

ESS/sec divides retained ESS by total serial chain runtime, including burn-in.
The fresh short benchmark puts RATTLE ahead of affine in per-update throughput.
The prior affine follow-up nevertheless had poor shifted joint exploration
(latent-max R-hat 1.078); no new affine convergence claim is made here.

A benchmark process was initially launched while chains were finishing and
terminated before it wrote an execution record or timings. Its empty startup
log is retained. The reported benchmark was run after chain analysis finished,
without other experiment work. No chain was rerun or extended.

## 8. Reproduction, preservation and limits

The [configuration](evidence/student_selected_mle_rattle_2026-10-10/config.json),
[source/runtime provenance](evidence/student_selected_mle_rattle_2026-10-10/chains_execution.json),
[summary CSV](evidence/student_selected_mle_rattle_2026-10-10/summary.csv),
[full comparisons](evidence/student_selected_mle_rattle_2026-10-10/comparisons.json),
[trace figure](evidence/student_selected_mle_rattle_2026-10-10/traces.png),
[oracle audit](evidence/student_selected_mle_rattle_2026-10-10/oracle_checks.json),
[component evidence](evidence/student_selected_mle_rattle_2026-10-10/n3_components.json),
[benchmark](evidence/student_selected_mle_rattle_2026-10-10/benchmark.json),
[saved-state invariants](evidence/student_selected_mle_rattle_2026-10-10/saved_state_invariants.json),
and [preservation check](evidence/student_selected_mle_rattle_2026-10-10/preservation_check.json)
make the claims reviewable. A manifest records raw chain hashes and their
connections to saved results. Bulky chain arrays remain in ignored
`tmp/student_selected_mle_rattle_20261010/`; source checkout contains compact
results and provenance.

From repository root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/test_student_selected_mle_rattle.py tests/test_student_rattle_diagnostics.py \
  tests/test_student_selected_mle_gibbs.py tests/test_student_location_mle_hardening.py \
  tests/test_student_global_mle_audit.py -q --assert=plain

PYTHONPATH=/tmp/student-mle-oracle-deps-20261003 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/validate_student_selected_mle_rattle.py \
  --config docs/research/evidence/student_selected_mle_rattle_2026-10-10/config.json \
  --phase analyze
```

The oracle path supplies SymPy 1.14.0 outside the production environment. An
independent environment with that version can replace the PYTHONPATH setting.
To regenerate chains, copy the config to a **fresh raw/evidence directory** and
run `--phase chains`, then `--phase analyze` and `--phase benchmark`. The script
refuses to overwrite existing chains. It requires the preserved, hash-checked
reference banks and Gibbs comparison chains; this task does not regenerate
unchecked caches. Pilot data are excluded from comparisons. No large campaign,
new family, curvature-gradient force or sampler optimization was attempted.

Remaining limits are finite projection/phase tolerances, the inherited parameter
uniform guard, singular strata outside this regular target, no universal
selector certification, subset oracle checks, weak n=3 tail precision and
failed n=10 latent mixing. A correct invariant measure is not a proof of
irreducibility or useful exploration.

## 9. Decision

**CORRECTNESS:** The exact regular-branch kernel has the selected-MLE invariant
distribution, under the stated reversible/volume-preserving projection-domain
conditions. The separate implementation passes the focused numerical checks and
explicitly reports unresolved selection/curvature/arithmetic. Floating-point
universal certification is not established.

**VALIDATION:** n=3 passes the frozen screens provisionally, with a low and
imprecise tail estimate. Both n=10 cases pass parameter and coarse summary
comparisons but fail joint latent-mixing validation.

**MIXING:** The tested n=3 selected-support chambers are explored. n=10 latent
tails relax too slowly; high acceptance and frequent branch changes do not fix
this. Gibbs remains the better validated joint sampler for these cases.

**PERFORMANCE:** Corrected RATTLE is cheaper per update than corrected Gibbs or
affine, but has much poorer latent ESS/sec than Gibbs in this experiment.

**NEXT STEP:** Further focused RATTLE tail-transport work is needed before
claiming useful joint validation or proceeding to its optimization/cleanup.
Keep this endpoint-correct kernel as the invariant-target baseline. A future
small, separately reviewed nonlocal reversible move or mixed RATTLE/Gibbs kernel
could address tail transport; that is a proposal, not an implemented change.
No Gibbs optimization or repository cleanup was begun.
