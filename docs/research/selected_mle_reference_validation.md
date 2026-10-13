# Selected-MLE location reference validation

Audit date: 2026-10-03. Scope: fixed scale/shape, location only. This report follows the [target audit](selected_mle_target_audit.md). Production samplers, selectors, references, caches, and existing research files were left unchanged.

**Conclusion.** Five of six cases pass the frozen numerical screens; the shifted Student case remains unvalidated because of low reference importance ESS and unresolved rare-tail precision. The direct weighted construction and the existing local affine sampler have the same selected-MLE joint target, provided their estimator really is the specified translation-equivariant global selector. The raw weighted formula is correct. Independent small-sample reference banks provide a usable benchmark with Monte Carlo error; the existing selector is accurate on those banks but has documented boundary failures. Neither an unchecked cache nor an agreement between two routines sharing the same selector is sufficient validation.

Claims below are marked **derivation**, **inspection**, or **numerical evidence**. These are distinct levels of support.

## 1. Target and direct weighted construction

**Derivation.** Write $X=\mu\mathbf1+E$, where $E$ has density $f_0(e)=\prod_i f(e_i)$, independently of the prior $\pi(\mu)$. Fix a measurable selected global MLE $T$ with

$$T(e+a\mathbf1)=T(e)+a.$$

Use the lower location to break exact global ties. For the release $t=\mu_\star$, let $Z=T(E)$. Then $T(X)=\mu+Z$, and, if $g$ is the sampling density of $Z$,

$$p(\mu\mid T(X)=t)=\frac{\pi(\mu)g(t-\mu)}{C(t)},\qquad
C(t)=\mathbb E_{E\sim f_0}[\pi(t-T(E))].$$

More usefully, for any integrable joint diagnostic $h$,

$$
\mathbb E[h(\mu,X)\mid T(X)=t]
=
\frac{\mathbb E_{E\sim f_0}\!\left[
\pi(t-T(E))\,h\!\left(t-T(E),E+(t-T(E))\mathbf1\right)\right]}
{\mathbb E_{E\sim f_0}[\pi(t-T(E))]}.
$$

For fixed $e$, changing $\mu$ to $t=\mu+T(e)$ has derivative one. This identity can also be obtained directly by Fubini and one-dimensional translation, without differentiating $T$. No additional curvature weight belongs in this construction. The surface Jacobian from the target audit is already accounted for by this change of variables; appending it would change the target.

For independent base datasets $e_b$, the particles and weights are

$$
\mu_b=t-T(e_b),\quad x_b=e_b+\mu_b\mathbf1,\quad
w_b=\pi(\mu_b),\quad \bar w_b=w_b/\sum_c w_c.
$$

Their weighted empirical distribution is a consistent Monte Carlo reference, not an exact finite-sample posterior distribution. Its importance ESS is $(\sum w_b)^2/\sum w_b^2$. The predictive density estimate is $B^{-1}\sum w_b$, including any stabilizing exponential factor and the prior's normalization.

Translation equivariance gives $T(x_b)=t$. It does not follow merely from a zero score. Failed or nonfinite MLE calculations must abort or be resolved; dropping those datasets changes the reference distribution.

For a bounded prior, fresh independent base draws accepted with probability $\pi(t-T(e))/\sup\pi$ give independent posterior joint draws. With a normal prior this acceptance probability is $\exp[-(\mu_b-m_0)^2/(2s_0^2)]$. Thus direct normalization with prior rejection and weighted Monte Carlo are two realizations of this same identity.

## 2. Existing affine sampler

**Inspection and derivation.** [models/loc_affine.py](../../models/loc/affine.py) implements a *local MH-within-Gibbs chain*, not the independent prior-rejection procedure above. Its latent proposal is

$$U=x+\varepsilon,\quad \varepsilon\sim N(0,\tau_q^2I),\qquad
x'=t\mathbf1+U-T(U)\mathbf1.$$

It reconstructs the latent dataset this way; it updates $\mu$ separately by a symmetric random-walk MH step using prior times full-data likelihood.

An independent way to check the acceptance ratio is to use the contrast projection $P=I-\mathbf1\mathbf1^\top/n$ and $v=Px$. Every orbit has the unique representative

$$x(v)=v+(t-T(v))\mathbf1.$$

Decompose an unconstrained dataset as $v+a\mathbf1$. Lebesgue measure is $\sqrt n\,dv\,da$ and $T(v+a\mathbf1)=T(v)+a$. Conditioning on the released value therefore leaves joint density

$$\pi(\mu)\,p_\mu(x(v))\,d\mu\,dv$$

up to a constant. Equivalently, on smooth pieces, $dH_{n-1}(x)/\|\nabla T(x)\|=\sqrt n\,dv$.

The proposal becomes $v'=v+P\varepsilon$, a symmetric Gaussian random walk on the contrast subspace. Consequently its latent acceptance probability is exactly

$$\min\{1,p_\mu(x')/p_\mu(x)\}.$$

This is what the code computes. The prior cancels in the fixed-$\mu$ latent step and is included in the $\mu$ step. No extra Jacobian belongs in either acceptance ratio. This proof assumes the stated selector, fixed covariance proposal, and a proper conditional target. It does not prove finite-run mixing or excuse a numerical selector choosing a different statistic.

The chain recomputes the selected MLE after normalization and at retained states. Those checks use the *same* production selector and are not independent certification. Its initializer fixes every latent observation to $t$; the new runs disperse initial $\mu$, but do not claim dispersed latent initial configurations.

## 3. Actual repository paths

| Path / call | Implemented target or role |
|---|---|
| [loc_student.get_mle](../../models/loc/student/score_gibbs.py) → [student_location_mle.global_mle](../../models/loc/student/mle.py) | Selected global location candidate with a floating-point likelihood tie convention; not one wide-bracket score solve. |
| [loc_affine.global_location_mle](../../models/loc/affine.py) | Normal sample mean; monotone logistic score solve; same shared Student selector, with optional known-scale standardization. |
| [loc_affine.run_gibbs](../../models/loc/affine.py) | Local affine-normalized latent MH and prior-weighted $\mu$ MH; selected-MLE joint target under the assumptions above. |
| [raw_weighted_posterior_moments](../../kde_ref/moments.py) | Correct particles $t-z_b$, prior weights, normalization, quantiles and importance ESS. Its correctness depends on the supplied MLE errors. |
| [prepare_feedback_evidence.py](../../reporting/presentation/prepare_feedback_evidence.py) | Already implements the direct joint construction at $t=0$, prior $N(0,10^2)$: $\mu=-T(E)$, $X=E-T(E)$, weight proportional to $\exp[-T(E)^2/200]$. This construction is not new. |
| Student and affine benchmark generators | Fresh standardized datasets, then their respective MLE wrappers. Both Student paths share the selector; agreement between them alone is circular. |
| [reference_adapter.load_or_simulate_mle_errors](../../kde_ref/reference_adapter.py) | Caches standardized Student errors under the version string selected_global_v1. |
| [posterior.get_normalized_posterior_pdf](../../kde_ref/posterior.py) | KDE approximation to $\pi(\mu)g(t-\mu)$; sign and release shift are correct. |
| [run_affine_experiment.py](../../scripts/run_affine_experiment.py) | Existing comparison runner uses a KDE posterior for headline distances. This audit instead compares directly to weighted particles. |

## 4. MLE selector: algorithm, oracle, and limits

**Inspection.** The production selector centers at the sample median and scales by $\sqrt{k}$. Up to 17 deterministic EM starts provide an incumbent. It searches data-gap intervals using score and score-derivative bounds, bisects ambiguous intervals, polishes sign changes, and prunes intervals whose likelihood upper bound is below the incumbent. There are closed-form two-point/equal-replication cases, a default location tolerance of $2\times10^{-13}$, and an interval budget of one million.

Candidates within $64\epsilon\max(1,|\ell_{\rm best}|,n)$ in its standardized $k=1$ objective are treated as tied; the smallest location wins. Thus “selected MLE” here includes a *numerical* tie region, not just exact ties. No user initialization enters this deterministic selector. Reversing the observations gave identical answers in the audit; the EM incumbent is not itself the final selection proof.

**Independent oracle.** [selected_mle_reference_oracle.py](../../scripts/selected_mle_reference_oracle.py) treats each supplied binary float and $k$ as an exact rational. Define

$$D(m)=\prod_i[k+(x_i-m)^2].$$

Maximizing the Student likelihood is equivalent to minimizing $D$. All real stationary points are roots of $D'$; every global minimum is among them because $D$ tends to infinity. The oracle uses exact rational real-root isolation, including multiplicities, and interval bounds on each factor of $D$ to separate the global winner. It refines from 16 to 32 or 64 decimal digits if needed. Exact data symmetry can prove a tied pair; otherwise an unresolved comparison raises. It does not import production selection logic. The root-isolation API is documented in the [official SymPy polynomial reference](https://docs.sympy.org/latest/modules/polys/reference.html#sympy.polys.polytools.intervals).

This certifies the winner for the supplied rational data when a bound succeeds; it is not a formal certification of production floating-point code or of all real-valued input data. The returned location is still rounded to a float.

**Numerical evidence: every independently generated Student reference draw.**

| Bank | Draws | Multiple-root datasets | Maximum selected-location error | Maximum score residual | Maximum likelihood gap |
|---|---:|---:|---:|---:|---:|
| $k=2,n=3$ | 4,096 | 262 | $2.13\,10^{-13}$ | $2.14\,10^{-13}$ | $3.55\,10^{-15}$ |
| $k=2,n=10$ | 4,096 | 54 | $1.94\,10^{-13}$ | $4.92\,10^{-13}$ | $7.11\,10^{-15}$ |
| $k=3,n=10$ | 4,096 | 3 | $2.19\,10^{-13}$ | $4.41\,10^{-13}$ | $3.55\,10^{-15}$ |

No reference draw was discarded; neither oracle nor production selection failed in these 12,288 cases. Score residuals omit the positive $k+1$ multiplier, matching the repository helper. The floating-point likelihood gaps here are at rounding scale.

**Confirmed limitations, separated by type.**

1. **Incomplete stationary-root enumeration.** In 57 adversarial/random datasets, the public stationary_points helper omitted 26 sign-changing roots across 15 datasets. Its internal search updates an incumbent and then prunes inferior roots, even when the initial incumbent is minus infinity. For $k=1$, $x=(-10,-10,10,10,10)$, the oracle finds approximately $(-9.92448,-2.04218,9.96666)$; the helper returns only the last root. The selected global maximum is correct here. Its advertised all-roots contract, and the module's unrestricted certification language, are incorrect.
2. **Nearly degenerate location accuracy.** For $k=2$, $x=\pm\operatorname{float}(\sqrt2)$, the exact binary-input maximizers are approximately $\pm1.65358\,10^{-8}$. Standardization rounds to the transition and production returns zero. The selected-location criterion fails, while the likelihood loss is only $7.01\,10^{-33}$. A tiny score or objective error cannot establish location accuracy near a flat likelihood.
3. **Numerical ties change the selected statistic.** For $k=2$, start with five observations at $-4$ and five at $4$, then add $2^{-35}$ to the first negative observation. The exact unique global maximum is near $+3.74165738677$; production chooses near $-3.74165738677$. The location error is 7.48331, although a high-precision calculation gives a likelihood loss of only $4.76\,10^{-22}$. This is consistent with its numerical-tie convention, but differs from the exact selected global MLE. Exact ties and near ties must not be conflated.
4. **Extreme finite inputs are not safely covered.** At $k=2$, $(0,1,10^{200})$ exhausts the interval budget; $(-10^{200},10^{200})$ returns a nonfinite result. At a common offset $2^{40}$, the returned location can have score residual about $5.03\,10^{-5}$; the oracle rounds to the same float, and rounding the real maximizer to a float leaves a log-likelihood loss of about $4.14\,10^{-9}$. This last example is precision loss, not selection of the wrong mode.

For the 57-case ordinary audit, maximum selected score residual was $7.35\,10^{-13}$; maximum score residual among all roots returned by the helper was $6.73\,10^{-8}$. Reversal error was zero and translation-by-32 equivariance error at most $7.11\,10^{-15}$. These observations do not prove uniform translation equivariance in finite arithmetic. The global wrapper also lacks the positive finite location-tolerance check present in the public stationary helper.

## 5. Frozen numerical validation

The [configuration](evidence/selected_mle_reference_2026-10-03/config.json) was frozen before simulations. There are six cases, four chains each, 12,000 iterations and 2,000 burn-in iterations: 40,000 retained states per case. Reference banks contain 16,384 Gaussian draws or 4,096 Student draws, with *every* Student draw selected by the independent oracle. The shifted cases reuse the same standardized bank with new prior weights, as the identity permits; they are not independent reference replications.

Centered cases use $t=0$ and prior $N(0,2^2)$. Shifted cases use $t=1.25$ and prior $N(-0.75,0.6^2)$. Gaussian controls use $n=10$, known $\sigma=1.3$. Student controls use $(k,n)=(2,3),(2,10),(3,10)$, plus shifted $(2,10)$. This deliberately checks a release and prior that cannot hide a sign error through symmetry.

Reference MCSE uses 200 whole-particle bootstrap resamples. Affine MCSE is the larger of a 500-state batch estimate and the four-chain mean-of-estimates uncertainty. Rank-normalized split/folded R-hat is checked for $\mu$ and maximum absolute latent residual. The reported mean ESS is an approximation from SD and mean MCSE, not bulk or tail ESS. KS and Wasserstein distances compare empirical chain draws with raw weighted particles; no IID KS p-value is asserted.

Predeclared screens: discrepancies at most four combined MCSE, both R-hats at most 1.05, mean ESS at least 400, importance ESS at least 500. These are finite-run diagnostic screens, not a simultaneous statistical test or proof of exploration. Rare events with zero observations and zero estimated MCSE are unresolved, not proved to have zero probability.

| Case / method | Mean | SD | q05 | q50 | q95 | P(mu < t-0.75) | P(mu > t+0.75) |
|---|---:|---:|---:|---:|---:|---:|---:|
| gaussian_center / weighted | 0.00060873 | 0.40159 | -0.65392 | 0.00039139 | 0.65592 | 0.029611 | 0.030073 |
| gaussian_center / affine | -0.0013752 | 0.39858 | -0.65633 | 0.0026055 | 0.65196 | 0.031125 | 0.03005 |
| gaussian_shift_prior / weighted | 0.60081 | 0.35414 | -0.03458 | 0.61406 | 1.1692 | 0.36877 | 2.0343e-05 |
| gaussian_shift_prior / affine | 0.61909 | 0.34017 | 0.05998 | 0.62116 | 1.1761 | 0.3633 | 2.5e-05 |
| student_k2_n3 / weighted | -0.0064754 | 0.76866 | -1.2053 | -0.020197 | 1.2503 | 0.142 | 0.14053 |
| student_k2_n3 / affine | 0.0071304 | 0.78588 | -1.2613 | 0.012513 | 1.2744 | 0.14145 | 0.14697 |
| student_k2_n10 / weighted | 0.0095266 | 0.42038 | -0.65 | 0.0059473 | 0.69663 | 0.03371 | 0.039446 |
| student_k2_n10 / affine | -0.0059391 | 0.41552 | -0.66724 | -0.01101 | 0.67098 | 0.034275 | 0.0342 |
| student_k3_n10 / weighted | -0.0018632 | 0.39248 | -0.67812 | 0.002152 | 0.63631 | 0.03445 | 0.028056 |
| student_k3_n10 / affine | -0.00064152 | 0.39466 | -0.64365 | 0.0022396 | 0.6367 | 0.030825 | 0.029675 |
| student_shift_prior / weighted | 0.52791 | 0.42047 | -0.30475 | 0.57458 | 1.1617 | 0.44427 | 2.1346e-05 |
| student_shift_prior / affine | 0.48789 | 0.43159 | -0.27207 | 0.52261 | 1.1325 | 0.4782 | 0 |

| Case | Weight ESS | Mean ESS | R-hat mu / latent max | KS | W1 | Largest absolute MCSE discrepancy | Screen |
|---|---:|---:|---|---:|---:|---:|---|
| gaussian_center | 16370 | 5736 | 1.0007 / 1.0033 | 0.0103 | 0.0049 | 0.61 | Pass |
| gaussian_shift_prior | 2331 | 4849 | 1.0008 / 1.0030 | 0.0266 | 0.0187 | 1.74 | Pass |
| student_k2_n3 | 4000 | 3594 | 1.0007 / 1.0069 | 0.0245 | 0.0267 | 1.90 | Pass |
| student_k2_n10 | 4091 | 5220 | 1.0008 / 1.0266 | 0.0213 | 0.0156 | 1.80 | Pass |
| student_k3_n10 | 4092 | 2352 | 1.0013 / 1.0147 | 0.0103 | 0.0061 | 1.61 | Pass |
| student_shift_prior | 464 | 1008 | 1.0027 / 1.0473 | 0.0522 | 0.0440 | 9.26 | FAIL |

**The shifted Student case does not pass.** Its importance ESS is 463.6 versus the frozen minimum 500. Its upper tail at mu > 2 has weighted probability 2.135e-5 and no affine events; the mechanical discrepancy is 9.26 estimated MCSE because the chain estimate gives zero MCSE. That normal approximation is invalid for this unobserved rare event. Even 40,000 independent posterior draws would have about a 43% chance of zero events at this probability; correlated chains need not do better. This is unresolved rare-event precision, not evidence by itself of a wrong invariant target. The failure remains recorded, and no threshold was relaxed. Other shifted-case discrepancies are at most 2.34 combined MCSE.

The shifted Student mean is 0.4879 +/- 0.0136 affine MCSE versus 0.5279 +/- 0.0296 weighted MCSE. Its latent maximum R-hat, 1.0473, is close to the cutoff; individual-chain probabilities of max|X-t| > 6 range from 0.194 to 0.390. Consequently this run does not establish dependable latent-tail exploration. The five other cases pass the declared screens for the monitored quantities, subject to the rare-event caveat. Full nine-metric estimates and uncertainties are in summary.csv; posterior means alone are not the acceptance criterion.

For the Gaussian control, the exact posterior has variance $V=(s_0^{-2}+n/\sigma^2)^{-1}$ and mean $V(m_0/s_0^2+nt/\sigma^2)$. The latent residual distribution is $N(0,\sigma^2P)$ independently of $\mu$, so the expected standardized squared residual norm is $n-1=9$.

| Gaussian case | Analytic mean / SD | Weighted mean / SD | Affine mean / SD | Largest analytic discrepancy, weighted / affine MCSE |
|---|---|---|---|---|
| gaussian_center | 0.00000 / 0.40268 | 0.00061 / 0.40159 | -0.00138 / 0.39858 | 1.26 / 1.15 |
| gaussian_shift_prior | 0.61106 / 0.33913 | 0.60081 / 0.35414 | 0.61909 / 0.34017 | 1.98 / 1.67 |

Both Gaussian controls agree with the analytic posterior within 1.98 estimated MCSE for all seven parameter summaries. Standardized latent squared-norm means were weighted/affine 8.9991/9.0677 (centered) and 9.0525/8.9233 (shifted), near the analytic value 9; these norm means are supplementary checks, not separate uncertainty-calibrated tests.

For every case and method, 24 evenly spaced retained datasets plus the most extreme dataset were independently reselected: 300 checks total, including Gaussian mean controls. The largest released-MLE residual was 1.82e-13. There were 55 initial exact-isolation timeouts; all were resolved with the same rational polynomial and objective bounds using SymPy intervals(fast=True), taking 0.82 seconds total after imports. The original timeouts and fallback certificates are preserved separately. This is a diagnostic algorithm acceleration, not relaxed selection criteria or replacement of failed datasets. The production chain also checked every proposal and retained state with its own selector; its largest retained residual was 1.26e-13. Only the 300 checked datasets have this additional independent reconstruction check.

## 6. KDE and cached evidence

**Confirmed code defects, not fixed here.** The optional Gaussian-KDE/Gaussian-prior analytic helper computes quantiles of component means, omitting within-component variance. It also discards the original maximum log weight before calculating the normalizer. A two-component diagnostic gives helper 2.5%/50%/97.5% quantiles $(0.0147,0.1046,1.0208)$ instead of $(-1.4377,0.5971,2.5930)$, and normalizer 0.91465 instead of 0.18880. Its mean and variance formulas are correct. These defects do **not** affect the raw weighted reference or affine acceptance rule. See [kde_diagnostic.json](evidence/selected_mle_reference_2026-10-03/kde_diagnostic.json).

Ordinary KDE evaluation uses the correct $t-\mu$ sign and consistent component variance/bandwidth in this check. Smoothing can still shift summaries. Existing workflows do use KDE for quantitative summaries and comparison distances; consequently KDE is not currently confined to display everywhere. No KDE enters this audit's primary posterior summaries or distances.

**Inspection: cache limitations.** The versioned Student error cache excludes older score-root conventions, which is useful, but its loader does not verify source/tolerance fingerprints, stored metadata, array shape, or finite values. Disabling cache writes still permits reads of an existing cache. [audit_kde_reference.py](../../reporting/diagnostics/audit_kde_reference.py) resume keys omit released location and prior, while cached posterior summaries depend on both. Some downstream aggregation also omits these distinctions. Standardized *error draws* legitimately need neither prior nor release in their key; posterior *summaries* do.

The all-model reporting adapter can carry a score-conditioned model label onto selected-MLE reference rows, and some presentation artifacts are reused on file existence without verifying recorded source hashes. These are provenance/labeling hazards, not flaws in the location identity. No old cache was used for the new reference banks.

## 7. Recommendation and remaining scope

Adopt **raw weighted joint particles from fresh standardized datasets** as the canonical location-only selected-MLE reference, with explicit estimator/tie convention, complete failure accounting, source/configuration fingerprints, importance ESS and uncertainty for every reported quantity. Retain base datasets as well as MLE errors whenever latent validation is needed. The independently selected banks here are the initial small-case benchmark.

Use the existing affine chain as a second algorithmic cross-check, subject to the numerical evidence above. Independent prior rejection is an optional exact-draw realization when acceptance is affordable; resampling a finite weighted bank does not create an independent reference experiment. The same-bank rejection illustration in the saved comparison file is labeled accordingly.

Before treating production get_mle as universal ground truth, make a focused selector change separately: distinguish global candidate search from complete stationary-root enumeration; specify exact versus numerical ties; reject nonfinite outputs; validate tolerances; and add the adversarial regressions preserved here. Close numerical ties need higher precision or an explicit unresolved result if the desired statistic is the exact global maximizer. Protect posterior-cache keys and metadata before reusing old results.

The derivations are settled for a measurable translation-equivariant selector and a proper fixed-scale location posterior. What remains empirical is mixing and reference precision beyond these cases, especially rare latent tails, Cauchy stress cases and larger $n$. The present run does not establish location-scale validity.

**Next decision:** review this report and choose the estimator/tie contract. Then address the narrow selector/reference safeguards before or alongside a separately authorized corrected-Gibbs/RATTLE validation effort. This task does not begin that cleanup.

## 8. Reproduction and preservation

New validation code is isolated in [validate_selected_mle_reference.py](../../scripts/validate_selected_mle_reference.py), the independent [oracle](../../scripts/selected_mle_reference_oracle.py), and [four focused tests](../../tests/test_selected_mle_reference_validation.py). SymPy/mpmath were installed into an isolated temporary dependency directory, not the project environment:

    .venv/bin/python -m pip install --no-compile --target /tmp/student-mle-oracle-deps-20261003 sympy==1.14.0 mpmath==1.3.0
    export PYTHONPATH=/tmp/student-mle-oracle-deps-20261003
    timeout 1200s .venv/bin/python scripts/validate_selected_mle_reference.py --config docs/research/evidence/selected_mle_reference_2026-10-03/config.json --phase selector
    timeout 1200s .venv/bin/python scripts/validate_selected_mle_reference.py --config docs/research/evidence/selected_mle_reference_2026-10-03/config.json --phase reference
    timeout 1200s .venv/bin/python scripts/validate_selected_mle_reference.py --config docs/research/evidence/selected_mle_reference_2026-10-03/config.json --phase chains
    timeout 1200s .venv/bin/python scripts/validate_selected_mle_reference.py --config docs/research/evidence/selected_mle_reference_2026-10-03/config.json --phase analyze
    timeout 1200s .venv/bin/python scripts/validate_selected_mle_reference.py --config docs/research/evidence/selected_mle_reference_2026-10-03/config.json --phase resolve_latent
    timeout 1200s .venv/bin/python scripts/validate_selected_mle_reference.py --config docs/research/evidence/selected_mle_reference_2026-10-03/config.json --phase kde

Use a copy of the configuration with a fresh output directory for regeneration: reference/chain phases refuse to overwrite saved samples. Each process uses one CPU; executed phases had a 1,200-second wrapper ceiling, with a 600-second bank ceiling; the final exact fallback phase had a 180-second ceiling. No scheduler jobs or production grids were submitted.

An initial analysis pass was terminated after 714 seconds when exact algebra on a reconstructed latent state stalled. The analysis harness then gained a ten-second per-state alarm and incremental evidence writes. Simulations, seeds, weights, and criteria were unchanged. A metadata-only conversion of the production tie tolerance into original Student log-likelihood units was also corrected in this new harness; original phase evidence and source hashes are retained.

The four focused test functions passed when invoked directly. A targeted pytest invocation timed out after 120 seconds without a result, including with plugin autoload disabled; a successful full pytest run is **not** claimed. See [verification.json](evidence/selected_mle_reference_2026-10-03/verification.json) for checks and limitations.

Compact numerical evidence lives beside the configuration, including [summary.csv](evidence/selected_mle_reference_2026-10-03/summary.csv), [comparison.json](evidence/selected_mle_reference_2026-10-03/comparison.json), and selector diagnostics. Bulky base datasets, chains, and per-draw oracle records stay under the ignored output directory named in the configuration. The [artifact manifest](evidence/selected_mle_reference_2026-10-03/artifact_manifest.json) records paths, sizes and SHA256 hashes; seeds and commands permit regeneration. All 429 pre-existing tracked/untracked source and research files in the starting snapshot match their initial hashes. Existing working-tree changes predated this task and were preserved.
