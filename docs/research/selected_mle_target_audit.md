# Released location MLE: target audit

Date: 2026-10-02. Scope: known-scale location models; current working tree,
including pre-existing uncommitted work. **Audit only: no sampler, selector,
registry, or historical-result changes. Stage 2 has not started.**

**Finding:** Student Gibbs and RATTLE implement score conditioning. Exact
conditioning on the selected MLE requires both global selection and an additional
curvature factor. The existing affine sampler has a valid selected-statistic
target argument; direct selected-MLE weighted references already exist, including
latent-data references. Numerical validation of the selector and sampler remains
distinct from these mathematical results.

Labels below distinguish **proved** statements under stated assumptions,
**inspected** implementation facts, and **unresolved** validation questions.

## 1. Definitions and common reference measure — proved

Write the released value as $t=\mu_\star$, the unknown parameter as $\mu$,
the prior as $\pi_0(\mu)$, and the data density as

\[
F_\mu(x)=\prod_{i=1}^n f(x_i-\mu).
\]

Sections 1–2 concern smooth likelihoods (Student, logistic and Gaussian).
Laplace requires the order-statistic treatment in section 7.
Define $C_t(x)=S(t;x)$, $M_C=\{x:C_t(x)=0\}$, and a **specified measurable
selector** $T(x)\in\arg\max_m F_m(x)$, including its tie convention. Its
level set is $M_T=\{x:T(x)=t\}\subseteq M_C$.

Conditioning on a continuous statistic means a regular conditional distribution,
not a positive-probability event. On regular strata, assume the following coarea
integrals have finite positive normalizers and define the density version at the
released value. Equivalent shrinking-slab limits require continuity/limit
conditions; exceptional singular values require separate treatment.

With $H^{n-1}$ denoting Euclidean surface measure and

\[
J_C(x)=\|\nabla_x C_t(x)\|,\qquad J_T(x)=\|\nabla_xT(x)\|,
\]

the two **joint** targets are

\[
\Pi_C(d\mu,dx)\propto
\pi_0(\mu)\frac{F_\mu(x)}{J_C(x)}\,d\mu\,dH^{n-1}(x),\quad x\in M_C,
\tag{1}
\]

\[
\Pi_T(d\mu,dx)\propto
\pi_0(\mu)\frac{F_\mu(x)}{J_T(x)}\,d\mu\,dH^{n-1}(x),\quad x\in M_T.
\tag{2}
\]

Thus score conditioning means the density-disintegration law for
**the fixed statistic $S(t;X)$ at zero**. Its parameter likelihood is

\[
g_{C_t,\mu}(0)=\int_{M_C}F_\mu(x)/J_C(x)\,dH^{n-1}(x),
\]

whereas MLE conditioning uses $g_{T,\mu}(t)$, the analogous integral in (2).
The posterior is proportional to prior times the respective statistic density.
Do not normalize the latent conditional separately for each $\mu$ and then
mix it using the original prior: that omits this posterior reweighting.

The general conditional-density formula is the coarea construction in
[Diaconis, Holmes and Shahshahani, Proposition 2](https://arxiv.org/pdf/1206.6913#page=9).
The model-specific deductions below are derived here and agree with the existing
[paper](../paper/paper.tex), lines 243–305 and 458–485.

## 2. Relationship and equality conditions — proved

On a differentiable selected-root branch with $\partial_mS(t;x)\ne0$,
differentiate $S(T(x);x)=0$:

\[
\nabla_xT(x)=-\frac{\nabla_xS(t;x)}{\partial_mS(t;x)},\qquad
\frac1{J_T(x)}=\frac{|\partial_mS(t;x)|}{J_C(x)}.
\]

Consequently, on the common score surface,

\[
\boxed{\quad d\Pi_T\ \propto\ R_t(x)\,d\Pi_C,\qquad
R_t(x)=|\partial_m S(t;x)|\,\mathbf1\{T(x)=t\}.\quad}
\tag{3}
\]

The indicator is understood **on the score surface**, not as conditioning on an
ordinary event of positive ambient probability. For a likelihood score,

\[
|\partial_m S(t;x)|=|\partial_m^2\log F_m(x)|_{m=t}
\]

is the absolute observed curvature. At a nondegenerate local maximum it is
the positive observed information. Rejecting inferior stationary points alone
gives a truncated score law, still missing this factor. Both factors depend on
$x,t$, not on the unknown $\mu$, so the usual full-data parameter block
$\pi(\mu\mid x)\propto\pi_0(\mu)F_\mu(x)$ remains correct for either target.

The normalized **joint** targets coincide iff $R_t$ is a positive constant
$\Pi_C$-almost everywhere. Root uniqueness only settles the indicator. For
the parameter marginals alone, the weaker necessary and sufficient condition is
that $E_{\Pi_C}[R_t(X)\mid\mu]$ be constant in $\mu$, almost everywhere.
No such general cancellation is established for Student or logistic here.

Even the same zero set does not determine a conditioning measure: replacing
$C_t(x)$ by $a(x)C_t(x)$, with smooth nonzero $a$, divides its score
surface density by $|a(x)|$. In the estimator formula the numerator and
denominator change together and cancel. Constant score rescalings are harmless.

## 3. Student consequences and exceptional cases — proved unless noted

For unit scale, use the reduced score; the omitted multiplier $k+1$ is constant:

\[
C_t(x)=\sum_i\psi(y_i),\quad y_i=x_i-t,\quad
\psi(y)=\frac{y}{k+y^2},\quad
\psi'(y)=\frac{k-y^2}{(k+y^2)^2}.
\]

Equation (3) becomes

\[
R_t(x)=\left|\sum_i\psi'(y_i)\right|\mathbf1\{T(x)=t\},\qquad
J_C=\sqrt{\sum_i\psi'(y_i)^2}.
\tag{4}
\]

**These are different quantities:** absolute sum of derivatives versus square
root of sum of squared derivatives. A Gram correction does not supply (4).

For finite data and $k>0$, the known-scale likelihood attains a maximum in
$[\min_i x_i,\max_i x_i]$. Every maximum is stationary, but roots can be
minima, inferior local maxima, or tied global maxima. Comparing likelihoods
at all relevant candidates and applying the selector convention is essential.

A strict-local-maximum test is insufficient. For $k=1,t=0$, take

\[
x=(4,4,c),\qquad c=(-17+\sqrt{33})/16.
\]

Then $C_0(x)=0$, $\sum_i\psi'(x_i)>0$, yet $F_4(x)>F_0(x)$.
For example, the reciprocal likelihood product at zero exceeds 289,
while at four it is $1+(c-4)^2<24$. Zero is a strict local maximum but
not global. A strict curvature test also excludes degenerate maxima.

**Ties cannot always be dismissed as probability zero.** For two observations
$x=(m-d,m+d)$, $d>\sqrt{k}$, the two global maxima are
$m\pm\sqrt{d^2-k}$. This is an open set of two-observation samples. At
$d=\sqrt{k}$, the midpoint is a degenerate maximum. A lower-location selector
resolves the tie equivariantly; the implicit-derivative formula applies only
where that selected branch is regular.

**A real singular target exception:** for $n=2$, the score coarea law in (1)
is not normalizable. On the regular line $y=(a,-a)$,

\[
\frac{dH^1}{J_C}=\frac{(k+a^2)^2}{|k-a^2|}\,da.
\]

The Student product density is strictly positive near $a=\pm\sqrt{k}$,
giving logarithmically infinite mass. A separately specified slab-limit law
would require a separate analysis. For $n=3$, $J_C>0$ everywhere on the
finite score-zero surface: all coordinates would otherwise be
$\pm\sqrt{k}$, whose three score contributions cannot sum to zero. This
does not by itself prove all estimator regularity or integrability claims.

**Unresolved proof:** the n=3 component note asserts a global-MLE classification
by signs of outer coordinates, but its Sturm-sign calculation is not supplied
([note](../student_n3_score_manifold_components.md), lines 158–200).
The saved grid and quartic identity support it; they do not complete that proof.
The [discussion](../student_n3_gibbs_rattle_discussion.tex), lines 157–166,
correctly separates score targets and observed trapping from exact-MLE sampling.
Continuous-path topology does not alone establish finite-step RATTLE behavior.

## 4. Pair transformation and current Gibbs — proved and inspected

For $0<|z|<1/(2\sqrt{k})$, the two inverse branches are

\[
y_\pm(z)=\frac{1\pm\sqrt{1-4kz^2}}{2z},\qquad
w_b(z;\mu)=\frac{f(t+y_b(z)-\mu)}{|\psi'(y_b(z))|},\quad
q_\mu(z)=\sum_b w_b(z;\mu).
\]

Fix the other coordinates and let $z_i+z_j=\delta$. With respect to
$dz_i$ and counting measure on branch labels,

\[
p_C(z,b_i,b_j)\propto w_{b_i}(z;\mu)w_{b_j}(\delta-z;\mu).
\tag{5}
\]

Indeed the pair arc-length Jacobian divided by its score coarea Jacobian is
$1/|\psi'(y_i)\psi'(y_j)|$. Thus the marginal is
$q_\mu(z)q_\mu(\delta-z)$, and conditional branch probabilities are
proportional to $w_{b_i}w_{b_j}$. Inverse Jacobians in both expressions are
**not double counting**: marginal times conditional recovers (5).

Density-only branch probabilities combined with this $q$ are inconsistent.
For centered Cauchy data, the correct outer probability is $1/2$ at every
interior nonzero $z$; at $z=0.01$, density-only weighting gives about
$0.00010001$. At a fold the branches merge and this chart is singular; at
$z=0$ only the inner inverse is finite. These are chart boundaries, not two
distinct finite branches that can be counted without qualification.

**Current code already uses the correct score weights.**
[loc_student.py](../../models/loc/student/score_gibbs.py), lines 147–175, 193–250, uses
$\log f-\log|\psi'|$ for both $q$ and branch reconstruction, with the
truncated-normal Hastings normalization in the score-coordinate update.
The production `numba_full` dispatch preserves pair score sums and refreshes
branches even after a rejected score-coordinate proposal. Neither the pair
marginal nor reconstruction includes (4).

The [earlier branch audit](../student_inverse_branch_correctness_audit.md),
lines 107–126, already recorded that production was correct and the old markdown
was wrong. Current [gibbs_student.md](../gibbs_student.md), lines 536–570,
already has corrected branch formulas. **This is not a newly found production
branch-weight bug.**

For an exact-MLE pair update, (5) must be multiplied by $R_t(x)$. Its four
branch weights generally do not factorize; their sum is the **changed marginal
target for $z$**. Correcting only the branch draw is insufficient.

## 5. Current RATTLE — inspected

[loc_student_rattle.py](../../models/loc/student/score_rattle.py), lines 82–123,
implements

\[
G=\sum_i\psi'(y_i)^2,\qquad U=-\log F_\mu(x)+\tfrac12\log G.
\]

The intended surface density is therefore $e^{-U}=F_\mu/J_C$: exactly
the **score coarea target**, under ideal constrained-HMC assumptions. The
Gram correction and reverse check are mandatory (lines 470–473). There is
no global-selection test and no $-\log|\sum_i\psi'(y_i)|$ curvature term.
Remaining in an MLE-valid component gives neither that missing weighting nor
evidence of exploration over the full score target.

For both Gibbs and RATTLE, these are statements about intended ideal kernels.
Gibbs trims score bounds by `EPS_Z=1e-12` and stabilizes log derivatives;
RATTLE uses Gram floors, gradient thresholds, and finite projection/reverse
tolerances. Their quantitative effect is not bounded by this audit.

## 6. Existing affine sampler and direct references — proved and inspected

Translation equivariance means $T(x+a\mathbf1)=T(x)+a$, including ties.
It provides two constructions already present in this repository.

**Local affine-normalization MH.** Let $P=I-\mathbf1\mathbf1^\top/n$,
$v=Px$, and $a=\bar x$. Then $T(x)=a+T(v)$, so on the selected level set

\[
x(v)=v+[t-T(v)]\mathbf1.
\]

Fubini in $(a,v)$, followed by the scalar shift $t=a+T(v)$, gives the
target density with respect to Lebesgue measure on $\mathbf1^\perp$:

\[
\pi(\mu,v\mid T=t)\propto\pi_0(\mu)F_\mu(x(v)).
\]

Proposing $u=x+\varepsilon$, $\varepsilon\sim N(0,\tau_q^2I)$, and
$x'=t\mathbf1+u-T(u)\mathbf1$ gives $v'=v+P\varepsilon$, a symmetric
Gaussian random walk. **The correct latent acceptance ratio is just
$F_\mu(x')/F_\mu(x)$.** This proof needs measurable equivariance, not
smoothness or uniqueness. On regular surface charts the same result follows
because the induced proposal is $A(x',x)/J_T(x')$, with symmetric orbit
factor $A\propto\exp[-\|P(x'-x)\|^2/(2\tau_q^2)]$; the Jacobians cancel.

This is precisely the current [loc_affine.py](../../models/loc/affine.py),
lines 112–129 and 155–194, and [affine_plan.md](../affine_plan.md), sections
4–6. It is a **local** proposal; older independence-proposal results are not
validation of the current version. The target argument is sound, conditional
on the implemented $T$ being the intended equivariant selector. Agreement
with a second computation using that same selector is not independent
validation of global selection.

**Direct joint reference.** Draw $E\sim F_0$, compute $A=T(E)$, and form

\[
\mu=t-A,\qquad X=t\mathbf1+E-A\mathbf1,\qquad w=\pi_0(t-A).
\tag{6}
\]

These are importance particles for the joint selected-statistic posterior:
in the generative model $X=\mu\mathbf1+E$, the release is
$T(X)=\mu+T(E)$; substituting $\mu=t-T(E)$ has unit scalar Jacobian.
No extra curvature weight is missing from (6). Normalizing the weighted
sample gives a Monte Carlo approximation, not finite-sample exact posterior
draws. With a bounded prior, rejection using $w/\sup\pi_0$ gives exact
independent draws under an exact selector. With a flat prior, (6) needs no
weighting and its posterior is proper in this location construction.

Existing paths:

| Path | What it actually computes |
| --- | --- |
| [kde_ref/moments.py](../../kde_ref/moments.py), lines 31–63 | Parameter particles $t-T(E)$ weighted by the Gaussian prior. |
| [prepare_feedback_evidence.py](../../reporting/presentation/prepare_feedback_evidence.py), lines 66–85 | **Joint** particles `ref_mu=-mle`, `ref_x=base-mle`, and prior weights for $t=0$; the construction is already implemented. |
| [loc_student.py](../../models/loc/student/score_gibbs.py), lines 41–58; [loc_affine.py](../../models/loc/affine.py), lines 132–152 | New MLE-error simulations call the shared global selector. |
| [kde_ref/moments.py](../../kde_ref/moments.py), lines 78–84 | KDE substitutes a smoothed error density \$g(t-\mu)\$ in the release likelihood; smoothing is optional. |
| [reference_adapter.py](../../kde_ref/reference_adapter.py), lines 59–65, 264–307 | Filters convention-tagged rows and versions Student caches to separate legacy score-root references. |

The saved joint-reference artifacts at `tmp/presentation_feedback_2/diagnostic.npz`
and `metadata.json` are present; they were not regenerated. The reference is
independent of the Markov chains, but shares their estimator implementation.
Historical arbitrary-score-root caches do not become selected-MLE references
because today's selector changed.

The selector [student_location_mle.py](../../models/loc/student/mle.py),
lines 318–336, compares candidate likelihoods and chooses the lower location
within a numerical tie tolerance. Interval budgets, discarded tolerance-small
ambiguous intervals (lines 234–281), extreme arithmetic and near ties still
need independent verification. The name `certified` is not itself a proof of
floating-point global correctness on every input.

## 7. Other location families — proved and inspected

| Family | Score versus selected-MLE conditioning |
| --- | --- |
| Gaussian, known variance | $T=\bar x$, unique root, constant curvature $n/\sigma^2$. The two targets coincide. |
| Logistic, unit scale | Unique root of $\sum\tanh(y_i/2)=0$, but curvature $\frac12\sum\operatorname{sech}^2(y_i/2)$ varies. Same support does **not** mean same conditional measure. |
| Laplace, odd $n$ | The unique median is an order statistic; on each ordering chart $J_T=1$. Smooth score differentiation is inapplicable at the median. |
| Laplace, even $n$ | The likelihood maximizer is an interval. Conditioning that it contains $t$ is a positive-volume event; conditioning on the midpoint median $T=t$ is a different, codimension-one target ($J_T=1/\sqrt2$ on each ordering chart). |

A concrete logistic counterexample is $n=2,t=\mu=0,x=(a,-a)$. The
selected MLE is the mean, so its conditional density in $a$ is proportional
to $\operatorname{sech}^4(a/2)$; score conditioning instead gives
$\operatorname{sech}^2(a/2)$. Unique-root reasoning cannot remove curvature.
[logistic_gibbs_numba.py](../../models/loc/logistic/_numba.py), lines 37–89,
implements the score pushforward, and default
[loc_logistic_rattle.py](../../models/loc/logistic/score_rattle.py), lines 237–256
and 413, implements the score Gram target. Neither has the extra curvature.

[Laplace Gibbs](../../models/loc/laplace/_numba.py), lines 71–80, pins a
fixed label at the median and fixed left/right label groups for odd $n$.
Its ideal target gives the correct parameter and permutation-invariant latent
summaries, but only one label chart of the exchangeable conditional dataset.
For even $n$, it implements the median-interval event and permutes labels;
`get_mle` uses the midpoint median. The distinction is already documented, and
[audit_reference_all_models.py](../../reporting/diagnostics/audit_reference_all_models.py),
lines 178–199 and 407–446, already supplies a separate interval reference.
Do not label all current Laplace reference comparisons mismatched.

## 8. Confirmed inconsistencies and boundaries

| Finding | Status / precise location |
| --- | --- |
| Student and logistic Gibbs described as `x | mu, MLE=mu_star` | Incorrect target descriptions: [Student](../../models/loc/student/score_gibbs.py), line 372; [logistic](../../models/loc/logistic/score_gibbs.py), line 110. |
| Pair score preservation said to preserve the “global MLE constraint” | Incorrect: [gibbs_student.md](../gibbs_student.md), lines 179–204. Its corrected branch formulas do not repair its inferential framing. |
| Student/logistic Gibbs and RATTLE marked as matching MLE references | Incorrect metadata: [model_registry.py](../../models/model_registry.py), lines 93–127, defaults `target_matches_reference=True` and never overrides these rows. It also uses a method-independent score target description for their reference rows. |
| Density-only inverse-branch law | Mathematically wrong for (5), **already corrected in current branch docs and absent from current production code**. |
| Newer documents distinguish the targets | Correct: [paper](../paper/paper.tex), [branch follow-up](../student_branch_probability_followup.md), [RATTLE target](../rattle.md), [affine derivation](../affine_plan.md), and [location-scale note](../loc_scale.tex), lines 117–159. Their presence does not mean every sampler implements all those factors. |
| RATTLE notation | The [note](../rattle.md) defines effective $U=V+\frac12\log G$ at lines 36–42 but early step equations still use $\nabla V$ (e.g. line 83); later pseudocode uses $U$. The current Student code includes the Gram gradient. |
| n=3 classification and RATTLE exploration | Incomplete proof / empirical evidence, respectively; neither establishes exact selected-MLE sampling. |

No inconsistency listed here was silently fixed during the audit.

## 9. Small verification and next decision

The [deterministic diagnostic](selected_mle_target_audit_checks.py) checks the
strict-local/non-global counterexample, an analytic two-point tie, Cauchy branch
weights, logistic curvature variation and an affine normalization. **It passed**,
including five existing Gram/affine test functions invoked directly, in 16.33 s.
The non-global strict maximum had reduced curvature 0.122245 and a competing
log-likelihood advantage of 2.927651; the Cauchy outer weight was 0.5.

Two pytest attempts hit their 120 s and 60 s ceilings; the narrower attempt
stopped during collection, even with plugin autoload disabled. No pytest pass
is claimed, and the collection delay was not diagnosed in this target audit.
The subsequent direct diagnostic had a 45 s ceiling. Exact commands and outcomes are in
[the verification record](selected_mle_target_audit_verification.json).
Each check was limited to one CPU; no empirical branch-frequency batch, dense
selector grid, broad experiment or Slurm job. All 426 pre-existing non-ignored
files checked against the pre-audit hashes remained unchanged.

```bash
.venv/bin/python docs/research/selected_mle_target_audit_checks.py
```

These checks establish neither posterior convergence nor universal selector
correctness. The [September checkpoint](../checkpoints/2026-09-20/baseline.md)
remains historical baseline evidence, not a new exact-MLE validation result.

**Proposed correction strategy, pending review:** freeze (1)–(4) and the tie
convention; correct target descriptions/metadata in a separate change; then
validate the **existing affine implementation** against Gaussian analytic and
independently checked selected-MLE joint references. A subsequent Student
single-pair cross-check can use all four $w_iw_jR_t$ weights and their sum,
or an outer MH factor $R_t(x')/R_t(x)$ applied to a complete score-reversible
single-pair proposal at fixed $\mu$. Do not apply that ratio blindly to an
arbitrary composite Gibbs sweep. No RATTLE bridging rewrite is needed now.

**Genuinely unresolved:** numerical global-selector certification and its
declared domain; effects of ties/degeneracies and numerical cutoffs; completion
of the n=3 classification proof; and quantitative posterior/exploration
validation of the existing affine chain against references that do not merely
reuse the same selector unchecked. Location-scale global selection is outside
this audit. **Await user review before Stage 2.**
