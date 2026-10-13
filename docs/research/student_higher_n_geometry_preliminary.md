# Preliminary Student n=4,5 score geometry and selected-MLE support

Date: 2026-10-10. Location only, fixed positive scale and k>0. This investigation gives partial theorems, an exact finite-fold graph, and six independently checked small examples. **It is not a full classification of the n=4,5 manifold or selected-MLE fiber.**

**Main conclusions:** scaling and the inverse-branch framework generalize. Odd-n score manifolds are regular; even n has balanced singular fold states, including a degenerate selected global MLE at n=4. Global uniqueness, the median-root theorem, and whole-central-component MLE validity all break. At n=5, selected-MLE conditioning removes a proper region of an individual score chamber, even where zero remains a strict local likelihood maximum.

## 1. General score polytope and finite folds — proof

Normalize r_i=(x_i-t)/(sigma sqrt(k)) and a_i=r_i/(1+r_i²). The score-zero condition is sum a_i=0, with |a_i|<=1/2. Hence

\[
P_n=[-1/2,1/2]^n\cap\{\textstyle\sum_i a_i=0\}
\]

has dimension n-1. Both inverse branches remain

\[
r_{\rm in}(a)=\frac{2a}{1+\sqrt{1-4a^2}},\qquad
r_{\rm out}(a)=\frac{1+\sqrt{1-4a^2}}{2a}.
\]

They merge at finite r=±1 when a=±1/2. An outer inverse diverges at a=0; this is not a finite contact. Inner coordinates can cross zero. These facts hold for every n and k; scaling changes statistical mass, not this normalized geometry or likelihood-root classification.

A chamber specifies each coordinate as inner or as a signed outer branch. Its relative-open score domain is a convex polytope intersection. All 3^n sign/inner assignments are feasible except the two all-outer/all-same-sign assignments, giving **3^n-2 chambers**. Here “inner” does not mean the observation is zero.

For a single coordinate i to fold at a_i=s/2, hold the other branch labels fixed. Let p be the number of other outer coordinates forced to sign s, q the number forced to -s, and r the number of other inner coordinates. The possible sum of s*a_j over j!=i is the open interval

\[
\left(-(q+r)/2,(p+r)/2\right).
\]

It contains -1/2 precisely when q+r>1, equivalently **p<=n-3**. This proves the relative-open, single-fold adjacency rule. At n=3 no same-sign other outer coordinate is allowed; at n=4 one is allowed and at n=5 two are allowed. The n=3 rule must not be reused literally.

An exact n=4 example is a=(1/2,1/10,-3/10,-3/10), with r=(1,5+2 sqrt(6),-1/3,-1/3). The first coordinate folds while another outer coordinate has the **same** sign. This connection is impossible at n=3. The numerical witness and its residual are saved in `higher_n_fold_witness.json`.

Polytope vertices have n-1 active box constraints. For even n the remaining coordinate must also be ±1/2, with equal numbers of signs. For odd n it must be zero, and the other n-1 signs are balanced. Thus n=4 has six vertices (a regular octahedron in its three-dimensional hyperplane), while n=5 has 30 vertices. Some inverse sheets over an odd-n vertex have an outer zero coordinate and no finite preimage.

## 2. Finite-fold graph — exact combinatorics, limited interpretation

[student_higher_n_geometry.py](../../reporting/diagnostics/student_higher_n_geometry.py) implements the proved adjacency rule. Independent linear programs maximize a strict interior margin on every candidate face for n=4,5, rather than repeating the count formula. The n=3 graph is independently matched to the existing component graph.

| n | Chambers | Single-fold edges | Connected-component sizes of this graph |
|---|---:|---:|---|
| 3 | 25 | 24 | 13 and six of size 2 |
| 4 | 79 | 160 | 63 and eight of size 2 |
| 5 | 241 | 720 | 221 and ten of size 2 |

[Machine-readable graphs](evidence/student_independence_geometry_2026-10-10/higher_n_graphs.json) include every node and graph component. These counts classify this finite-face adjacency graph; they are not a homotopy model, a selected-MLE classification, or a proof that the larger graph region is a plane. Higher-codimension contacts and singular neighborhoods need their own analysis. No likelihood grid is used to declare global support.

**Partial manifold theorem: 2n separated tail components for n>=3.** Fix distinguished index i and sign s, and require s*r_j>1 for every j!=i. Score balance forces r_i=-s*c, c>0. Write

\[
\psi(c)=\frac{c}{1+c^2},\qquad
a_j=s p_j\psi(c),\quad p_j>0,\quad\sum_{j\ne i}p_j=1.
\]

The other observations are their outer inverses. Since n-1>=2, every p_j<1; their scores stay strictly below the fold bound and their observations stay strictly outside 1. The parameters are (0,infinity) times the open (n-2)-simplex; the inverse uses c=-s*r_i and p_j=s*a_j/psi(c). This is a diffeomorphism to R^(n-1), including the distinguished coordinate's finite fold at c=1.

No majority outer coordinate can reach r_j=s on the score-zero set: its score would be s/2, the other n-2 majority scores strictly have sign s, and the distinguished score cannot cancel more than s/2. Nor can the distinguished observation pass through zero. Thus this connected region is both open and closed in the finite score set, and is a genuine connected component. There are 2n such disjoint components. Limits at an outer zero score or at c=0/infinity require an observation to diverge. This partial result does **not** classify the complement or establish selected-MLE validity throughout any higher-n tail component.

## 3. Regularity — proof and an exact selected singularity

In normalized observations, the score gradient consists of

\[
\psi'(r_i)=\frac{1-r_i^2}{(1+r_i^2)^2}.
\]

All entries vanish iff every r_i=±1. Score zero then requires equal numbers of signs. Therefore **every odd-n finite score-zero manifold is regular**, including n=5. At even n there are binomial(n,n/2) critical configurations at fixed release and scale; n=4 has six. Regularity of the score manifold is different from positivity of J=sum psi': G can be positive while J is negative or zero.

At r=(-1,-1,1,1), C=J=G=0 and

\[
D(s)=\prod_i[1+(r_i-s)^2]=(s^4+4)^2.
\]

Zero is the **unique global minimizer** of D, with quartic rather than quadratic leading likelihood loss. The score derivative has a root of multiplicity three there. Thus even a selected global maximum can have zero curvature and lie at a singular score constraint. The regular density F_mu J/sqrt(G) cannot simply be evaluated at this point as a 0/0 formula.

The leading local score expansion is (sum negative-side perturbations squared - sum positive-side perturbations squared)/4, an indefinite quadratic cone. This is evidence of a qualitatively different singular neighborhood; we do not infer a global topology or an exceptional-stratum conditional measure from that expansion.

## 4. Competing maxima, inferior local maxima, and median failures

The likelihood is proportional to D(s)^(-(k+1)/2), so exact comparisons of D prove the following normalized examples. [Exact expressions and independent rational-oracle checks](evidence/student_independence_geometry_2026-10-10/higher_n_examples.json) are saved separately. Algebraic observations are rounded only for the oracle, which treats their supplied binary values as exact rationals.

**n=4, exact global ties.** For r=(-2,-2,2,2),

\[
D(s)=(s^4-6s^2+25)^2.
\]

Both ±sqrt(3) are global likelihood maxima, zero is an inferior likelihood minimum, and the statistic selects **-sqrt(3)**. Selecting the positive root and shifting it to the release would still leave a lower-location equal global maximum; “is a global maximum” alone is not the selected-statistic condition. Translation equivariance nevertheless holds for the exact lower-tie selector. This already disproves higher-n global uniqueness.

**n=4, strict inferior local maximum.** Put c=(25-2 sqrt(46))/21 and r=(-c,7,7,7). Exact score balance holds because psi(c)=21/50. We have 0<c<2/3 and

\[
J>45/169-144/2500>0.
\]

But D(0)=(1+c²)50³>125000, whereas D(7)=1+(7+c)²<65. Zero is a strict local likelihood maximum but decisively non-global. The oracle selects approximately 6.95625276745. This lies in one of the separated tail components and illustrates why local curvature is insufficient.

**n=5, valid and invalid states in the same chamber.** Consider r=(-a,-a,0,a,a), a>1. All these states have exact score zero and remain in the same chamber (four outer coordinates, two of each sign; the middle coordinate is inner). Also J=1+4 psi'(a)>=1/2>0, using the exact minimum psi'=-1/8. Define v=s², A=a² and

\[
P(v)=v^2+2(1-A)v+(1+A)^2,\qquad
D(s)=(s^2+1)P(s^2)^2.
\]

Its derivative is 2s P(s²) R(s²), where

\[
R(v)=5v^2+(10-6A)v+(A-1)^2+4.
\]

At a=6/5, all coefficients of R are strictly positive. Hence D strictly increases with |s|, and zero is its unique global minimum. At a=5,

\[
D(s)=(s^2+1)(s^4-48s^2+676)^2,
\quad D'(s)=10s(s^4-48s^2+676)(s^4-28s^2+116).
\]

D(5)=265226<D(0)=456976. The global minima are exactly ±sqrt(14+4 sqrt(5))≈±4.79001794464; ±sqrt(14-4 sqrt(5)) are intervening likelihood minima. Zero remains a strict inferior local likelihood maximum. The lower global root is selected.

The path a∈[6/5,5] stays in that same score chamber with positive J. Selected-MLE conditioning therefore removes **a proper portion of a chamber and of its containing component**, not just whole n=3-style tail components. The strict inequalities and nondegenerate selected root at a=6/5 persist on a neighborhood within the score manifold, as does the strict non-global gap at a=5. These are regions, not only isolated symmetric examples. This investigation does not solve the transition value or classify the rest of that chamber.

**Existence of a three-way global tie, proof without computing its parameter.** On the compact a interval [6/5,5], J>=1/2 makes zero a uniformly strict local minimum of D: smooth polynomial coefficients give a common small delta>0 with D_a(s)>D_a(0) for 0<|s|<delta. Every global minimizer lies in [-a,a], since the score outside the data range has a fixed sign. The continuous function min_(delta<=|s|<=5)[D_a(s)-D_a(0)] is positive at a=6/5 and negative at a=5. At a zero of that function, zero and a nonzero symmetric pair are all global minima. The selected statistic chooses a negative minimizer, not zero. Thus there is at least one parameter with **at least three** global maxima; we have not isolated it or ruled out additional transitions. The existing exact selector fallback handles an isolated winner or a proved symmetric pair and can report unresolved more general ties. That is an explicit computational limitation to examine in the follow-up, not an observed failure in the six binary-input examples.

**n=5, no median-bracket root.** For r=(-2,-2,0,100,100), the median is zero. On the entire interval s∈[-1,1], the two negative-observation scores are each <=-3/10, the middle score is <=1/2, and the positive scores are each <=1/99. Thus

\[
C(s)\le-6/10+1/2+2/99<0.
\]

There is **no** stationary root in median±1. The selected root is approximately -1.76299696693. The n=3 bracketing theorem fails, not merely its uniqueness conclusion. At s=0 this example has positive J but is not stationary; it is not used as an exact score-zero factorization.

## 5. Implications and unresolved questions

For arbitrary n, on **exact** score zero the unconditional identity D(s)-D(0)=s² Q_(2n-2)(s) holds, after retaining the linear term until stationarity is established. The degree is six at n=4 and eight at n=5, not four. The n=3 symmetric-function/discriminant proof does not transfer. For example the n=5 symmetric invalid state has e2=-50, so negativity of e2 alone no longer establishes global positivity.

All-outer chambers have J<0 and are excluded from regular selected-MLE conditioning. More broadly, selected support needs global likelihood comparison **and exact tie selection**, besides local curvature. No complete support decision rule for the graph's large region or for the inner half of all higher-n tail components has been proved here. The n=5 example proves that an unweighted graph-component filter cannot be correct.

The selected statistic can switch discontinuously at competing-mode ties, so the n=3 global smooth-T/plane argument fails. The statistic's exact fiber and its topological or measure-theoretic closure must also be distinguished near tie boundaries. We have not assigned conditional mass to singular configurations or proved a general measure-zero statement for every degenerate stratum.

The [independence-MH representation](student_independence_mh_validation.md) survives these geometric obstacles: it requires a measurable translation-equivariant **selected** statistic, not smoothness, unique roots, or a regular score manifold. Numerical selection and rounded tie reconstruction can still fail explicitly. Continuous constrained dynamics and finite RATTLE projections must retain their existing distinct connectivity interpretations; no higher-n RATTLE experiment was run.

**Verified computation:** all six primary examples match the independent oracle's global winner and complete stationary-root list within 1e-10; maximum rounded winner discrepancy is zero and maximum diagnostic log-likelihood gap is 8.89e-16. Final focused tests pass **64 checks**, including eight geometry tests and the existing selector-hardening regressions. Initial geometry tests caught float division in the research summary (7 passes, 1 failure); inputs were converted to exact SymPy numbers and the corrected outputs were independently regenerated. Initial outputs/logs remain archived.

**Recommended one follow-up:** resolve the global-selection boundary in the single family (-a,-a,0,a,a), including the tie convention and changes in its selected-MLE fiber. It gives a tractable exact-algebra problem inside one chamber and directly tests the higher-n assumption that geometry filters can replace global selection. Do not initiate a full n=4,5 classification or a larger-n grid from these findings.

Reproduction: `PYTHONPATH=/tmp/student-mle-oracle-deps-20261003 .venv/bin/python scripts/study_student_higher_n_geometry.py`. The saved evidence contains exact polynomials, graph definitions, binary datasets, root multiplicities, selected roots, likelihood/score diagnostics, source hashes, and execution timing. Existing samplers and evidence were preserved.
