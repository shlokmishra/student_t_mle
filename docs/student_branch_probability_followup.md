# Student inverse-branch probabilities and MLE filtering: follow-up note

Status: analytic direction for later work. This is not part of the main
general-audience presentation.

## Question

Do inverse configurations that fail the selected/global-MLE condition already
receive little probability under the score-conditioned Gibbs update? If not,
can they be removed during branch reconstruction without changing the desired
target?

## Score-conditioned branch law

For

\[
z=\psi(y)=\frac{y}{k+y^2},
\]

let \(y_{\mathrm{in}}(z)\) and \(y_{\mathrm{out}}(z)\) be the two finite
inverse values. Conditional on a fixed score coordinate, the production Gibbs
sampler uses

\[
w(y\mid z,\mu)=\frac{f_\mu(y+\mu^\star)}{|\psi'(y)|}.
\]

For a pair with fixed \((z_i,z_j)\),

\[
P_{\mathrm{score}}(b_i,b_j\mid z_i,z_j,\mu)
\propto w(y_{i,b_i})w(y_{j,b_j}).
\]

This is the correct conditional law for the score-conditioned target. It does
not contain a global-MLE indicator.

When \(\mu=\mu^\star\), define

\[
r=2\sqrt{k}|z|,\qquad d=\sqrt{1-r^2}.
\]

Then

\[
P_{\mathrm{score}}(\mathrm{outer}\mid z)
=
\frac{(1-d)^{(k-1)/2}}
{(1-d)^{(k-1)/2}+(1+d)^{(k-1)/2}}.
\]

Consequences:

- For \(k=1\), the probability is exactly \(1/2\) for every interior
  nonzero \(z\).
- For \(k>1\), far outer inverses are suppressed as \(z\to0\).
- At the finite branch merge, the probability approaches \(1/2\) for every
  \(k\).
- When \(\mu\ne\mu^\star\), the likelihood can favor an outer inverse, so
  there is no universal ordering.

## Why branch code is not an MLE classifier

MLE status depends on the complete reconstructed dataset. For \(n=3\), the
selected root is the global MLE exactly when no two outer observations have the
same sign. Thus a code such as `110` may be MLE-valid or non-MLE depending on
the outer signs. For larger \(n\), the likelihood generally must be evaluated;
even a signed branch code need not determine the selected maximum.

## What naive rejection targets

Dropped or rejected non-MLE configurations change the score-conditioned
target because those states have positive score-target mass.

If the intended target is instead exact selected-MLE conditioning, the needed
factor is

\[
R_{\mathrm{MLE}}(x)=
\left|\sum_{\ell=1}^n\psi'(x_\ell-\mu^\star)\right|
\mathbf 1\{\mu^\star\text{ is the selected global MLE}\}.
\]

At fixed score coordinates, the exact branch law is therefore

\[
P_{\mathrm{MLE}}(b_i,b_j\mid z_i,z_j,x_{-(i,j)},\mu)
\propto
w(y_{i,b_i})w(y_{j,b_j})R_{\mathrm{MLE}}(x).
\]

Resampling until an MLE-valid branch appears uses the indicator but omits the
curvature factor. It targets the score distribution truncated to MLE-valid
states, not the exact estimator-conditioned distribution. Correcting only the
branch refresh is also insufficient if the score-coordinate move is left at
its old marginal target.

Two valid constructions remain to test:

1. Enumerate the four pair branches, use weights
   \(w_iw_jR_{\mathrm{MLE}}\), and use their sum as the corrected marginal
   score density.
2. Use a complete score-reversible pair update as a proposal and add an
   outer Metropolis correction with acceptance
   \(\min\{1,R_{\mathrm{MLE}}(x')/R_{\mathrm{MLE}}(x)\}\).

## Focused experiment for later

For saved \(n=3\) and \(n=10\) contexts, enumerate all four inverse pairs and
record their score weights, signed branch labels, global-MLE indicators,
curvature factors, and exact selected-MLE weights. Summarize:

- score-target probability assigned to non-MLE pairs;
- rejection rate of an indicator filter;
- total-variation distance between naive truncation and the exact corrected
  branch law;
- dependence on \(k\), score distance from zero/merge, and
  \(\mu-\mu^\star\).

Useful plots are an outer-probability curve for \(k=1,2,3\), an \(n=3\)
hexagon heatmap of conditional non-MLE mass, and naive-versus-exact branch
probability panels. No broad production rerun is needed before this
enumeration study.

