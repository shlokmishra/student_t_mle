# Student location Gibbs inverse-branch correctness audit

## Target and pair disintegration

For fixed parameter $\mu$, let

$$
F_\mu(y)=\prod_{r=1}^n f_{\mu,r}(y_r),
\qquad
C(y)=\sum_{r=1}^n\psi(y_r),
\qquad
J_C(y)=\|\nabla C(y)\|.
$$

The score-conditioned latent target with respect to Hausdorff measure is

$$
\Pi_H(dy\mid\mu,C=0)
\propto \frac{F_\mu(y)}{J_C(y)}\,dH^{n-1}(y).
$$

Equivalently, by the coarea identity, its integrals can be written as

$$
\int h(y)F_\mu(y)\delta_0(C(y))\,dy.
$$

Fix all coordinates except $i,j$, and put

$$
\delta=-\sum_{r\ne i,j}\psi(y_r),
\qquad
g(y_i,y_j)=\psi(y_i)+\psi(y_j)-\delta.
$$

Disintegrating the ambient delta measure over the pair gives

$$
\Pi_{ij}(dH^1\mid y_{-(i,j)},\mu)
\propto
\frac{f_{\mu,i}(y_i)f_{\mu,j}(y_j)}
{J_{ij}(y_i,y_j)}\,dH^1,
$$

where

$$
J_{ij}=\sqrt{\psi'(y_i)^2+\psi'(y_j)^2}.
$$

On one inverse-branch pair, parameterize the curve by

$$
z_i=z,\qquad z_j=\delta-z.
$$

Then

$$
\frac{dy_i}{dz}=\frac1{\psi'(y_i)},
\qquad
\frac{dy_j}{dz}=-\frac1{\psi'(y_j)},
$$

and therefore

$$
\frac{dH^1}{dz}
=\sqrt{\frac1{\psi'(y_i)^2}+\frac1{\psi'(y_j)^2}}
=\frac{J_{ij}}{|\psi'(y_i)\psi'(y_j)|}.
$$

The $J_{ij}$ factors cancel, but the two inverse-map Jacobians do not.  The
joint density with respect to $dz$ and counting measure on branch pairs is

$$
p(z,a,b)\propto
\frac{f_{\mu,i}(y_{i,a}(z))f_{\mu,j}(y_{j,b}(\delta-z))}
{|\psi'(y_{i,a}(z))|\,|\psi'(y_{j,b}(\delta-z))|}.
$$

Consequently,

$$
q_\mu(z)=\sum_{a\in\{\mathrm{in},\mathrm{out}\}}
\frac{f_\mu(y_a(z))}{|\psi'(y_a(z))|},
$$

$$
p(z\mid\delta,\mu)\propto q_\mu(z)q_\mu(\delta-z),
$$

and

$$
P(a,b\mid z,\delta,\mu)
\propto
\frac{f_\mu(y_{i,a})f_\mu(y_{j,b})}
{|\psi'(y_{i,a})|\,|\psi'(y_{j,b})|}.
$$

The last expression factorizes by coordinate.  Thus sampling the two branch
labels independently is exactly equivalent to sampling the four joint weights.
Using inverse Jacobians in both the marginal computation and the conditional
branch draw is not double-counting: $p(z)P(a,b\mid z)=p(z,a,b)$.

## Repository verdict

Verdict: **A. The production implementation is correct; the standalone Gibbs
markdown was wrong.**

The production `numba_full` path in `models/loc_student.py` implements

$$
\log w(y)=\log f_\mu(y)-\log|\psi'(y)|
$$

in `inverse_branch_log_weight`.  `q_logpdf_no_const` applies a log-sum-exp over
the two branch weights.  `sample_inverse_branch` uses the same weights, and
`update_pairs` applies it independently to both pair coordinates.  `run_gibbs`
rejects every backend name except `numba_full`; there is no active JAX Gibbs
implementation.  The NumPy profiling replica in
`scripts/profile_student_sampler_comparison.py` uses the same formulas.

`docs/paper/paper.tex` already states the correct branch formula.  The incorrect
density-only formulas in `docs/gibbs_student.md` were corrected in this audit.

## Centered Student outer-branch probability

For $\mu=\mu^\star$, put

$$
r=2\sqrt{k}|z|,
\qquad d=\sqrt{1-r^2}.
$$

The density-only and correct outer probabilities are respectively

$$
P_{\rm old}(\mathrm{out}\mid z)
=\frac{(1-d)^{(k+1)/2}}
{(1-d)^{(k+1)/2}+(1+d)^{(k+1)/2}},
$$

$$
P_{\rm correct}(\mathrm{out}\mid z)
=\frac{(1-d)^{(k-1)/2}}
{(1-d)^{(k-1)/2}+(1+d)^{(k-1)/2}}.
$$

In particular, the correct Cauchy probability is exactly $1/2$ at every
interior nonzero $z$.  As $z\to0$, the density-only formula suppresses the
outer branch by two additional powers of $z$.

## RATTLE

The production RATTLE potential is

$$
U(x,\mu)=-\log f_\mu(x)+\frac12\log\{\|\nabla C(x)\|^2\},
$$

so its surface density is $f_\mu(x)/J_C(x)$.  The Gram correction is mandatory
and disabling it is rejected.  This is the corresponding correct coarea target
for score conditioning, independently of the sampler's global reducibility.
