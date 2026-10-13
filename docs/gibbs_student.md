# Constrained Gibbs Sampler for Bayesian Inference from a Released Student-t MLE

## Purpose

This document specifies the constrained Gibbs algorithm used to sample from the posterior distribution of a Student-t location parameter when the **full data are latent** and only the **MLE of the location parameter** is observed.

The goal is to make the sampler mathematically explicit enough that an implementation agent can:

- understand the target distribution,
- understand why the latent-data update must stay on a constraint manifold,
- implement the `mu` update correctly,
- implement the pairwise latent `x` update correctly,
- understand the `y -> z` transformation and its two inverse branches,
- preserve the MLE score constraint exactly up to numerical tolerance.

---

# 1. Statistical model

Let

$$
\mu \in \mathbb{R}
$$

be the unknown location parameter with prior density

$$
\pi(\mu).
$$

The unobserved data are

$$
x_1,\dots,x_m \mid \mu \overset{\mathrm{iid}}{\sim} t_k(\mu),
$$

where:

- `k` is the known degrees of freedom,
- the scale is fixed to 1,
- `mu` is the location parameter.

The Student-t density is therefore proportional to

$$
t_k(x\mid\mu)
\propto
\left(1+\frac{(x-\mu)^2}{k}\right)^{-(k+1)/2}.
$$

The full data $x_{1:m}$ are **not observed**.

Instead, we observe only the maximum likelihood estimator

$$
\hat\mu = \mu^*.
$$

For fixed $k$, the location MLE satisfies the score equation

$$
\sum_{i=1}^m
\frac{x_i-\hat\mu}
{k+(x_i-\hat\mu)^2}
=0.
$$

Therefore, conditioning on the released MLE value $\hat\mu=\mu^*$ means that the latent data must satisfy

$$
\boxed{
\sum_{i=1}^m
\frac{x_i-\mu^*}
{k+(x_i-\mu^*)^2}
=0.
}
$$

---

# 2. Inferential target

The desired posterior is

$$
p(\mu\mid \hat\mu=\mu^*)
\propto
p(\hat\mu=\mu^*\mid\mu)\pi(\mu).
$$

Direct evaluation of the likelihood of the released MLE,

$$
p(\hat\mu=\mu^*\mid\mu),
$$

is inconvenient.

Instead, augment the state with latent data

$$
x=(x_1,\dots,x_m)
$$

that are compatible with the observed MLE.

Define the constraint function

$$
C(x)
=
\sum_{i=1}^m
\frac{x_i-\mu^*}
{k+(x_i-\mu^*)^2}.
$$

The latent data are restricted to the level set

$$
\boxed{
\mathcal M_{\mu^*}
=
\{x\in\mathbb R^m:C(x)=0\}.
}
$$

The Markov chain therefore evolves on the augmented state

$$
(\mu,x),
\qquad x\in\mathcal M_{\mu^*}.
$$

The sampler alternates between:

1. updating $\mu$ conditional on the current latent dataset $x$, and
2. updating $x$ conditional on the current $\mu$, while keeping $x$ on $\mathcal M_{\mu^*}$.

---

# 3. Overall Gibbs structure

At iteration $t$, suppose the current state is

$$
(\mu^{(t-1)},x^{(t-1)}).
$$

A Gibbs iteration consists of two blocks.

## 3.1 Block A: update `mu`

Sample

$$
\mu^{(t)}
\sim
p(\mu\mid x^{(t-1)}),
$$

where

$$
\boxed{
p(\mu\mid x)
\propto
\pi(\mu)
\prod_{i=1}^m t_k(x_i\mid\mu).
}
$$

This is an ordinary one-dimensional posterior update because once the latent dataset $x$ is fixed, the MLE constraint no longer directly enters the conditional distribution of $\mu$.

In practice this conditional may be sampled using a one-dimensional Metropolis-Hastings update or another suitable scalar sampler.

---

## 3.2 Block B: update latent data `x`

With the newly updated value $\mu^{(t)}$, update

$$
x^{(t)}
\sim
p(x\mid \mu^{(t)},\hat\mu(x)=\mu^*).
$$

The important restriction is

$$
\boxed{x^{(t)}\in\mathcal M_{\mu^*}}.
$$

A global update of all $m$ coordinates is difficult, so the sampler performs **local pairwise moves**.

At each pair update:

- choose two coordinates $i,j$,
- hold all other coordinates fixed,
- replace $(x_i,x_j)$ by $(\tilde x_i,\tilde x_j)$,
- require the pair's contribution to the global score equation to remain unchanged.

Because all other coordinates remain fixed, preserving the pair contribution automatically preserves the global MLE constraint.

---

# 4. Pairwise score-preserving update

Define the single-coordinate score contribution

$$
s(x;\mu^*)
=
\frac{x-\mu^*}
{k+(x-\mu^*)^2}.
$$

For the selected pair $(i,j)$, define

$$
\boxed{
\delta
=
s(x_i;\mu^*)+s(x_j;\mu^*).
}
$$

Explicitly,

$$
\delta
=
\frac{x_i-\mu^*}
{k+(x_i-\mu^*)^2}
+
\frac{x_j-\mu^*}
{k+(x_j-\mu^*)^2}.
$$

The proposed replacement must satisfy

$$
\boxed{
s(\tilde x_i;\mu^*)+s(\tilde x_j;\mu^*)=\delta.}
$$

Then the total score remains unchanged:

$$
\sum_{\ell=1}^m s(\tilde x_\ell;\mu^*)
=
\sum_{\ell=1}^m s(x_\ell;\mu^*)
=0.
$$

This is the central invariant of the latent-data update.

---

# 5. Reparameterization around the observed MLE

Define

$$
y=x-\mu^*.
$$

Then

$$
x=y+\mu^*.
$$

Conditional on the current parameter value $\mu$,

$$
x\sim t_k(\mu),
$$

so

$$
y=x-\mu^*
\sim
t_k(\mu-\mu^*).
$$

Denote the density of $y$ by

$$
f_y(y).
$$

The score contribution becomes

$$
\frac{y}{k+y^2}.
$$

Define the transformation

$$
\boxed{
z=\psi(y)=\frac{y}{k+y^2}.}
$$

For the pair,

$$
z_i=\psi(y_i),
\qquad
z_j=\psi(y_j).
$$

The pair constraint is now simply

$$
\boxed{z_i+z_j=\delta.}
$$

This transformation is useful because it turns the nonlinear Student score constraint into a linear sum constraint in `z`-space.

---

# 6. Geometry of the `y -> z` map

The map is

$$
\psi(y)=\frac{y}{k+y^2}.
$$

Its derivative is

$$
\boxed{
\psi'(y)
=
\frac{k-y^2}{(k+y^2)^2}.
}
$$

The map is not globally one-to-one.

Its extrema occur at

$$
y=\pm\sqrt{k},
$$

so the image of $\psi$ is

$$
\boxed{
-\frac{1}{2\sqrt{k}}
<z<
\frac{1}{2\sqrt{k}}.
}
$$

For most interior $z$-values, there are two possible `y` values that map to the same `z`.

Solving

$$
z=\frac{y}{k+y^2}
$$

for $y$ gives the two inverse branches

$$
\boxed{
y_\pm(z)
=
\frac{1\pm\sqrt{1-4kz^2}}{2z}.
}
$$

Thus

$$
\boxed{
\psi^{-1}(z)
=
\{y_-(z),y_+(z)\}.
}
$$

Implementation code must treat the transformation as **two-branched**. A `z` value generally does not identify a unique latent `y` value.

---

# 7. Pushforward density in `z`-space

We need the density of

$$
z=\psi(y)
$$

when

$$
y\sim f_y.
$$

Because $\psi$ has multiple inverse branches, the pushforward density is the sum of the contributions from every inverse image:

$$
\boxed{
q(z)
=
\sum_{y\in\psi^{-1}(z)}
\frac{f_y(y)}{|\psi'(y)|}.
}
$$

Using

$$
|\psi'(y)|
=
\frac{|k-y^2|}{(k+y^2)^2},
$$

we obtain

$$
\boxed{
q(z)
=
\sum_{y\in\{y_+(z),y_-(z)\}}
f_y(y)
\frac{(k+y^2)^2}{|k-y^2|}.
}
$$

This is the marginal density of a single transformed score contribution.

---

# 8. Conditional distribution of a pair in `z`-space

For the selected pair,

$$
z_i+z_j=\delta.
$$

Once $z_i=z$ is selected,

$$
z_j=\delta-z.
$$

The conditional density for $z_i$, up to normalization, is therefore

$$
\boxed{
\tilde q(z)
\propto
q(z)q(\delta-z).
}
$$

This density must be restricted to the region where **both** transformed coordinates are valid:

$$
z\in
\left(-\frac{1}{2\sqrt{k}},\frac{1}{2\sqrt{k}}\right)
$$

and

$$
\delta-z
\in
\left(-\frac{1}{2\sqrt{k}},\frac{1}{2\sqrt{k}}\right).
$$

Equivalently, the valid support for `z_i` is the intersection

$$
\boxed{
\left(-\frac{1}{2\sqrt{k}},\frac{1}{2\sqrt{k}}\right)
\cap
\left(\delta-\frac{1}{2\sqrt{k}},\delta+\frac{1}{2\sqrt{k}}\right).
}
$$

After drawing

$$
z_i\sim\tilde q,
$$

set

$$
\boxed{z_j=\delta-z_i.}
$$

The pair constraint is then satisfied exactly in transformed space.

---

# 9. Recovering latent `y` values: branch sampling

Given $z_i,z_j$, each transformed coordinate has two possible inverse values:

$$
\psi^{-1}(z_i)
=
\{y_{i,-},y_{i,+}\},
$$

$$
\psi^{-1}(z_j)
=
\{y_{j,-},y_{j,+}\}.
$$

Therefore there are four candidate latent pairs:

$$
(y_{i,-},y_{j,-}),
\quad
(y_{i,-},y_{j,+}),
\quad
(y_{i,+},y_{j,-}),
\quad
(y_{i,+},y_{j,+}).
$$

Each candidate pair is assigned probability proportional to its density with
respect to the transformed coordinates.  The inverse-map Jacobians remain in
this conditional distribution:

$$
\boxed{
w_{ab}
\propto
\frac{f_y(y_{i,a})f_y(y_{j,b})}
{|\psi'(y_{i,a})|\,|\psi'(y_{j,b})|},
\qquad
a,b\in\{-,+\}.
}
$$

Equivalently,

$$
w_{ab}\propto
f_y(y_{i,a})f_y(y_{j,b})
\left|\frac{dy_{i,a}}{dz_i}\right|
\left|\frac{dy_{j,b}}{dz_j}\right|.
$$

The four weights factor into an `i` contribution and a `j` contribution, so
the two branch labels may be sampled independently using one-coordinate
log-weights

$$
\log w_i(a)=\log f_y(y_{i,a})-\log|\psi'(y_{i,a})|,
$$

and similarly for coordinate `j`.  The Jacobians in this branch draw do not
double-count those used in $q(z)$: sampling the marginal $q(z)$ and then this
conditional branch distribution reconstructs the joint transformed density.

Normalize the four weights:

$$
P(a,b\mid z_i,z_j)
=
\frac{w_{ab}}
{\sum_{a',b'}w_{a'b'}}.
$$

Sample one branch pair according to these probabilities.

Finally transform back to the original latent coordinates:

$$
\boxed{
\tilde x_i=y_i+\mu^*,
\qquad
\tilde x_j=y_j+\mu^*.
}
$$

---

# 10. Full pair-update algorithm

For a selected pair `(i, j)`:

1. Compute

   $$
   y_i=x_i-\mu^*,
   \qquad
   y_j=x_j-\mu^*.
   $$

2. Compute

   $$
   z_i=\psi(y_i),
   \qquad
   z_j=\psi(y_j).
   $$

3. Store the pairwise score contribution

   $$
   \delta=z_i+z_j.
   $$

4. Build the one-dimensional conditional target

   $$
   \tilde q(z)
   \propto
   q(z)q(\delta-z)
   $$

   on the support where both `z` and `delta-z` are valid.

5. Draw a new

   $$
   z_i'\sim\tilde q.
   $$

6. Set

   $$
   z_j'=\delta-z_i'.
   $$

7. Compute the two inverse branches for each coordinate:

   $$
   y_{i,\pm}'=y_\pm(z_i'),
   \qquad
   y_{j,\pm}'=y_\pm(z_j').
   $$

8. Form the four possible branch pairs.

9. Weight each pair by

   $$
   \frac{f_y(y_i')f_y(y_j')}
   {|\psi'(y_i')|\,|\psi'(y_j')|}.
   $$

10. Sample one branch pair.

11. Transform back:

   $$
   x_i'=y_i'+\mu^*,
   \qquad
   x_j'=y_j'+\mu^*.
   $$

12. Replace only coordinates `i` and `j` in the latent vector.

The new pair satisfies

$$
\psi(x_i'-\mu^*)+
\psi(x_j'-\mu^*)
=
\delta,
$$

so the global score constraint remains unchanged.

---

# 11. Full Gibbs iteration

A complete Gibbs iteration is conceptually:

```text
Given current state (mu, x):

1. Update mu conditional on x.

2. Randomly pair the coordinates of x.

3. For every selected pair (i, j):
      a. compute the pair score contribution delta
      b. sample a new score contribution z_i
      c. set z_j = delta - z_i
      d. invert both z values through both y branches
      e. sample one of the four branch combinations
      f. write the resulting x_i, x_j back

4. Return updated (mu, x).
```

If all pair updates preserve their own `delta`, then after the sweep

$$
C(x)=0
$$

remains true.

---

# 12. Initialization requirement

The chain must start from

$$
\boxed{x^{(0)}\in\mathcal M_{\mu^*}.}
$$

That means

$$
\sum_i
\frac{x_i^{(0)}-\mu^*}
{k+(x_i^{(0)}-\mu^*)^2}
=0.
$$

A simple way to guarantee this is to initialize with symmetric pairs around $\mu^*$:

$$
(\mu^*+a,\mu^*-a).
$$

Their score contributions cancel because

$$
\psi(a)+\psi(-a)=0.
$$

Similarly, the fully central initialization

$$
x_i=\mu^*\quad\forall i
$$

satisfies the constraint because each score term is zero.

---

# 13. Quantities that must remain invariant

The implementation should explicitly monitor these identities.

## Global MLE score constraint

After every sweep,

$$
\boxed{
\left|
\sum_i
\frac{x_i-\mu^*}
{k+(x_i-\mu^*)^2}
\right|
\approx0.
}
$$

## Pairwise identity

For every pair update,

$$
\boxed{
\psi(x_i^{\text{old}}-\mu^*)
+
\psi(x_j^{\text{old}}-\mu^*)
=
\psi(x_i^{\text{new}}-\mu^*)
+
\psi(x_j^{\text{new}}-\mu^*).
}
$$

This is the local identity responsible for preserving the global manifold constraint.

---

# 14. Important numerical details

## 14.1 Valid `z` support

Every proposed transformed coordinate must satisfy

$$
|z|<\frac{1}{2\sqrt{k}}.
$$

The paired coordinate `delta - z` must satisfy the same restriction.

The implementation should determine the feasible interval before sampling rather than repeatedly proposing invalid values whenever possible.

---

## 14.2 Two inverse branches are essential

The map

$$
y\mapsto\frac{y}{k+y^2}
$$

is not one-to-one.

Ignoring one branch changes the latent target distribution and destroys correctness.

Both inverse branches must therefore be considered except at special points where they coincide.

---

## 14.3 Behavior near `z = 0`

The explicit formula

$$
y_\pm(z)
=
\frac{1\pm\sqrt{1-4kz^2}}{2z}
$$

contains division by `z`.

A production implementation must therefore handle values numerically close to zero carefully rather than directly evaluating the formula without safeguards.

Mathematically, the two branches correspond to very different `y` scales near zero: one branch is near the center, while the other can be extremely large in magnitude.

This branch structure is particularly important in heavy-tailed regimes.

---

## 14.4 Behavior near `|y| = sqrt(k)`

At

$$
|y|=\sqrt{k},
$$

we have

$$
\psi'(y)=0.
$$

Therefore the Jacobian factor

$$
\frac{1}{|\psi'(y)|}
$$

becomes large near the extrema of the map.

Numerical implementations should use stable log-density calculations and avoid unnecessary direct exponentiation.

---

## 14.5 Log-domain evaluation

The pushforward density is a sum over branches:

$$
q(z)
=
\sum_b
\frac{f_y(y_b)}{|\psi'(y_b)|}.
$$

In code this should generally be evaluated as

```text
log q(z)
= logsumexp_b [ log f_y(y_b) - log |psi'(y_b)| ]
```

rather than by directly evaluating branch densities and summing them.

Likewise,

$$
\log \tilde q(z)
=
\log q(z)+\log q(\delta-z)+\text{constant}.
$$

This is especially important for heavy tails and extreme inverse branches.

---

# 15. Conceptual interpretation

The sampler works because the Student MLE constraint can be rewritten coordinatewise as

$$
\sum_i z_i=0,
\qquad
z_i=\frac{x_i-\mu^*}{k+(x_i-\mu^*)^2}.
$$

The latent-data update therefore operates by redistributing the score contributions between pairs while preserving their sum.

For a pair,

$$
z_i+z_j=\delta
$$

is held fixed.

The difficult part is that the map from latent data to score contribution is many-to-one: the same `z` may correspond to a central latent value or a much more extreme latent value. The branch-sampling step restores this information by sampling among all inverse images with the appropriate Student-t density and inverse-Jacobian weights.

So the latent Gibbs update has two conceptually distinct pieces:

1. **move along the linear score constraint in `z`-space**, and
2. **sample which latent-data inverse branch generated each new score contribution**.

---

# 16. Minimal mathematical pseudocode

```text
Input:
    observed MLE mu_star
    degrees of freedom k
    prior pi(mu)
    initial x satisfying C(x)=0
    initial mu

for t = 1,...,T:

    # A. Parameter update
    sample mu ~ pi(mu) * product_i t_k(x_i | mu)

    # B. Latent-data sweep
    randomly pair coordinates

    for each pair (i,j):

        yi = x_i - mu_star
        yj = x_j - mu_star

        zi = yi / (k + yi^2)
        zj = yj / (k + yj^2)

        delta = zi + zj

        sample zi_new from
            q(zi_new) * q(delta - zi_new)
        restricted to valid z support

        zj_new = delta - zi_new

        branches_i = inverse_branches(zi_new)
        branches_j = inverse_branches(zj_new)

        candidates = CartesianProduct(branches_i, branches_j)

        weight candidate (yi_new, yj_new) by
            f_y(yi_new) * f_y(yj_new)
            / (abs(psi_prime(yi_new)) * abs(psi_prime(yj_new)))

        sample one candidate according to normalized weights

        x_i = yi_new + mu_star
        x_j = yj_new + mu_star

    assert abs(C(x)) is within tolerance

return retained mu samples and, when needed, latent x samples
```

---

# 17. Core implementation contract for coding agents

Any implementation of this sampler should preserve the following contract:

1. `mu_star` is fixed throughout the chain; it is the observed released MLE.
2. `mu` is a sampled model parameter and changes during the chain.
3. `x` is latent and must always satisfy the observed-MLE score constraint.
4. A latent pair update must preserve its previous `delta` exactly up to numerical tolerance.
5. `z = y / (k + y^2)` has bounded support and two inverse branches.
6. The pushforward density `q(z)` must sum contributions from both inverse branches.
7. After drawing a pair in `z`-space, the implementation must sample the inverse branches rather than deterministically choosing one.
8. Numerical evaluation of branch densities and Jacobians should be done in the log domain.
9. The global score residual should be monitored as a correctness diagnostic.
10. Changes that ignore the branch structure or permit pair updates to change `delta` are changes to the mathematical algorithm, not mere implementation optimizations.

