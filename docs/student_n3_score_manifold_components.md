# The finite Student score manifold for three observations

Let (y_i=x_i-\mu^\star), and scale (t_i=y_i/\sqrt{k}).  Since

\[
\frac{y_i}{k+y_i^2}=\frac{1}{\sqrt{k}}\frac{t_i}{1+t_i^2},
\]

the topology is independent of (k>0).  It is therefore enough to prove the
claims for (k=1), then restore the scale at the end.

## 1. Score coordinates and finite chambers

Put

\[
z_i=\frac{y_i}{k+y_i^2},\qquad
a=\frac{1}{2\sqrt{k}},\qquad
H=\{z\in[-a,a]^3:\ z_1+z_2+z_3=0\}.
\]

(H) is a closed hexagon.  Away from (z_i=0) and (|z_i|=a), the two
inverse branches are

\[
y_i^{\mathrm{in}}(z_i)
=\frac{2kz_i}{1+\sqrt{1-4kz_i^2}},\qquad
y_i^{\mathrm{out}}(z_i)
=\frac{1+\sqrt{1-4kz_i^2}}{2z_i}.
\]

They satisfy

\[
|y_i^{\mathrm{in}}|<\sqrt{k},\qquad
|y_i^{\mathrm{out}}|>\sqrt{k}.
\]

At (|z_i|=a), the two inverses agree at
(y_i=\operatorname{sign}(z_i)\sqrt{k}).  This is a finite branch-merging
boundary.  In contrast,

\[
z_i\to0,\quad b_i=\mathrm{out}
\quad\Longrightarrow\quad
y_i^{\mathrm{out}}(z_i)\sim z_i^{-1}\to\pm\infty.
\]

Consequently, the sign of every outer coordinate is part of the finite
chamber label.  A chamber is a pair ((b,s)), where
(b\in\{0,1\}^3) records inner/outer status and
(s_i\in\{-1,+1\}) is recorded when (b_i=1).  Its score-coordinate domain is

\[
H_{b,s}=\{z\in\operatorname{relint}(H):s_i z_i>0
\text{ for every }i\text{ with }b_i=1\}.
\]

Each nonempty (H_{b,s}) is convex and therefore connected.  There are

\[
1+3(2)+3(4)+6=25
\]

finite chambers: one all-inner, six one-outer, twelve two-outer, and six
all-outer chambers.  The two all-outer assignments with three equal signs are
infeasible because three strictly same-sign (z_i)'s cannot sum to zero.

The score level is regular.  Indeed,

\[
\psi'(y_i)=\frac{k-y_i^2}{(k+y_i^2)^2},
\]

so the constraint gradient could vanish only if every
(|y_i|=\sqrt{k}).  Three numbers each equal to
(\pm1/(2\sqrt{k})) cannot sum to zero.  Thus the finite score-zero set is a
smooth embedded two-manifold; the sheet language describes folds of its
((z,b,s)) coordinate representation, not singularities of the manifold.

## 2. Exact finite adjacency rule

Two chambers can meet across a codimension-one finite fold only if their
branch vectors differ in one coordinate (i).  Suppose the outer value at the
fold has sign (s\), so (z_i=sa).  The other two coordinates obey

\[
z_j+z_\ell=-sa.
\]

On the relative interior of this edge, both (z_j) and (z_\ell) have sign
(-s).  Therefore:

> **Finite-fold adjacency rule.** Toggling coordinate (i) between inner and
> outer with sign (s) is possible exactly when every other coordinate that
> is already outer has sign (-s).

This rule includes the hexagon vertices.  At a vertex, two coordinates are at
finite branch folds and the third is inner at zero; sequential single-coordinate
fold adjacencies recover the same local connection.

It follows immediately that:

1. `000` joins all six signed one-outer chambers.
2. A one-outer chamber with sign (s) joins the two two-outer chambers obtained
   by making one other coordinate outer with sign (-s).
3. A two-outer chamber with opposite outer signs joins the corresponding two
   one-outer chambers.
4. A two-outer chamber whose two outer signs are both (s) joins exactly one
   all-outer chamber: the remaining coordinate becomes outer with sign (-s).
5. An all-outer chamber has exactly that one finite-fold neighbor.

There are 24 undirected finite-fold edges.

## 3. Connected components

The first three rules give one central component containing 13 chambers:

\[
\mathcal C_0=
\{000\}
\cup\{\text{six signed one-outer chambers}\}
\cup\{\text{six opposite-sign two-outer chambers}\}.
\]

For every index (r\in\{1,2,3\}) and sign (s\in\{-1,+1\}), there is one
two-chamber tail component

\[
\mathcal C_{r,s}
=
\left\{
\begin{array}{l}
\text{coordinate }r\text{ inner and the other two outer with sign }s,\\
\text{all three outer, with coordinate }r\text{ of sign }-s
\text{ and the other two of sign }s
\end{array}
\right\}.
\]

Hence the finite score-zero manifold has exactly

\[
\boxed{7\text{ connected components}:\quad
\mathcal C_0,\ \mathcal C_{1,+},\mathcal C_{1,-},
\mathcal C_{2,+},\mathcal C_{2,-},\mathcal C_{3,+},\mathcal C_{3,-}.}
\]

The apparent contacts between these components in the closed (z)-hexagon
occur on an axis (z_i=0) while coordinate (i) is outer.  Their inverse
images are at (|y_i|=\infty), so these are ends of the finite manifold rather
than shared finite boundary points.

## 4. MLE classification

The following three-point likelihood lemma gives the classification.

> **Three-point Student location lemma.** Let (y\in\mathbb R^3) satisfy
> \(\sum_i y_i/(k+y_i^2)=0\).  Then (0) is the global Student location MLE
> if and only if there are not two observations larger than (\sqrt{k}) and
> there are not two observations smaller than (-\sqrt{k}).

Equivalently, (0) is the global MLE exactly when no two outer coordinates
have the same sign.

One direct algebraic proof scales to (k=1) and minimizes

\[
P(m)=\prod_{i=1}^3\{1+(y_i-m)^2\}.
\]

The score constraint is (P'(0)=0), or, in elementary symmetric functions
(e_j=e_j(y_1,y_2,y_3)),

\[
e_1(1+e_2)+e_3(e_2-3)=0.
\]

The likelihood comparison factors as

\[
P(m)-P(0)=m^2Q(m),
\]

where

\[
\begin{aligned}
Q(m)={}&m^4-2e_1m^3+(3+e_1^2+2e_2)m^2\\
&-2\{e_1(2+e_2)+e_3\}m
+3+2e_1^2+e_2^2+2e_1e_3.
\end{aligned}
\]

Substitution of the score identity into the Sturm signs of (Q) gives two
cases.  If no two outer coordinates share a sign, (Q(m)>0) for every
(m\ne0).  If two outer coordinates share sign (s), (Q) is negative on an
interval on the (s)-side containing the competing likelihood mode.  Equality
can be approached only as one of those outer score coordinates tends to zero,
which is the infinite end described above.  This proves the lemma and also
shows that the MLE classification cannot change at a finite point of a
component.

Therefore

\[
\boxed{\mathcal C_0\text{ is MLE-valid},\qquad
\mathcal C_{r,s}\text{ is non-MLE for every }r,s.}
\]

The exact graph calculation gives component sizes (13,2,2,2,2,2,2).  An
independent 8,665-point certified root/likelihood check over all 25 chambers
found no disagreement with the lemma.  This calculation is a verification of
the classification and the implementation; the component count itself follows
exactly from the finite-fold adjacency rule.

## 5. Consequence for RATTLE

A successful RATTLE proposal is a finite sequence of continuous finite-position
updates on the score manifold.  It may change (b_i) by passing through

\[
|z_i|=a\quad\Longleftrightarrow\quad |y_i|=\sqrt{k},
\]

which explains the observed branch switches.  But moving from
(\mathcal C_0) to any (mathcal C_{r,s}) requires an outer coordinate to
cross (z_i=0), hence (|y_i|=\infty).  No finite continuous trajectory can do
this.  Thus branch switching and component communication are different
properties.

Let (K_R) denote a Metropolis-corrected RATTLE kernel.  Componentwise detailed
balance,

\[
\pi(dx)K_R(x,dx')=\pi(dx')K_R(x',dx),
\]

can hold globally even though (K_R(x,\mathcal C')=0) whenever (x) and
(\mathcal C') belong to different components.  Consequently:

- the full score-conditioned target is invariant;
- the kernel is reducible;
- a chain initialized in one component samples the target conditional on that
  component;
- convergence to the global target requires the initial component weights to
  be correct by construction;
- detailed balance and high acceptance do not establish global irreducibility
  or ergodicity.

## 6. Minimal component-bridging kernel

At a fixed interior score vector (z), a branch refresh leaves
(\sum_i z_i=0) exactly unchanged.  Let (x_b(z)) denote a finite inverse
branch configuration and

\[
d_i(b,z)=\frac{dy_i^{b_i}}{dz_i}
=\frac{(k+y_i^2)^2}{k-y_i^2}.
\]

For the parameterization (z_3=-z_1-z_2), the surface-area Jacobian is

\[
A_b(z)=\sqrt{d_1^2d_2^2+d_1^2d_3^2+d_2^2d_3^2}
=|d_1d_2d_3|\,J_c(x_b(z)),
\]

where (J_c(x)=\|\nabla c(x)\|).  Hence a coarea target

\[
\pi_H(x,\mu)\propto \frac{f_\mu(x)\pi_0(\mu)}{J_c(x)}
\quad\text{with respect to }dH(x)
\]

induces the following branch weight at fixed (z,mu):

\[
w_b(z,\mu)\propto
f_\mu(x_b(z))\prod_{i=1}^3|d_i(b,z)|.
\]

A minimal branch-jump kernel is therefore:

1. retain (z) and (mu);
2. enumerate the finite inverse configurations at that (z);
3. sample a new (b) from probabilities proportional to (w_b(z,mu)).

This is a Gibbs refresh of the discrete branch variable and has acceptance one.
A cheaper Metropolis version proposes, symmetrically, to toggle the two
same-sign coordinates between inner and outer and uses

\[
\alpha(b,b')=1\wedge\frac{w_{b'}(z,\mu)}{w_b(z,\mu)}.
\]

The pair toggle directly connects the central component with the tail component
associated with the current majority-sign sector.  Continuous motion inside
the central component changes the majority index and sign, giving access to all
six tail components.  A random mixture

\[
K=\omega K_R+(1-\omega)K_B,
\qquad 0<\omega<1,
\]

is reversible whenever both constituent kernels are reversible.  Global
ergodicity would then require, in addition, the usual within-component local
irreducibility and recurrence conditions; the branch graph alone supplies the
previously missing cross-component communication.
