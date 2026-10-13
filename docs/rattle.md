# RATTLE / Constrained HMC Algorithm (Lelièvre–Rousset–Stoltz)

## Target

We want to sample a probability measure supported on a smooth constraint manifold

$$
\mathcal M = \{ q \in \mathbb R^d : \xi(q)=0 \},
$$

where

$$
\xi:\mathbb R^d \to \mathbb R^m,
\qquad m<d.
$$

For the Student-t score-conditioned sampler, the target density on the
manifold is the coarea-conditioned density

$$
\nu(dq) \propto \exp(-V(q))\,G_M(q)^{-1/2}\,\sigma_{\mathcal M}(dq),
$$

Introduce momentum \(p\) constrained to the cotangent space

$$
T_q^*\mathcal M
=
\left\{
p\in\mathbb R^d :
[\nabla \xi(q)]^T M^{-1}p = 0
\right\}.
$$

Equivalently, the extended Hamiltonian uses the effective potential
\(U(q)=V(q)+\tfrac12\log G_M(q)\):

$$
H(q,p)
=
U(q)
+
\frac12 p^T M^{-1}p.
$$

The phase-space target is

$$
\mu(dq,dp)
\propto
\exp(-H(q,p))
\,\sigma_{T^*\mathcal M}(dq,dp).
$$

Sampling \((q,p)\sim\mu\) gives \(q\sim\nu\) marginally.

---

# One RATTLE Step

Input:

$$
(q^n,p^n)\in T^*\mathcal M,
\qquad
\Delta t>0.
$$

A RATTLE step computes

$$
(q^{n+1},p^{n+1})\in T^*\mathcal M.
$$

## 1. Half Momentum Update + Constraint Multiplier

$$
p^{n+1/2}
=
p^n
-
\frac{\Delta t}{2}\nabla V(q^n)
+
\nabla\xi(q^n)\lambda^{n+1/2}.
$$

---

## 2. Position Update

$$
q^{n+1}
=
q^n
+
\Delta t\,M^{-1}p^{n+1/2}.
$$

Choose \(\lambda^{n+1/2}\) so that

$$
\xi(q^{n+1})=0.
$$

Equivalently, define the unconstrained position

$$
\widetilde q^n
=
q^n
+
\Delta t M^{-1}
\left[
p^n
-
\frac{\Delta t}{2}\nabla V(q^n)
\right].
$$

Then project by solving

$$
q^{n+1}
=
\widetilde q^n
+
M^{-1}\nabla\xi(q^n)\theta,
$$

subject to

$$
\xi(q^{n+1})=0.
$$

The multipliers satisfy

$$
\theta
=
\Delta t\,\lambda^{n+1/2}.
$$

In practice this nonlinear projection is solved using Newton's method.

---

## 3. Second Half Momentum Update

$$
p^{n+1}
=
p^{n+1/2}
-
\frac{\Delta t}{2}\nabla V(q^{n+1})
+
\nabla\xi(q^{n+1})\lambda^{n+1}.
$$

Choose \(\lambda^{n+1}\) so that the final momentum is tangent:

$$
[\nabla\xi(q^{n+1})]^T M^{-1} p^{n+1}=0.
$$

Define the Gram matrix

$$
G_M(q)
=
[\nabla\xi(q)]^T
M^{-1}
\nabla\xi(q).
$$

Then

$$
\lambda^{n+1}
=
-
G_M(q^{n+1})^{-1}
[\nabla\xi(q^{n+1})]^T
M^{-1}
\left[
p^{n+1/2}
-
\frac{\Delta t}{2}\nabla V(q^{n+1})
\right].
$$

Equivalently,

$$
p^{n+1}
=
\Pi_{T^*_{q^{n+1}}\mathcal M}
\left[
p^{n+1/2}
-
\frac{\Delta t}{2}\nabla V(q^{n+1})
\right],
$$

where

$$
\Pi_{T_q^*\mathcal M}
=
I
-
\nabla\xi(q)
G_M(q)^{-1}
[\nabla\xi(q)]^T
M^{-1}.
$$

---

# RATTLE with Momentum Reversal

The proposal map is

$$
\Phi_{\Delta t}(q,p)
=
(q_1,-p_1),
$$

where \((q_1,p_1)\) is obtained by one RATTLE step from \((q,p)\).

Momentum reversal is included because the Metropolized map should behave like an involution.

---

# Reverse Projection Check

The practical algorithm checks that the proposal is actually reversible.

## Forward Move

Run one RATTLE step:

$$
(q,p)
\longrightarrow
(q_1,p_1).
$$

The proposed state becomes

$$
(q_1,-p_1).
$$

If the forward projection fails, reject immediately.

---

## Reverse Move

Starting from \((q_1,-p_1)\), run another RATTLE step:

$$
(q_1,-p_1)
\longrightarrow
(q_2,-p_2).
$$

Check whether this returns to the original state:

$$
(q_2,p_2)=(q,p).
$$

In practice it is sufficient to check only the position:

$$
\|q_2-q\| < \eta_{\rm rev}.
$$

Reject if

$$
\|q_2-q\| \ge \eta_{\rm rev}.
$$

Only if the reverse check passes do we proceed to the Metropolis step.

---

# Metropolis Acceptance Step

If forward and reverse projections both succeed, accept

$$
(q',p')=(q_1,-p_1)
$$

with probability

$$
\alpha
=
\min\left(
1,
\exp\bigl[-H(q',p')+H(q,p)\bigr]
\right).
$$

Equivalently,

$$
\alpha
=
\min\left(
1,
\exp\bigl[H(q,p)-H(q',p')\bigr]
\right).
$$

If accepted:

$$
(q^{n+1},p^{n+1})
=
(q',p').
$$

If rejected:

$$
(q^{n+1},p^{n+1})
=
(q,p).
$$

---

# Practical Constrained HMC Algorithm (Full Momentum Refreshment)

Parameters:

- timestep \(\Delta t\)
- mass matrix \(M\)
- reverse-check tolerance \(\eta_{\rm rev}\)
- Newton tolerance \(\varepsilon_{\rm newt}\)

Current position:

$$
q\in\mathcal M.
$$

## Step 1. Fresh Tangent Momentum

Sample

$$
G\sim N(0,I_d).
$$

Project \(G\) onto \(T_q^*\mathcal M\):

$$
p
=
\Pi_{T_q^*\mathcal M}G,
$$

with

$$
[\nabla\xi(q)]^T M^{-1} p = 0.
$$

---

## Step 2. Forward RATTLE Proposal

Half momentum update:

$$
\widetilde p
=
p
-
\frac{\Delta t}{2}\nabla U(q).
$$

Unconstrained position update:

$$
\widetilde q
=
q
+
\Delta t M^{-1}\widetilde p.
$$

Solve the nonlinear projection:

$$
q_{\rm prop}
=
\widetilde q
+
M^{-1}\nabla\xi(q)\theta,
$$

such that

$$
\xi(q_{\rm prop})=0.
$$

If Newton fails, reject.

Correct half-step momentum:

$$
\widetilde p
\leftarrow
\widetilde p
+
\nabla\xi(q)\frac{\theta}{\Delta t}.
$$

Second half momentum update:

$$
\widetilde p
\leftarrow
\widetilde p
-
\frac{\Delta t}{2}\nabla U(q_{\rm prop}).
$$

Project momentum:

$$
p_{\rm prop}
=
\Pi_{T^*_{q_{\rm prop}}\mathcal M}(\widetilde p).
$$

Forward output:

$$
(q_{\rm prop},p_{\rm prop}).
$$

Proposed HMC state:

$$
(q',p')
=
(q_{\rm prop},-p_{\rm prop}).
$$

---

## Step 3. Reverse Projection Check

Run another RATTLE step from \((q',p')\).

Let the reverse result be

$$
(\widehat q,\widehat p).
$$

Reject if:

$$
\|\widehat q-q\|
\ge
\eta_{\rm rev}.
$$

or if the reverse Newton solve fails.

---

## Step 4. Metropolis Accept/Reject

Accept with probability

$$
\alpha
=
\min\left(
1,
\exp\bigl[
H(q,p)-H(q',p')
\bigr]
\right).
$$

---

The sampler stores \(q^{n+1}\) and discards momentum.  The next iteration
draws a new tangent Gaussian momentum.

## Validated Student-t trajectory settings

For the production Student-t implementation, trajectory parameters are
problem-size dependent.  The following settings were tuned at \(k=2\) with
central initialization, Gram correction, and the mandatory reverse check.  A
setting is retained only when all validation chains had zero projection and
reverse-check failures.

| n | step size | RATTLE steps |
|---:|---:|---:|
| 10 | 0.15 | 20 |
| 50 | 0.50 | 10 |
| 200 | 0.50 | 5 |
| 500 | 0.25 | 20 |

They are stored in
`configs/student_rattle_k2_tuned_20260817.json` and can be applied by the
cost-audit runner with:

```bash
python scripts/run_cost_audit.py --models student_t --methods rattle \
  --k-values 2 --n-values 10,50,200,500 \
  --rattle-settings-json configs/student_rattle_k2_tuned_20260817.json
```

Do not reuse this table unchanged for a different \(k\), initialization, or
materially different geometry.  For acceptance-only tuning of both the
Student-location Gibbs and RATTLE samplers, run
`scripts/tune_student_acceptance.py`.  Set `--rattle-num-steps` to control the
trajectory length; the tuner then selects the step size and proposal scales
only by distance to the requested acceptance targets.  Projection and reverse
failures remain hard safety exclusions.  Use `tune_student_rattle_latent.py`
only when the goal is explicitly to rank settings by latent mixing and ESS.

---

# Minimal Coding-Agent Pseudocode

```python
def constrained_hmc_step(q, dt, M, U, gradU, xi, grad_xi, eta_rev, newton_tol):

    G = normal_like(q)
    p = project_momentum(q, G, M, grad_xi)

    q0, p0 = q, p

    success_fwd, q_prop, p_prop = rattle_step(
        q0, p0, dt, M, U, gradU, xi, grad_xi, newton_tol
    )

    if not success_fwd:
        return q0, "reject_forward_projection"

    q_prime = q_prop
    p_prime = -p_prop

    success_rev, q_back, p_back = rattle_step(
        q_prime, p_prime, dt, M, U, gradU, xi, grad_xi, newton_tol
    )

    if not success_rev:
        return q0, "reject_reverse_projection"

    if norm(q_back - q0) >= eta_rev:
        return q0, "reject_non_reversible"

    log_alpha = (
        hamiltonian(q0, p0, M, U)
        - hamiltonian(q_prime, p_prime, M, U)
    )

    if log_uniform() < min(0.0, log_alpha):
        q = q_prime
        status = "accept"
    else:
        q = q0
        status = "reject_metropolis"

    return q, status
```
