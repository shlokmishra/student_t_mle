# Independent read-only mathematical review

Reviewer: `/root/selected_gibbs_math_review`. No source edits or experiments were delegated.

The reviewer independently derived the pair-chart conditional density
`w_i(z,b_i) w_j(delta-z,b_j) R(x) dz`, with
`w=f_mu/abs(psi')`, and checked cancellation under the joint proposal
`h(z'|z) [w_i'/q_i'] [w_j'/q_j']`. The resulting ratio is
`(q_i' q_j')/(q_i q_j) * h(z|z')/h(z'|z) * R(x')/R(x)`.
For a fixed-interval truncated Gaussian, the Hastings contribution is
`log Z(z) - log Z(z')`.

The code review found no material acceptance/state-cache error: entire-pair
rejection, sequential use of the updated full dataset, global selection, the
single finite inverse at zero, chart boundaries and selector failures are handled.
Fixed known-scale constants cancel.

Review caveats and their treatment:

- Pair moves preserve an initializer's tiny score offset. The implementation
  records initial residual and accumulated drift; this is not exact-arithmetic
  certification of floating-point conditioning.
- Runtime uses intended proposed scores. Reconstruction is checked explicitly,
  and tests compare its log ratio to one recomputed from actual proposed data.
- The unchanged parameter update rejects nonfinite differences. The new wrapper
  checks both log densities first and raises on unresolved values.
- The inherited parameter uniform floor remains unchanged and is disclosed as
  a finite-precision approximation separate from the mathematical target proof.

The reviewer did not certify global irreducibility or finite-run convergence.
