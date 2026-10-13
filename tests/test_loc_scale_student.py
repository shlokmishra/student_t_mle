import math

import numpy as np
import pytest

from models.loc_scale import student_gibbs as loc_scale_student
from models.loc_scale import student_rattle as loc_scale_student_rattle


def _params(**updates):
    params = {
        "n": 10,
        "k": 2.0,
        "num_iterations_T": 12,
        "proposal_std_mu": 0.2,
        "proposal_std_logsigma": 0.1,
        "prior_mean": 0.0,
        "prior_std": 5.0,
        "prior_logsigma_mean": 0.0,
        "prior_logsigma_std": 1.0,
        "slice_max_shrink_steps": 200,
        "rattle_step_size": 0.003,
        "rattle_num_steps": 2,
        "rattle_reverse_position_tol": 1e-7,
    }
    params.update(updates)
    return params


@pytest.mark.parametrize("n", [3, 4, 5, 10])
@pytest.mark.parametrize("k", [1.0, 2.0, 3.0])
def test_initial_state_satisfies_both_student_location_scale_scores(n, k):
    mu_star, sigma_star = -0.4, 1.7
    angles = loc_scale_student.init_angles(n, k)
    x = loc_scale_student.init_x(mu_star, sigma_star, n, k)
    residual = loc_scale_student.score_residual(x, mu_star, sigma_star, k)

    assert np.all(np.abs(angles) < math.pi)
    np.testing.assert_allclose(residual, np.zeros(2), atol=2e-13)
    assert loc_scale_student._strict_local_mle(loc_scale_student._geometry_sums(angles, k))


def test_closed_form_pair_reconstruction_preserves_triple_resultant():
    angles = loc_scale_student.init_angles(10, 2.0)
    old = angles[:3]
    resultant = np.array([np.cos(old).sum(), np.sin(old).sum()])
    candidate_t = old[0] + 0.08
    phi_j, phi_l, jacobian, feasible = loc_scale_student._reconstruct_pair(
        resultant[0], resultant[1], candidate_t
    )

    assert feasible
    assert jacobian > 0.0
    reconstructed = np.array([candidate_t, phi_j, phi_l])
    new_resultant = np.array([np.cos(reconstructed).sum(), np.sin(reconstructed).sum()])
    np.testing.assert_allclose(new_resultant, resultant, atol=2e-14)


def test_scalar_target_contains_density_pair_jacobian_and_mle_jacobian():
    k = 2.0
    angles = loc_scale_student.init_angles(10, k)
    triple = angles[:3]
    resultant_x = float(np.cos(triple).sum())
    resultant_y = float(np.sin(triple).sum())
    base_sums = loc_scale_student._geometry_sums(angles, k)
    for phi in triple:
        base_sums -= np.asarray(
            loc_scale_student._geometry_terms_from_y(loc_scale_student._angle_to_y(phi, k), k)
        )

    target, phi_j, phi_l = loc_scale_student._triple_log_target(
        triple[0], resultant_x, resultant_y, base_sums, 0.2, 1.3, 0.0, 1.0, k
    )
    candidate_sums = base_sums.copy()
    for phi in (triple[0], phi_j, phi_l):
        candidate_sums += np.asarray(
            loc_scale_student._geometry_terms_from_y(loc_scale_student._angle_to_y(phi, k), k)
        )
    _, _, pair_jacobian, feasible = loc_scale_student._reconstruct_pair(
        resultant_x, resultant_y, triple[0]
    )
    manual = sum(
        loc_scale_student._log_angle_density(phi, 0.2, 1.3, 0.0, 1.0, k)
        for phi in (triple[0], phi_j, phi_l)
    )
    manual += math.log(abs(loc_scale_student._det_h_numerator(candidate_sums)))
    manual -= math.log(pair_jacobian)

    assert feasible
    np.testing.assert_allclose(target, manual, rtol=1e-13, atol=1e-13)


def test_gibbs_slice_updates_t_and_preserves_both_scores():
    out = loc_scale_student.run_gibbs(
        np.array([0, 7], dtype=np.uint32), 0.0, 1.0, _params(), verbose=False
    )

    assert out["latent_update"] == "triple_angle_slice"
    assert out["mle_jacobian_included"] is True
    assert out["slice_acceptance_rate"] > 0.0
    assert np.any(np.linalg.norm(np.diff(out["x_chain"], axis=0), axis=1) > 1e-8)
    assert out["max_score_residual"] < 1e-11
    for x in out["x_chain"]:
        np.testing.assert_allclose(
            loc_scale_student.score_residual(x, 0.0, 1.0, 2.0), np.zeros(2), atol=1e-11
        )


def test_rattle_analytic_potential_gradient_matches_finite_difference():
    angles = loc_scale_student.init_angles(10, 2.0)
    analytic = loc_scale_student_rattle._grad_potential(angles, 0.2, 1.3, 0.0, 1.0, 2.0)
    numerical = np.empty_like(angles)
    step = 1e-6
    for index in range(angles.size):
        plus = angles.copy()
        minus = angles.copy()
        plus[index] += step
        minus[index] -= step
        numerical[index] = (
            loc_scale_student_rattle._potential_energy(plus, 0.2, 1.3, 0.0, 1.0, 2.0)
            - loc_scale_student_rattle._potential_energy(minus, 0.2, 1.3, 0.0, 1.0, 2.0)
        ) / (2.0 * step)

    np.testing.assert_allclose(analytic, numerical, rtol=2e-7, atol=2e-8)


def test_codimension_two_rattle_trajectory_is_tangent_and_reversible():
    angles = loc_scale_student.init_angles(10, 2.0)
    momentum, ok = loc_scale_student_rattle._project_momentum(
        np.arange(10, dtype=float) - 4.5, angles
    )
    assert ok

    proposed, proposed_momentum, ok, diagnostics = loc_scale_student_rattle._rattle_trajectory(
        angles, momentum, 0.2, 1.3, 0.0, 1.0, 2.0, 0.001, 2
    )
    assert ok, diagnostics
    np.testing.assert_allclose(
        loc_scale_student._score_residual_angles(proposed, 2.0), np.zeros(2), atol=1e-10
    )
    assert loc_scale_student_rattle._constraint_tangent_norm(proposed_momentum, proposed) < 1e-10

    reverse, _, reverse_ok, reverse_diagnostics = loc_scale_student_rattle._rattle_trajectory(
        proposed, -proposed_momentum, 0.2, 1.3, 0.0, 1.0, 2.0, 0.001, 2
    )
    assert reverse_ok, reverse_diagnostics
    np.testing.assert_allclose(reverse, angles, atol=1e-9)


def test_rattle_chain_preserves_scores_and_keeps_required_corrections():
    out = loc_scale_student_rattle.run_rattle(
        np.array([0, 11], dtype=np.uint32), 0.0, 1.0, _params(), verbose=False
    )
    diagnostics = out["projection_diagnostics"]

    assert diagnostics["reverse_check"] is True
    assert diagnostics["include_gram_correction"] is True
    assert diagnostics["include_mle_jacobian"] is True
    assert diagnostics["reverse_failures"] == 0
    assert out["x_acceptance_rate"] > 0.0
    for x in out["x_chain"]:
        np.testing.assert_allclose(
            loc_scale_student.score_residual(x, 0.0, 1.0, 2.0), np.zeros(2), atol=2e-9
        )


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"reverse_check": False}, "reverse_check"),
        ({"rattle_include_gram_correction": False}, "Gram correction"),
        ({"include_mle_jacobian": False}, "MLE Jacobian"),
        ({"require_local_mle": False}, "local-MLE"),
    ],
)
def test_production_rattle_rejects_disabled_correctness_guards(updates, message):
    with pytest.raises(ValueError, match=message):
        loc_scale_student_rattle.run_rattle(
            np.array([0, 1], dtype=np.uint32),
            0.0,
            1.0,
            _params(num_iterations_T=1, **updates),
            verbose=False,
        )


def test_student_location_scale_rejects_subunit_df_until_finite_mle_domain_is_handled():
    with pytest.raises(ValueError, match="k >= 1"):
        loc_scale_student.run_gibbs(
            np.array([0, 1], dtype=np.uint32), 0.0, 1.0, _params(k=0.5), verbose=False
        )
