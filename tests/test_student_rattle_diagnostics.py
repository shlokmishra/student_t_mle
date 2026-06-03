import numpy as np
import jax.random as random

from models import loc_student_rattle
from reporting.diagnostics.student_t_geometry import (
    constraint_value,
    grad_constraint,
    gram,
    potential_with_gram,
    potential_without_gram,
    psi,
    psi_prime,
)


def test_psi_and_derivative_known_values():
    y = np.array([0.0, 1.0, -1.0])
    k = 2.0
    np.testing.assert_allclose(psi(y, k), np.array([0.0, 1.0 / 3.0, -1.0 / 3.0]))
    np.testing.assert_allclose(psi_prime(y, k), np.array([0.5, 1.0 / 9.0, 1.0 / 9.0]))


def test_constraint_gram_matches_rattle_helpers():
    x = np.array([1.5, 2.0, 2.5, 3.0])
    mu_star = 2.0
    k = 2.0
    np.testing.assert_allclose(constraint_value(x, mu_star, k), loc_student_rattle._constraint_value(x, mu_star, k))
    np.testing.assert_allclose(grad_constraint(x, mu_star, k), loc_student_rattle._constraint_grad(x, mu_star, k))
    np.testing.assert_allclose(gram(x, mu_star, k), loc_student_rattle._constraint_gram(x, mu_star, k))


def test_gram_corrected_potential_is_finite_and_flagged():
    x = np.array([1.8, 2.2, 1.5, 2.5])
    mu_star = 2.0
    mu = 2.1
    k = 2.0
    u0 = potential_without_gram(x, mu, k)
    u1 = potential_with_gram(x, mu, mu_star, k)
    assert np.isfinite(u0)
    assert np.isfinite(u1)
    np.testing.assert_allclose(
        loc_student_rattle._potential_energy(x, mu, k, mu_star=mu_star, include_gram_correction=False),
        u0,
    )
    np.testing.assert_allclose(
        loc_student_rattle._potential_energy(x, mu, k, mu_star=mu_star, include_gram_correction=True),
        u1,
    )


def _tangent_momentum(x, p, mu_star, k):
    grad = loc_student_rattle._constraint_grad(x, mu_star, k)
    denom = float(np.dot(grad, grad))
    return p - grad * float(np.dot(grad, p)) / denom


def test_paper_fixed_direction_step_preserves_constraint_and_tangent_momentum():
    k = 2.0
    mu_star = 0.0
    mu = 0.1
    x0 = np.array([-0.5, 0.5, -1.0, 1.0])
    p0 = _tangent_momentum(x0, np.array([0.2, -0.1, 0.05, -0.03]), mu_star, k)
    x1, p1, ok, _ = loc_student_rattle._rattle_trajectory(
        x0,
        p0,
        mu,
        mu_star,
        k,
        step_size=0.005,
        num_steps=1,
        proj_tol=1e-12,
        proj_max_iters=25,
        grad_tol=1e-12,
        proj_damping=1.0,
        proj_line_search=True,
        proj_init_strategy="trial",
        include_gram_correction=True,
        projection_mode="paper_fixed_direction",
        tangent_tol=1e-9,
    )
    assert ok
    assert abs(loc_student_rattle._constraint_value(x1, mu_star, k)) < 1e-10
    grad1 = loc_student_rattle._constraint_grad(x1, mu_star, k)
    assert abs(float(np.dot(grad1, p1))) < 1e-9


def test_paper_fixed_direction_reverse_position_check_passes_for_small_step():
    k = 2.0
    mu_star = 0.0
    mu = 0.1
    x0 = np.array([-0.5, 0.5, -1.0, 1.0])
    p0 = _tangent_momentum(x0, np.array([0.2, -0.1, 0.05, -0.03]), mu_star, k)
    kwargs = dict(
        mu=mu,
        mu_star=mu_star,
        k=k,
        step_size=0.005,
        num_steps=1,
        proj_tol=1e-12,
        proj_max_iters=25,
        grad_tol=1e-12,
        proj_damping=1.0,
        proj_line_search=True,
        proj_init_strategy="trial",
        include_gram_correction=True,
        projection_mode="paper_fixed_direction",
        tangent_tol=1e-9,
    )
    x1, p1, ok, _ = loc_student_rattle._rattle_trajectory(x0, p0, **kwargs)
    assert ok
    x_rev, _, ok_rev, _ = loc_student_rattle._rattle_trajectory(x1, -p1, **kwargs)
    assert ok_rev
    assert np.linalg.norm(x_rev - x0) < 1e-6


def test_default_rattle_target_uses_gram_correction_and_paper_projection():
    params = {
        "k": 2.0,
        "n": 4,
        "num_iterations_T": 2,
        "proposal_std_mu": 0.1,
        "prior_mean": 0.0,
        "prior_std": 10.0,
        "rattle_step_size": 0.005,
        "rattle_num_steps": 1,
    }
    out = loc_student_rattle.run_rattle(random.PRNGKey(0), 0.0, params, verbose=False)
    diag = out["projection_diagnostics"]
    assert diag["include_gram_correction"] is True
    assert diag["projection_mode"] == "paper_fixed_direction"


def test_legacy_projection_mode_still_runs():
    params = {
        "k": 2.0,
        "n": 4,
        "num_iterations_T": 2,
        "proposal_std_mu": 0.1,
        "prior_mean": 0.0,
        "prior_std": 10.0,
        "rattle_step_size": 0.005,
        "rattle_num_steps": 1,
        "rattle_include_gram_correction": False,
        "rattle_projection_mode": "normal_newton_legacy",
    }
    out = loc_student_rattle.run_rattle(random.PRNGKey(1), 0.0, params, verbose=False)
    assert np.asarray(out["mu_chain"]).shape[0] == 3
    assert out["projection_diagnostics"]["projection_mode"] == "normal_newton_legacy"
