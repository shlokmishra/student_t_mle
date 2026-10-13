import numpy as np

from models.loc import affine as loc_affine


def _params(model, **overrides):
    return {
        "model": model,
        "n": 5,
        "num_iterations_T": 4,
        "tau_q": 1.0,
        "proposal_std_mu": 0.2,
        "prior_mean": 0.0,
        "prior_std": 10.0,
        **overrides,
    }


def test_centered_sum_squares_is_translation_invariant():
    z = np.array([-3.0, 0.5, 2.0, 8.0])
    assert loc_affine.centered_sum_squares(z) == loc_affine.centered_sum_squares(z + 17.0)
    assert not np.isclose(np.mean(z), 0.0)


def test_student_global_mle_selects_a_mode_instead_of_the_central_score_root():
    x = np.array([-10.0, -10.0, 10.0, 10.0])
    roots = loc_affine.student_location_stationary_points(x, k=1.0)
    assert roots.size == 3
    assert abs(roots[1]) < 1e-10
    selected = loc_affine.student_global_location_mle(x, k=1.0)
    assert selected < -9.0  # deterministic lower-location tie rule
    assert loc_affine.student_location_loglik(selected, x, 1.0) > loc_affine.student_location_loglik(0.0, x, 1.0)


def test_student_global_mle_is_translation_equivariant():
    x = np.array([-4.0, -1.5, -0.25, 2.0, 8.0])
    original = loc_affine.global_location_mle(x, "student", k=2.0)
    shifted = loc_affine.global_location_mle(x + 31.25, "student", k=2.0)
    np.testing.assert_allclose(shifted, original + 31.25, atol=1e-10, rtol=0.0)


def test_student_normalization_remains_numerically_on_manifold_at_n10():
    rng = np.random.default_rng(20260819)
    for _ in range(20):
        u = rng.normal(size=10)
        normalized = loc_affine.normalized_affine_proposal(u, 0.0, "student", k=2.0)
        assert abs(loc_affine.global_location_mle(normalized, "student", k=2.0)) < 1e-10


def test_student_selected_mle_matches_dense_global_search_for_each_requested_df():
    rng = np.random.default_rng(91)
    x = np.concatenate([rng.normal(-3.0, 0.35, 4), rng.normal(2.0, 0.6, 5)])
    grid = np.linspace(float(np.min(x)), float(np.max(x)), 20001)
    for k in (1.0, 2.0, 3.0):
        selected = loc_affine.student_global_location_mle(x, k)
        selected_loglik = loc_affine.student_location_loglik(selected, x, k)
        grid_loglik = np.array([loc_affine.student_location_loglik(mu, x, k) for mu in grid])
        assert selected_loglik >= float(np.max(grid_loglik)) - 1e-7


def test_normal_affine_proposal_is_centered_at_current_state():
    x = np.array([-1.0, 0.0, 1.0]) + 0.7
    expected_rng = np.random.default_rng(123)
    noise = expected_rng.normal(0.0, 0.4, size=x.size)
    accept_uniform = expected_rng.uniform()
    proposed = loc_affine.normalized_affine_proposal(x + noise, 0.7, "normal", scale=1.3)
    expected_log_ratio = loc_affine.location_loglik(proposed, -0.4, "normal", scale=1.3) - loc_affine.location_loglik(x, -0.4, "normal", scale=1.3)

    state, accepted, diagnostics = loc_affine.latent_affine_mh_step(
        x,
        mu_current=-0.4,
        mu_star=0.7,
        model="normal",
        tau_q=0.4,
        scale=1.3,
        rng=np.random.default_rng(123),
        return_diagnostics=True,
    )
    assert accepted == (np.log(accept_uniform) < min(0.0, expected_log_ratio))
    np.testing.assert_allclose(state, proposed if accepted else x)
    np.testing.assert_allclose(diagnostics["log_acceptance_ratio"], expected_log_ratio)
    assert abs(np.mean(state) - 0.7) < 1e-12


def test_local_affine_log_ratio_is_likelihood_difference_only():
    x = np.array([-1.0, 0.0, 1.0])
    proposed = np.array([-2.0, 0.0, 2.0])
    expected = (
        loc_affine.location_loglik(proposed, 0.3, "student", k=3.0)
        - loc_affine.location_loglik(x, 0.3, "student", k=3.0)
    )
    actual = loc_affine.affine_log_acceptance_ratio(x, proposed, 0.3, "student", k=3.0)
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0.0)


def test_local_gaussian_orbit_factor_is_symmetric():
    x = np.array([-2.0, 0.25, 0.75, 4.0])
    proposed = np.array([-1.0, -0.5, 1.5, 3.0])
    assert loc_affine.centered_sum_squares(proposed - x) == loc_affine.centered_sum_squares(x - proposed)


def test_normal_affine_gibbs_chain_stays_on_selected_mle_manifold():
    out = loc_affine.run_gibbs(7, 0.4, _params("normal", known_scale=1.0), verbose=False)
    assert out["mu_chain"].shape == (5,)
    assert out["x_chain"].shape == (5, 5)
    assert 0.0 <= out["latent_acceptance_rate"] <= 1.0
    assert out["mle_definition"] == "selected_global_location_mle"
    assert out["max_mle_residual"] < 1e-12
    np.testing.assert_allclose(np.mean(out["x_chain"], axis=1), 0.4, atol=1e-12, rtol=0.0)


def test_student_affine_gibbs_smoke_uses_selected_global_mle():
    out = loc_affine.run_gibbs(
        11,
        -0.25,
        _params("student", n=4, k=3.0, num_iterations_T=3),
        verbose=False,
    )
    assert out["mu_chain"].shape == (4,)
    assert 0.0 <= out["latent_acceptance_rate"] <= 1.0
    assert 0.0 <= out["mu_acceptance_rate"] <= 1.0
    for state in out["x_chain"]:
        selected = loc_affine.global_location_mle(state, "student", k=3.0)
        assert abs(selected + 0.25) < 1e-8


def test_logistic_affine_gibbs_smoke_stays_on_unique_mle_manifold():
    out = loc_affine.run_gibbs(
        13,
        0.35,
        _params("logistic", num_iterations_T=3),
        verbose=False,
    )
    for state in out["x_chain"]:
        selected = loc_affine.global_location_mle(state, "logistic")
        assert abs(selected - 0.35) < 1e-9


def test_affine_reference_samples_use_the_same_selected_mle():
    params = _params("student", n=4, k=2.0)
    samples = loc_affine.get_benchmark_mle_samples(19, params, num_simulations=6)
    assert samples.shape == (6,)
    assert np.all(np.isfinite(samples))
