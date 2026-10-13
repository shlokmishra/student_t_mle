import jax.random as random
import numpy as np

from models.loc.student import score_gibbs as loc_student
from reporting.diagnostics.audit_student_global_mle import (
    global_mle_dense,
    global_mle_multistart,
    legacy_wide_brent_score_root,
    loglik,
    simulation_key,
)


def test_global_selector_rejects_a_non_global_wide_brent_root():
    data_sets = np.asarray(random.t(simulation_key(0), df=1.0, shape=(100, 50)), dtype=float)
    data = data_sets[57]
    brent_mu = legacy_wide_brent_score_root(data, 1.0)
    global_mu = loc_student.get_mle(data, {"k": 1.0, "n": 50})
    global_ll = float(loglik(data, global_mu, 1.0))
    score_residual = abs(float(np.sum((data - global_mu) / (1.0 + (data - global_mu) ** 2))))
    dense_mu, dense_ll = global_mle_dense(data, 1.0, 25)

    assert global_ll - float(loglik(data, brent_mu, 1.0)) > 300.0
    assert abs(global_ll - dense_ll) < 1e-8
    assert abs(global_mu - dense_mu) < 1e-6
    assert score_residual < 1e-8


def test_global_selector_matches_wide_brent_on_a_regular_dataset():
    data = np.array([-1.2, -0.7, -0.2, 0.0, 0.1, 0.4, 0.8, 1.1])
    brent_mu = legacy_wide_brent_score_root(data, 2.0)
    global_mu = loc_student.get_mle(data, {"k": 2.0, "n": data.size})
    global_ll = float(loglik(data, global_mu, 2.0))
    _, multistart_ll, score_residual, _ = global_mle_multistart(data, 2.0)

    assert abs(global_ll - float(loglik(data, brent_mu, 2.0))) < 1e-10
    assert abs(global_ll - multistart_ll) < 1e-10
    assert abs(global_mu - brent_mu) < 1e-6
    assert score_residual < 1e-8


def test_global_selector_uses_lower_location_for_symmetric_tie():
    data = np.array([-10.0, -10.0, 10.0, 10.0])
    selected = loc_student.get_mle(data, {"k": 1.0, "n": data.size})

    assert selected < -9.0
    np.testing.assert_allclose(
        loglik(data, selected, 1.0),
        loglik(data, -selected, 1.0),
        rtol=0.0,
        atol=1e-12,
    )


def test_global_selector_matches_dense_search_on_adversarial_small_samples():
    rng = np.random.default_rng(20260820)
    for k in (1.0, 2.0, 3.0):
        for _ in range(40):
            data = np.concatenate(
                [rng.normal(-8.0, 0.15, 3), rng.normal(2.0, 0.4, 4), rng.standard_t(k, 3)]
            )
            selected = loc_student.get_mle(data, {"k": k, "n": data.size})
            _, dense_ll = global_mle_dense(data, k, 61)
            assert dense_ll - float(loglik(data, selected, k)) < 1e-8
