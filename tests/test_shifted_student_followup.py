"""Research harness consistency and uncertainty regressions; no kernel changes."""
import numpy as np

from models.loc import affine as loc_affine
from scripts.followup_shifted_student_reference import (
    run_affine_chain, event_evidence, diagnostics, autocorrelation_ess,
    particle_bootstrap,
)


CASE = dict(k=2., n=10, release=1.25, prior_mean=-.75, prior_std=.6,
            tau_q=.8, proposal_std_mu=.45, mle_tolerance=1e-8)


def test_dispersed_harness_matches_production_from_same_start():
    arrays, meta = run_affine_chain(np.full(10, 1.25), -.75, 8192, CASE, 20)
    result = loc_affine.run_gibbs(8192, 1.25, {
        **CASE, 'model': 'student', 'num_iterations_T': 20,
        'initial_mu': -.75, 'store_x_chain': True}, verbose=False)
    np.testing.assert_array_equal(arrays['mu'], result['mu_chain'])
    np.testing.assert_array_equal(arrays['x'], result['x_chain'])
    assert meta['max_retained_mle_residual'] < 1e-10


def test_dispersed_valid_outlier_start_remains_selected_mle():
    e = np.array([-12.]+[0.]*9)
    x = 1.25+e-loc_affine.global_location_mle(e, 'student', k=2.)
    arrays, meta = run_affine_chain(x, -.75, 8193, CASE, 20)
    assert meta['initial_max_residual'] > 10
    for state in arrays['x']:
        assert abs(loc_affine.global_location_mle(state, 'student', k=2.)-1.25) < 1e-10


def test_zero_independent_events_have_nonzero_upper_uncertainty():
    result = event_evidence(np.zeros(1000, dtype=bool))
    assert result['events'] == 0
    assert .003 < result['interval95'][1] < .004
    assert event_evidence([])['interval95'] is None
    result = diagnostics(np.zeros((4, 100)))
    assert result['ess_bulk'] is None and result['rhat'] is None


def test_zero_event_particle_bootstrap_reports_unavailable_uncertainty():
    case = {**CASE, 'tail_width': .75}
    result = particle_bootstrap(np.zeros(100), np.ones(100), np.full((100, 10), 1.25),
                                case, seed=773, repetitions=10)
    assert result['upper_tail']['se'] is None
    assert result['upper_tail']['percentile95'] is None


def test_ess_detects_correlated_and_disagreeing_chains():
    rng = np.random.default_rng(771)
    iid = rng.normal(size=(4, 4000))
    repeated = np.repeat(rng.normal(size=(4, 200)), 20, axis=1)
    assert autocorrelation_ess(iid) > 10000
    assert 100 < autocorrelation_ess(repeated) < 1200
    shifted = iid+np.array([0., 0., 4., 4.])[:, None]
    assert diagnostics(shifted)['rhat'] > 1.5
    assert autocorrelation_ess(shifted) < 20


def test_binary_diagnostics_remain_defined_with_one_constant_chain():
    rng = np.random.default_rng(772)
    a = (rng.random((4, 1000)) < .3).astype(float)
    a[0] = 1.
    result = diagnostics(a)
    assert result['rhat'] > 1.2
    assert result['ess_raw'] is not None
