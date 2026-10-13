"""Regressions for selected global Student location MLE correctness."""
import numpy as np
import pytest

from models.loc.student.mle import StudentMLEError, global_mle, stationary_points, score, loglik


def oracle(x, k):
    from scripts.selected_mle_reference_oracle import selected_mle_oracle
    return selected_mle_oracle(x, k, fast=True)


@pytest.mark.parametrize('power', [10, 20, 35, 45])
def test_near_tie_selects_unique_global_winner(power):
    x = np.array([-4.] * 5 + [4.] * 5)
    x[0] += 2.**(-power)
    expected = oracle(x, 2.)
    result, info = global_mle(x, 2., return_diagnostics=True)
    assert result > 0
    assert abs(result-expected['selected']) < 2e-13
    assert expected['loglik']-loglik(x, result, 2.) < 1e-12
    assert abs(score(x, result, 2.)) < 1e-12
    assert not info['exact_tie']


@pytest.mark.parametrize('x,k', [([-4.]*5+[4.]*5, 2.), ([-10.,0.,10.],1.)])
def test_exact_symmetry_uses_lower_global_maximum(x, k):
    result, info = global_mle(x, k, return_diagnostics=True)
    assert result == pytest.approx(oracle(x, k)['selected'], abs=2e-13)
    assert result <= 0
    if len(x) == 10:
        assert info['exact_tie']


@pytest.mark.parametrize('k', [1., 2., 3.])
@pytest.mark.parametrize('x', [[-10.,-10.,10.,10.,10.], [-10.,0.,10.], [0.,20.,21.]])
def test_stationary_enumerator_retains_inferior_roots(x, k):
    expected = oracle(x, k)
    actual = stationary_points(x, k)
    assert len(actual) == len(expected['roots'])
    np.testing.assert_allclose(actual, expected['roots'], rtol=0, atol=2e-13)
    assert max(abs(score(x, root, k)) for root in actual) < 2e-12


def test_nearly_flat_two_point_maximum_uses_original_binary_data():
    x = [-float(np.sqrt(2)), float(np.sqrt(2))]
    expected = oracle(x, 2.)
    result, info = global_mle(x, 2., return_diagnostics=True)
    assert result < 0
    assert abs(result-expected['selected']) < 2e-13
    assert info['absolute_error_bound'] <= 2e-13
    assert len(stationary_points(x, 2.)) == 3
    # Direct float summation has a roundoff floor; root accuracy is checked above.
    assert abs(score(x, result, 2.)) < 1e-15


def test_exact_degenerate_maximum_is_retained():
    assert global_mle([-1.,1.], 1.) == 0
    np.testing.assert_array_equal(stationary_points([-1.,1.], 1.), [0.])


def test_extreme_tied_pair_is_finite_and_diagnostics_do_not_overflow():
    x = [-1e200,1e200]
    assert global_mle(x, 2.) == -1e200
    assert np.isfinite(loglik(x, -1e200, 2.))
    assert np.isfinite(score(x, -1e200, 2.))


def test_unsupported_extreme_range_is_explicit():
    with pytest.raises(StudentMLEError, match='dynamic_range'):
        global_mle([0.,1.,1e200], 2.)


def test_large_offset_requires_representable_requested_accuracy():
    x = np.array([0.,.25,1.5]) + 2.**40
    with pytest.raises(StudentMLEError, match='output_precision'):
        global_mle(x, 2.)
    result, info = global_mle(x, 2., location_tol=1e-4, return_diagnostics=True)
    assert result == oracle(x, 2.)['selected']
    assert info['absolute_error_bound'] <= 1e-4


def test_pair_selection_only_checks_requested_winner_accuracy():
    result = global_mle([0.,2.**40],1.)
    assert abs(result-2.**-40) < 1e-18
    with pytest.raises(StudentMLEError, match='output_precision'):
        stationary_points([0.,2.**40],1.)


def test_centering_does_not_erase_small_winning_cluster_separation():
    x = [0.,1e-4,1e14,2e14,3e14]
    result, info = global_mle(x,1.,return_diagnostics=True)
    assert abs(result-5e-5) < 2e-13
    assert info['fallback']
    assert abs(result-oracle(x,1.)['selected']) < 2e-13


def test_centering_cannot_change_global_winner_despite_small_score():
    c, a = 1e14, .0078
    x = [-a,-a,a,a,c,c,c,c+.015625]
    # The true winning root is near c+1/256 and cannot meet default float accuracy.
    with pytest.raises(StudentMLEError, match='output_precision'):
        global_mle(x,1.)
    result, info = global_mle(x,1.,location_tol=.01,return_diagnostics=True)
    assert result == c
    assert info['fallback'] and not info['exact_tie']


@pytest.mark.parametrize('x,k', [([-10.,-10.,10.,10.,10.],1.), ([-4.]*5+[4.]*5,2.), ([-1.,.1,.5,2.],3.)])
def test_translation_and_permutation(x, k):
    x = np.array(x); result = global_mle(x, k)
    assert global_mle(x[::-1], k) == result
    assert abs(global_mle(x+32., k)-result-32.) < 2e-13
    np.testing.assert_allclose(stationary_points(x+32.,k), stationary_points(x,k)+32., rtol=0, atol=2e-13)


@pytest.mark.parametrize('function', [global_mle, stationary_points])
@pytest.mark.parametrize('tolerance', [0., -1., np.nan, np.inf])
def test_invalid_tolerance_is_rejected(function, tolerance):
    with pytest.raises(ValueError, match='location_tol'):
        function([0.,1.,2.],2.,location_tol=tolerance)


@pytest.mark.parametrize('data,k', [([],2.), ([[0.,1.]],2.), ([0.,np.inf],2.), ([0.,np.nan],2.), ([0.,1.],0.), ([0.,1.],np.inf)])
def test_invalid_inputs_are_rejected(data, k):
    with pytest.raises(ValueError):
        global_mle(data,k)


def test_accuracy_below_float_resolution_is_explicit():
    with pytest.raises(StudentMLEError, match='output_precision'):
        global_mle([-4.,4.],2.,location_tol=1e-30)


def test_ambiguous_large_sample_reports_exact_fallback_limit():
    with pytest.raises(StudentMLEError, match='exact_size_limit'):
        global_mle([-10.]*10+[10.]*10+[0.],1.)
