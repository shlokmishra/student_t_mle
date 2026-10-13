"""Small mathematical controls for the isolated selected-MLE reference oracle."""
import numpy as np
import pytest

pytest.importorskip("sympy")

from scripts.selected_mle_reference_oracle import selected_mle_oracle
from scripts.validate_selected_mle_reference import weight_particles


def test_exact_oracle_enumerates_two_point_roots_and_selects_lower_tie():
    result = selected_mle_oracle([-4.] * 5 + [4.] * 5, 2.)
    np.testing.assert_allclose(result['roots'], [-np.sqrt(14), 0, np.sqrt(14)], atol=1e-12)
    assert result['selection_status'] == 'exact_symmetric_global_tie_lower'
    assert result['selected'] < 0
    accelerated = selected_mle_oracle([-4.] * 5 + [4.] * 5, 2., fast=True)
    np.testing.assert_allclose(accelerated['roots'], [-np.sqrt(14), 0, np.sqrt(14)], atol=1e-12)
    assert accelerated['selection_status'] == 'exact_symmetric_global_tie_lower'
    assert accelerated['selected'] < 0


def test_exact_oracle_includes_degenerate_stationary_root():
    result = selected_mle_oracle([-1., 1.], 1.)
    assert result['roots'] == [0.]
    assert result['multiplicities'] == [3]
    assert result['selected'] == 0.


def test_oracle_translation_and_permutation():
    x = np.array([-10., -10., 10., 10., 10.])
    original = selected_mle_oracle(x, 1.)
    shifted = selected_mle_oracle(x[::-1] + 32., 1.)
    assert len(original['roots']) == 3
    np.testing.assert_allclose(shifted['selected'], original['selected'] + 32., atol=1e-12)


def test_reference_uses_release_minus_mle_and_prior_at_reconstructed_mu():
    base = np.array([[-2., 0.], [0., 2.], [1., 3.]])
    mle = base.mean(axis=1)
    case = dict(release=1.25, prior_mean=-.75, prior_std=.6)
    mu, x, w, z = weight_particles(base, mle, case)
    np.testing.assert_allclose(mu, 1.25-mle)
    np.testing.assert_allclose(x.mean(axis=1), 1.25)
    log_prior = -.5*((mu+.75)/.6)**2
    np.testing.assert_allclose(w/w.sum(), np.exp(log_prior)/np.exp(log_prior).sum())
    assert z > 0
    with pytest.raises(ValueError):
        weight_particles(base, np.array([np.nan, 1., 2.]), case)
