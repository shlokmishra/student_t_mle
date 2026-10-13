import math

import numpy as np
import pytest
from numba import njit

from models.loc.student import score_gibbs as loc_student

_sample_inverse_branch = loc_student.sample_inverse_branch


def _branch_probabilities(z, k, loc=0.0):
    y_lo, y_hi = loc_student.psi_inverse(float(z), float(k))
    log_weights = np.array(
        [
            loc_student.inverse_branch_log_weight(y_lo, loc, k),
            loc_student.inverse_branch_log_weight(y_hi, loc, k),
        ]
    )
    weights = np.exp(log_weights - np.max(log_weights))
    probabilities = weights / np.sum(weights)
    outer_index = int(abs(y_hi) > abs(y_lo))
    return (y_lo, y_hi), probabilities, float(probabilities[outer_index])


def _old_outer_probability(z, k, loc=0.0):
    branches = loc_student.psi_inverse(float(z), float(k))
    logs = np.array(
        [loc_student.student_logpdf_no_const(y, loc, k) for y in branches]
    )
    probabilities = np.exp(logs - np.max(logs))
    probabilities /= np.sum(probabilities)
    outer_index = int(abs(branches[1]) > abs(branches[0]))
    return float(probabilities[outer_index])


@njit(cache=True)
def _empirical_outer_frequency(y_lo, y_hi, loc, k, uniforms):
    count = 0
    threshold = math.sqrt(k)
    for value in uniforms:
        draw = _sample_inverse_branch(y_lo, y_hi, loc, k, value)
        count += abs(draw) > threshold
    return count / uniforms.size


@pytest.mark.parametrize("k", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("support_fraction", [0.05, 0.60], ids=["near_zero", "central"])
def test_empirical_inverse_branch_frequency_matches_jacobian_target(k, support_fraction):
    z = support_fraction / (2.0 * math.sqrt(k))
    branches, _, expected_outer = _branch_probabilities(z, k)
    uniforms = np.random.default_rng(194 + int(k * 10 + support_fraction * 100)).random(200_000)
    empirical = _empirical_outer_frequency(branches[0], branches[1], 0.0, k, uniforms)
    standard_error = math.sqrt(expected_outer * (1.0 - expected_outer) / uniforms.size)
    assert abs(empirical - expected_outer) <= 6.0 * standard_error + 2e-5


@pytest.mark.parametrize("k", [1.0, 2.0, 3.0])
def test_all_inverse_branch_pairs_preserve_pair_score(k):
    z_max = 1.0 / (2.0 * math.sqrt(k))
    z_i = 0.17 * z_max
    z_j = -0.31 * z_max
    delta = z_i + z_j
    branches_i = loc_student.psi_inverse(z_i, k)
    branches_j = loc_student.psi_inverse(z_j, k)
    for y_i in branches_i:
        for y_j in branches_j:
            reconstructed = loc_student.psi(y_i, k) + loc_student.psi(y_j, k)
            np.testing.assert_allclose(reconstructed, delta, rtol=0.0, atol=2e-15)


@pytest.mark.parametrize(
    "k,support_fraction,expected_old,expected_correct",
    [
        (1.0, 0.02, 0.0001000100020005, 0.5),
        (1.0, 0.50, 0.0669872981077807, 0.5),
        (2.0, 0.10, 0.0001259287263393, 0.0477329831333545),
        (2.0, 0.50, 0.0188747756753119, 0.2113248654051872),
        (3.0, 0.10, 0.0000063130117488, 0.0025062814466900),
        (3.0, 0.50, 0.0051283406946065, 0.0669872981077807),
    ],
)
def test_density_only_and_corrected_outer_probabilities_are_distinct(
    k, support_fraction, expected_old, expected_correct
):
    z = support_fraction / (2.0 * math.sqrt(k))
    old = _old_outer_probability(z, k)
    _, _, corrected = _branch_probabilities(z, k)
    np.testing.assert_allclose(old, expected_old, rtol=2e-11, atol=1e-14)
    np.testing.assert_allclose(corrected, expected_correct, rtol=2e-11, atol=1e-14)
    assert corrected > old


@pytest.mark.parametrize("k", [1.0, 2.0, 3.0])
def test_stable_inverse_recovers_very_small_nonzero_score(k):
    z = 1e-10 / math.sqrt(k)
    y_lo, y_hi = loc_student.psi_inverse(z, k)
    assert abs(y_hi) > math.sqrt(k)
    np.testing.assert_allclose(loc_student.psi(y_lo, k), z, rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(loc_student.psi(y_hi, k), z, rtol=1e-14, atol=0.0)
