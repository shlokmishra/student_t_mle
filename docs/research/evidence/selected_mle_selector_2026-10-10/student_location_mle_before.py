"""Certified global MLE for a Student-t location with known scale and df.

The Student score is redescending, so solving it once (or running EM from a
finite collection of starts) does not certify the global likelihood maximum.
This module isolates every sign-changing stationary root on the finite MLE
interval.  Interval enclosures for both the score and its derivative either
discard a box or prove that it contains at most one root.  Boxes that remain
ambiguous are bisected.  Comparing the likelihood at the isolated roots then
selects the global maximum, up to floating-point/root-isolation tolerance.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit


_MACHINE_EPS = np.finfo(np.float64).eps


def _validated_inputs(data, k):
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("Student location MLE requires a non-empty 1D sample")
    if not np.all(np.isfinite(x)):
        raise ValueError("Student location MLE data must be finite")
    k = float(k)
    if not math.isfinite(k) or k <= 0.0:
        raise ValueError("Student degrees of freedom k must be positive")
    return np.ascontiguousarray(x), k


@njit(cache=True)
def _score(x, mu, k):
    total = 0.0
    for value in x:
        residual = value - mu
        total += residual / (k + residual * residual)
    return total


@njit(cache=True)
def _loglik(x, mu, k):
    total = 0.0
    for value in x:
        residual = value - mu
        total += math.log1p(residual * residual / k)
    return -0.5 * (k + 1.0) * total


@njit(cache=True)
def _em_polish(x, mu, k):
    for _ in range(1000):
        weighted_sum = 0.0
        weight_sum = 0.0
        for value in x:
            residual = value - mu
            weight = 1.0 / (k + residual * residual)
            weighted_sum += value * weight
            weight_sum += weight
        updated = weighted_sum / weight_sum
        if abs(updated - mu) <= 0.25 * _MACHINE_EPS * (1.0 + abs(updated)):
            return updated
        mu = updated
    return mu


@njit(cache=True)
def _initial_mode_candidate(ordered, k):
    """Cheap incumbent for likelihood pruning; certification does not trust it."""
    num_starts = min(17, ordered.size)
    best_mu = ordered[ordered.size // 2]
    best_loglik = _loglik(ordered, best_mu, k)
    for index in range(num_starts):
        data_index = 0
        if num_starts > 1:
            data_index = int(round(index * (ordered.size - 1) / (num_starts - 1)))
        candidate = _em_polish(ordered, ordered[data_index], k)
        value = _loglik(ordered, candidate, k)
        if value > best_loglik:
            best_mu = candidate
            best_loglik = value
    return best_mu, best_loglik


@njit(cache=True)
def _loglik_upper_bound(x, left, right, k):
    total = 0.0
    for value in x:
        distance = 0.0
        if value < left:
            distance = left - value
        elif value > right:
            distance = value - right
        total += math.log1p(distance * distance / k)
    upper = -0.5 * (k + 1.0) * total
    return upper + 32.0 * _MACHINE_EPS * (1.0 + abs(upper)) * x.size


@njit(cache=True)
def _score_and_derivative_bounds(x, left, right, k):
    """Outward-rounded interval enclosures on ``[left, right]``."""
    score_lo = 0.0
    score_hi = 0.0
    derivative_lo = 0.0
    derivative_hi = 0.0
    sqrt_k = math.sqrt(k)
    sqrt_3k = math.sqrt(3.0 * k)

    for value in x:
        residual_lo = value - right
        residual_hi = value - left

        score_at_lo = residual_lo / (k + residual_lo * residual_lo)
        score_at_hi = residual_hi / (k + residual_hi * residual_hi)
        term_score_lo = min(score_at_lo, score_at_hi)
        term_score_hi = max(score_at_lo, score_at_hi)
        if residual_lo <= -sqrt_k <= residual_hi:
            term_score_lo = -0.5 / sqrt_k
        if residual_lo <= sqrt_k <= residual_hi:
            term_score_hi = 0.5 / sqrt_k
        score_lo += term_score_lo
        score_hi += term_score_hi

        denom_lo = k + residual_lo * residual_lo
        denom_hi = k + residual_hi * residual_hi
        derivative_at_lo = (residual_lo * residual_lo - k) / (denom_lo * denom_lo)
        derivative_at_hi = (residual_hi * residual_hi - k) / (denom_hi * denom_hi)
        term_derivative_lo = min(derivative_at_lo, derivative_at_hi)
        term_derivative_hi = max(derivative_at_lo, derivative_at_hi)
        if residual_lo <= 0.0 <= residual_hi:
            term_derivative_lo = -1.0 / k
        if (
            residual_lo <= -sqrt_3k <= residual_hi
            or residual_lo <= sqrt_3k <= residual_hi
        ):
            term_derivative_hi = 0.125 / k
        derivative_lo += term_derivative_lo
        derivative_hi += term_derivative_hi

    # A small outward pad covers roundoff accumulated in the sums.  It can
    # only cause extra subdivision; it cannot make us discard a valid root.
    score_pad = 16.0 * _MACHINE_EPS * (1.0 + abs(score_lo) + abs(score_hi)) * x.size
    derivative_pad = 16.0 * _MACHINE_EPS * (
        1.0 + abs(derivative_lo) + abs(derivative_hi)
    ) * x.size
    return (
        score_lo - score_pad,
        score_hi + score_pad,
        derivative_lo - derivative_pad,
        derivative_hi + derivative_pad,
    )


@njit(cache=True)
def _bisect_score_root(x, left, right, score_left, score_right, k, location_tol):
    if score_left == 0.0:
        return left
    if score_right == 0.0:
        return right
    for _ in range(200):
        middle = left + 0.5 * (right - left)
        if middle == left or middle == right:
            break
        score_middle = _score(x, middle, k)
        if score_middle == 0.0 or right - left <= location_tol * (
            1.0 + abs(left) + abs(right)
        ):
            return middle
        if (score_left < 0.0) == (score_middle < 0.0):
            left = middle
            score_left = score_middle
        else:
            right = middle
            score_right = score_middle
    return left + 0.5 * (right - left)


@njit(cache=True)
def _stationary_points_certified(x, k, location_tol, incumbent_mu, incumbent_loglik):
    ordered = np.sort(x)
    data_min = ordered[0]
    data_max = ordered[-1]
    if data_min == data_max:
        return np.array([data_min])
    if ordered.size == 2:
        center = 0.5 * (data_min + data_max)
        half_gap = 0.5 * (data_max - data_min)
        if half_gap * half_gap <= k:
            return np.array([center])
        offset = math.sqrt(half_gap * half_gap - k)
        # The center is the intervening local minimum; include it because this
        # public helper reports sign-changing stationary points.
        return np.array([center - offset, center, center + offset])
    count_min = 1
    while count_min < ordered.size and ordered[count_min] == data_min:
        count_min += 1
    count_max = 1
    while count_max < ordered.size and ordered[ordered.size - 1 - count_max] == data_max:
        count_max += 1
    if count_min + count_max == ordered.size and count_min == count_max:
        # Equal replication of a two-point sample has exactly the same score
        # roots as the unreplicated pair, including the flat transition case.
        center = 0.5 * (data_min + data_max)
        half_gap = 0.5 * (data_max - data_min)
        if half_gap * half_gap <= k:
            return np.array([center])
        offset = math.sqrt(half_gap * half_gap - k)
        return np.array([center - offset, center, center + offset])

    # Data gaps are a cheap initial partition.  The interval-bound routine
    # accounts explicitly for every per-observation extremum inside a gap, so
    # inserting x_i +/- sqrt(k) and x_i +/- sqrt(3k) is unnecessary and costs
    # roughly five times more boxes in the common, already-unimodal case.
    nodes = np.unique(ordered)

    stack = []
    for index in range(nodes.size - 1):
        stack.append((nodes[index], nodes[index + 1]))
    roots = []
    if math.isfinite(incumbent_loglik):
        roots.append(incumbent_mu)
    score_scale = max(1.0, ordered.size / math.sqrt(k))
    score_zero_tol = 64.0 * _MACHINE_EPS * score_scale

    # Score roots at partition nodes are uncommon but possible for symmetric
    # samples.  Retaining them also makes the lower-location tie rule stable.
    for node in nodes:
        if abs(_score(ordered, node, k)) <= score_zero_tol:
            roots.append(node)

    processed_intervals = 0
    while stack:
        processed_intervals += 1
        if processed_intervals > 1_000_000:
            raise RuntimeError(
                "Student global-MLE certification exceeded its interval budget"
            )
        left, right = stack.pop()
        if right <= left:
            continue
        if (
            math.isfinite(incumbent_loglik)
            and _loglik_upper_bound(ordered, left, right, k) < incumbent_loglik
        ):
            continue
        score_left = _score(ordered, left, k)
        score_right = _score(ordered, right, k)
        if abs(score_left) <= score_zero_tol:
            roots.append(left)
        if abs(score_right) <= score_zero_tol:
            roots.append(right)
        score_lo, score_hi, derivative_lo, derivative_hi = _score_and_derivative_bounds(
            ordered, left, right, k
        )

        if score_lo > 0.0 or score_hi < 0.0:
            continue

        sign_change = (score_left < 0.0 < score_right) or (score_right < 0.0 < score_left)
        derivative_has_fixed_sign = derivative_lo > 0.0 or derivative_hi < 0.0
        interval_small = right - left <= location_tol * (1.0 + abs(left) + abs(right))
        if sign_change and (derivative_has_fixed_sign or interval_small):
            root = _bisect_score_root(
                ordered, left, right, score_left, score_right, k, location_tol
            )
            roots.append(root)
            root_loglik = _loglik(ordered, root, k)
            if root_loglik > incumbent_loglik:
                incumbent_mu = root
                incumbent_loglik = root_loglik
            continue
        if derivative_has_fixed_sign:
            continue
        if interval_small:
            # A non-sign-changing stationary point cannot be a strict maximum.
            # It is harmless for global selection, while a sign-changing root
            # was handled above.
            continue

        middle = left + 0.5 * (right - left)
        if middle == left or middle == right:
            continue
        stack.append((left, middle))
        stack.append((middle, right))

    if not roots:
        # For non-constant finite data the score is positive at the sample
        # minimum and negative at the maximum, so this signals a numerical bug.
        raise RuntimeError("Student global-MLE root isolation found no maximum candidate")

    candidates = np.sort(np.asarray(roots))
    distinct = [candidates[0]]
    for candidate in candidates[1:]:
        local_scale = 1.0 + abs(candidate) + abs(distinct[-1])
        if candidate - distinct[-1] > 8.0 * location_tol * local_scale:
            distinct.append(candidate)
    return np.asarray(distinct)


def stationary_points(data, k, *, location_tol=2e-13):
    """Return all isolated sign-changing stationary points in the MLE interval."""
    x, k = _validated_inputs(data, k)
    location_tol = float(location_tol)
    if not math.isfinite(location_tol) or location_tol <= 0.0:
        raise ValueError("location_tol must be positive and finite")
    center = float(np.median(x))
    scale = math.sqrt(k)
    standardized = np.ascontiguousarray((x - center) / scale)
    roots = _stationary_points_certified(
        standardized, 1.0, location_tol, 0.0, -math.inf
    )
    return center + scale * roots


def global_mle(data, k, *, location_tol=2e-13):
    """Return the selected global Student location MLE.

    All strict local maxima are isolated before likelihood comparison.  Exact
    or floating-point-indistinguishable likelihood ties use the lower-location
    convention.
    """
    x, k = _validated_inputs(data, k)
    center = float(np.median(x))
    scale = math.sqrt(k)
    standardized = np.ascontiguousarray((x - center) / scale)
    incumbent_mu, incumbent_loglik = _initial_mode_candidate(np.sort(standardized), 1.0)
    candidates = _stationary_points_certified(
        standardized, 1.0, float(location_tol), incumbent_mu, incumbent_loglik
    )
    values = np.asarray([_loglik(standardized, candidate, 1.0) for candidate in candidates])
    best = float(np.max(values))
    tie_tol = 64.0 * _MACHINE_EPS * max(1.0, abs(best), x.size)
    return float(center + scale * np.min(candidates[values >= best - tie_tol]))


def loglik(data, mu, k):
    """Student location log likelihood, omitting constants in ``mu``."""
    x, k = _validated_inputs(data, k)
    return float(_loglik(x, float(mu), k))


def score(data, mu, k):
    """Student location score without the positive ``k + 1`` multiplier."""
    x, k = _validated_inputs(data, k)
    return float(_score(x, float(mu), k))
