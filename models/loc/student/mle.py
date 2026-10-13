"""Student location MLE with selective exact-arithmetic safeguards.

The ordinary path uses deterministic EM incumbents and floating-point interval
search. Ambiguous selection/flat roots use rational polynomial bounds. This is
not universal floating-point certification; unsupported or unresolved cases
raise StudentMLEError instead of using a numerical likelihood tie convention.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from numba import njit


_MACHINE_EPS = np.finfo(np.float64).eps


def _validated_inputs(data, k):
    if np.iscomplexobj(data):
        raise ValueError("Student location MLE data must be real")
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

    # Conservative roundoff padding for the ordinary floating-point path.
    # This heuristic is not a formally directed-rounded interval calculation.
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
        if score_middle == 0.0 or right - left <= location_tol:
            return middle
        if (score_left < 0.0) == (score_middle < 0.0):
            left = middle
            score_left = score_middle
        else:
            right = middle
            score_right = score_middle
    return left + 0.5 * (right - left)


@njit(cache=True)
def _stationary_points_certified(x, k, location_tol, incumbent_mu, incumbent_loglik,
                                 prune=True, comparison_margin=0.0):
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
    if prune and math.isfinite(incumbent_loglik):
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
        if processed_intervals > 100_000:
            raise RuntimeError(
                "Student stationary search exceeded its 100000-interval budget"
            )
        left, right = stack.pop()
        if right <= left:
            continue
        if (
            prune and math.isfinite(incumbent_loglik)
            and _loglik_upper_bound(ordered, left, right, k)
            < incumbent_loglik - max(comparison_margin, 128.0 * _MACHINE_EPS
                                    * max(1.0, abs(incumbent_loglik), ordered.size))
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
        interval_small = right - left <= location_tol
        if sign_change and derivative_has_fixed_sign:
            root = _bisect_score_root(
                ordered, left, right, score_left, score_right, k, location_tol
            )
            roots.append(root)
            root_loglik = _loglik(ordered, root, k)
            if prune and root_loglik > incumbent_loglik:
                incumbent_mu = root
                incumbent_loglik = root_loglik
            continue
        if derivative_has_fixed_sign:
            continue
        if interval_small:
            raise RuntimeError("unresolved stationary interval (flat or multiple root)")

        middle = left + 0.5 * (right - left)
        if middle == left or middle == right:
            raise RuntimeError("stationary interval cannot be subdivided in float arithmetic")
        stack.append((left, middle))
        stack.append((middle, right))

    if not roots:
        # For non-constant finite data the score is positive at the sample
        # minimum and negative at the maximum, so this signals a numerical bug.
        raise RuntimeError("Student global-MLE root isolation found no maximum candidate")

    candidates = np.sort(np.asarray(roots))
    distinct = [candidates[0]]
    for candidate in candidates[1:]:
        if candidate - distinct[-1] > 8.0 * location_tol:
            distinct.append(candidate)
    return np.asarray(distinct)


class StudentMLEError(RuntimeError):
    """Selection or requested absolute location accuracy could not be resolved."""

    def __init__(self, reason, message):
        self.reason = reason
        super().__init__(f"Student location MLE [{reason}]: {message}")


def _validated_tolerance(location_tol):
    tolerance = float(location_tol)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("location_tol must be positive and finite")
    return tolerance


def _rounded_interval(left, right, tolerance):
    """Round a rational enclosure only when its error fits the requested units."""
    result = float((left + right) / 2)
    if not math.isfinite(result):
        raise StudentMLEError("output_precision", "the selected location is not finite")
    rounded = Fraction(result)
    bound = max(abs(rounded-left), abs(rounded-right))
    if bound > Fraction(tolerance):
        raise StudentMLEError(
            "output_precision",
            f"a float cannot represent this root within absolute location_tol={tolerance:g}",
        )
    return result, float(bound)


def _two_point_roots(x, k, tolerance, *, select=False):
    """Exact input arithmetic for pairs and equal replication of two points."""
    ordered = np.sort(x)
    if ordered[0] == ordered[-1]:
        return np.array([ordered[0]]), float(ordered[0]), 0.0, False
    if np.count_nonzero(ordered == ordered[0]) != x.size / 2:
        return None
    if np.count_nonzero(ordered == ordered[-1]) != x.size / 2:
        return None
    low, high = Fraction(float(ordered[0])), Fraction(float(ordered[-1]))
    center, gap = (low+high)/2, (high-low)/2
    discriminant = gap*gap-Fraction(k)
    if discriminant <= 0:
        middle, middle_error = _rounded_interval(center, center, tolerance)
        return np.array([middle]), middle, middle_error, False
    # Exact floor of sqrt(discriminant) on a binary grid. No subtraction of
    # rounded square roots or standardized half-gaps near the flat transition.
    bits = max(60, math.ceil(-math.log2(tolerance))+8)
    integer = math.isqrt((discriminant.numerator << (2*bits)) // discriminant.denominator)
    lower, upper = Fraction(integer, 1 << bits), Fraction(integer+1, 1 << bits)
    left, error = _rounded_interval(center-upper, center-lower, tolerance)
    if select:
        return None, left, error, True
    middle, _ = _rounded_interval(center, center, tolerance)
    right, _ = _rounded_interval(center+lower, center+upper, tolerance)
    return np.array([left, middle, right]), left, error, True


def _exact_roots(x, k, tolerance, *, select):
    """Separate global products with QQ bounds, independently of float scores.

    Imports SymPy lazily. Exact symmetry proves equal objective values at mirror
    roots; unresolved nonsymmetric ties are errors, not numerical tie breaks.
    This bounded-size fallback is intentionally distinct from the validation
    oracle in scripts/selected_mle_reference_oracle.py.
    """
    if x.size > 20:
        raise StudentMLEError("exact_size_limit", "ambiguous cases with n > 20 require an external oracle")
    try:
        import sympy as sp
    except ImportError as exc:
        raise StudentMLEError("exact_dependency", "install sympy>=1.14,<2 for ambiguous inputs") from exc
    origin = sp.Rational(float(np.sort(x)[x.size//2]))
    values = sorted(sp.Rational(float(v))-origin for v in x)
    kk = sp.Rational(k)
    variable = sp.Symbol("location")
    product = sp.Poly(1, variable, domain=sp.QQ)
    for value in values:
        product *= sp.Poly([1, -2*value, value*value+kk], variable, domain=sp.QQ)
    derivative = product.diff()
    symmetric = all(v+values[-1-i] == values[0]+values[-1] for i, v in enumerate(values))
    base_width = min(sp.Rational(tolerance)/32, sp.Rational(1, 10**16))
    for digits in (16, 32, 64, 128):
        intervals = derivative.intervals(eps=min(base_width, sp.Rational(1, 10**digits)), fast=True)
        if not select:
            roots = [_rounded_interval(Fraction(a+origin), Fraction(b+origin), tolerance)[0]
                     for ((a,b), _) in intervals]
            if any(a >= b for a, b in zip(roots, roots[1:])):
                raise StudentMLEError("root_resolution", "distinct stationary roots cannot be represented by distinct floats")
            return np.asarray(roots), {"method": "exact_rational_roots", "root_count": len(roots)}
        bounds = []
        for ((a,b), _) in intervals:
            lower = upper = sp.S.One
            for value in values:
                nearest = 0 if a <= value <= b else min((value-a)**2, (value-b)**2)
                farthest = max((value-a)**2, (value-b)**2)
                lower *= kk+nearest
                upper *= kk+farthest
            bounds.append((lower, upper))
        best_upper = min(upper for _, upper in bounds)
        possible = [i for i, (lower, _) in enumerate(bounds) if lower <= best_upper]
        exact_tie = symmetric and len(possible) == 2 and sum(possible) == len(intervals)-1
        if len(possible) != 1 and not exact_tie:
            continue
        winner = possible[0]
        a, b = intervals[winner][0]
        result, error = _rounded_interval(Fraction(a+origin), Fraction(b+origin), tolerance)
        return result, {"method": "exact_rational_bounds", "exact_tie": exact_tie,
                        "root_count": len(intervals), "digits": digits,
                        "absolute_error_bound": error}
    raise StudentMLEError("selection_unresolved", "global product bounds still overlap at 128 digits")


@njit(cache=True)
def _standardization_error(x, center, scale, values):
    total = 0.0
    for i in range(x.size):
        displacement = scale*values[i]
        reconstructed = center+displacement
        # Bound multiplication/addition/subtraction error in the reconstruction.
        total += abs(reconstructed-x[i]) + 2*_MACHINE_EPS*(
            abs(center)+abs(displacement)+abs(x[i]))
    # Rounded sqrt(k) changes the effective scale; its log-scale error is
    # covered conservatively by four eps per observation on IEEE float64.
    return total/scale+4*_MACHINE_EPS*x.size


def _standardize(x, k):
    # An observed order statistic avoids overflow in an even-sample median.
    center = float(np.sort(x)[x.size//2])
    scale = math.sqrt(k)
    with np.errstate(over="ignore", invalid="ignore"):
        values = (x-center)/scale
    if not np.isfinite(values).all() or np.max(np.abs(values)) > 1e50:
        raise StudentMLEError(
            "dynamic_range",
            "general samples with standardized magnitudes above 1e50 are unsupported; no search attempted",
        )
    # log likelihood at k=1 is 1-Lipschitz in each observation. Account for
    # the data perturbation introduced by rounding the centered coordinates.
    objective_error = _standardization_error(x, center, scale, values)
    return center, scale, np.ascontiguousarray(values), objective_error


@njit(cache=True)
def _polish_regular_root(x, mu):
    derivative = 0.0
    for _ in range(8):
        value = _score(x, mu, 1.0)
        derivative = 0.0
        for observation in x:
            r = observation-mu
            derivative += (r*r-1.0)/((1.0+r*r)*(1.0+r*r))
        if abs(derivative) <= 1e-6*x.size:
            return mu, derivative, value
        step = value/derivative
        mu -= step
        if abs(step) <= _MACHINE_EPS*(1.0+abs(mu)):
            break
    return mu, derivative, _score(x, mu, 1.0)


@njit(cache=True)
def _original_root_check(x, mu, scale):
    """Check original observations, independently of the centering roundoff."""
    value = derivative = 0.0
    for observation in x:
        r = (observation-mu)/scale
        denominator = 1.0+r*r
        value += r/denominator
        derivative += (r*r-1.0)/(denominator*denominator)
    return value, derivative


def stationary_points(data, k, *, location_tol=2e-13):
    """Return distinct real stationary roots, including degenerate ones.

    location_tol is an absolute tolerance in original data units. Complete
    enumeration disables likelihood pruning. Ambiguous intervals use QQ root
    isolation; unsupported accuracy/range cases raise StudentMLEError.
    """
    x, k = _validated_inputs(data, k)
    x = np.sort(x)
    tolerance = _validated_tolerance(location_tol)
    special = _two_point_roots(x, k, tolerance)
    if special is not None:
        return special[0]
    center, scale, values, objective_error = _standardize(x, k)
    try:
        if objective_error > 1e-8*x.size:
            raise RuntimeError("substantial data loss during standardization")
        roots = _stationary_points_certified(values, 1.0, tolerance/(8*scale), 0.0, -math.inf, False)
        refined = []
        for root in roots:
            root, derivative, residual = _polish_regular_root(values, root)
            if abs(derivative) <= 1e-6*x.size or abs(residual/derivative)*scale > tolerance/4:
                raise RuntimeError("flat or insufficiently polished stationary root")
            result = center+scale*root
            if not math.isfinite(result) or abs(np.spacing(result)) > tolerance:
                raise RuntimeError("output needs exact precision check")
            value, derivative = _original_root_check(x, result, scale)
            if abs(derivative) <= 1e-6*x.size or abs(value/derivative)*scale > tolerance/4:
                raise RuntimeError("centering lost requested root accuracy")
            refined.append(result)
        refined = np.sort(np.asarray(refined))
        if np.any(np.diff(refined) <= tolerance):
            raise RuntimeError("stationary candidates coalesced during polishing")
        return refined
    except RuntimeError:
        return _exact_roots(x, k, tolerance, select=False)[0]


def global_mle(data, k, *, location_tol=2e-13, return_diagnostics=False):
    """Global likelihood maximizer; smaller location only for proved exact ties.

    location_tol is absolute in original data units. The fast ordinary path is
    numerical, not formally certified. Flat roots, close objectives, search
    ambiguity and output precision trigger exact rational bounds at n <= 20.
    StudentMLEError explicitly reports unresolved selection or accuracy.
    """
    x, k = _validated_inputs(data, k)
    x = np.sort(x)
    tolerance = _validated_tolerance(location_tol)
    special = _two_point_roots(x, k, tolerance, select=True)
    if special is not None:
        result = special[1]
        info = {"method": "exact_two_point", "fallback": False,
                "absolute_error_bound": special[2], "exact_tie": special[3]}
    else:
        center, scale, values, objective_error = _standardize(x, k)
        reason = None
        try:
            if objective_error > 1e-8*x.size:
                raise RuntimeError("substantial data loss during standardization")
            incumbent_mu, incumbent_ll = _initial_mode_candidate(np.sort(values), 1.0)
            candidates = _stationary_points_certified(values, 1.0, tolerance/(8*scale),
                                                      incumbent_mu, incumbent_ll, True, 2*objective_error)
            objectives = np.asarray([_loglik(values, candidate, 1.0) for candidate in candidates])
            if not np.isfinite(objectives).all():
                raise RuntimeError("nonfinite objective")
            order = np.argsort(objectives)
            best = objectives[order[-1]]
            ambiguity = max(2*objective_error, 128*_MACHINE_EPS*max(1.0, abs(best), x.size))
            if len(order) > 1 and best-objectives[order[-2]] <= ambiguity:
                raise RuntimeError("ambiguous likelihood comparison")
            root, derivative, residual = _polish_regular_root(values, candidates[order[-1]])
            if derivative >= -1e-6*x.size:
                raise RuntimeError("flat or nonmaximum selected root")
            if abs(residual/derivative)*scale > tolerance/4:
                raise RuntimeError("insufficient root accuracy")
            result = float(center+scale*root)
            if not math.isfinite(result) or abs(np.spacing(result)) > tolerance:
                raise RuntimeError("output needs exact precision check")
            value, derivative = _original_root_check(x, result, scale)
            if derivative >= -1e-6*x.size or abs(value/derivative)*scale > tolerance/4:
                raise RuntimeError("centering lost requested root accuracy")
            info = {"method": "fast_float", "fallback": False, "exact_tie": False,
                    "score_residual_standardized": abs(residual)}
        except RuntimeError as exc:
            reason = str(exc)
        if reason is not None:
            result, info = _exact_roots(x, k, tolerance, select=True)
            info.update(fallback=True, fallback_reason=reason)
    if not math.isfinite(result):
        raise StudentMLEError("nonfinite_output", "no nonfinite estimate may be returned")
    return (result, info) if return_diagnostics else result


def loglik(data, mu, k):
    """Student log likelihood, omitting constants in mu; overflow-safe diagnostics."""
    x, k = _validated_inputs(data, k)
    mu = float(mu)
    if not math.isfinite(mu):
        raise ValueError("mu must be finite")
    if max(np.max(np.abs(x)), abs(mu)) > 1e150 or k < 1e-150:
        residual = x.astype(np.longdouble)-np.longdouble(mu)
        result = float(-np.longdouble(k+1)/2*np.sum(np.log1p(residual*residual/np.longdouble(k))))
    else:
        result = float(_loglik(x, mu, k))
    if not math.isfinite(result):
        raise StudentMLEError("diagnostic_range", "log likelihood cannot be represented as a float")
    return result


def score(data, mu, k):
    """Reduced Student location score; overflow-safe diagnostics."""
    x, k = _validated_inputs(data, k)
    mu = float(mu)
    if not math.isfinite(mu):
        raise ValueError("mu must be finite")
    if max(np.max(np.abs(x)), abs(mu)) > 1e150 or k < 1e-150:
        residual = x.astype(np.longdouble)-np.longdouble(mu)
        result = float(np.sum(residual/(np.longdouble(k)+residual*residual)))
    else:
        result = float(_score(x, mu, k))
    if not math.isfinite(result):
        raise StudentMLEError("diagnostic_range", "score cannot be represented as a float")
    return result


