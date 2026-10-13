"""Guarded unit-scale n=3 root certificates, not a replacement MLE selector.

Binary64 inputs denote exact real numbers. With IEEE round-to-nearest basic
operations and gradual underflow, nextafter encloses each operation. No
fastmath, Newton residual heuristic, or approximate score-zero identity is used.
An inconclusive certificate must fall back to the existing selected-state check.
"""
from dataclasses import dataclass
import math

import numpy as np
from numba import njit


REASONS = ('bracket', 'root_interval_disjoint', 'unrepresentable_bracket',
           'interval_unresolved', 'fold_adjacent', 'curvature_or_displacement',
           'unique_root_domain', 'score_signs')


@njit(cache=True)
def _difference(a, b):
    v = a-b
    return np.nextafter(v, -np.inf), np.nextafter(v, np.inf)


@njit(cache=True)
def _square(lo, hi):
    upper = np.nextafter(max(lo*lo, hi*hi), np.inf)
    lower = 0. if lo <= 0. <= hi else max(0., np.nextafter(min(lo*lo, hi*hi), -np.inf))
    return lower, upper


@njit(cache=True)
def score_interval(x, location, k):
    """Outward interval for sum (x-location)/(k+(x-location)^2)."""
    total_lo = total_hi = 0.
    for i in range(3):
        lo, hi = _difference(x[i], location)
        sl, su = _square(lo, hi)
        dl, du = np.nextafter(k+sl, -np.inf), np.nextafter(k+su, np.inf)
        if not (math.isfinite(lo) and math.isfinite(hi) and math.isfinite(du) and dl > 0.):
            return np.nan, np.nan
        a, b, c, d = lo/dl, lo/du, hi/dl, hi/du
        vl = np.nextafter(min(a,b,c,d), -np.inf)
        vu = np.nextafter(max(a,b,c,d), np.inf)
        total_lo = np.nextafter(total_lo+vl, -np.inf)
        total_hi = np.nextafter(total_hi+vu, np.inf)
    return total_lo, total_hi


@njit(cache=True)
def _certificate(x, t, k, tolerance, curvature, score):
    # Inward endpoints avoid claiming the rounding of t +/- tolerance is exact.
    lo = np.nextafter(t-tolerance, t)
    hi = np.nextafter(t+tolerance, t)
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < t < hi):
        return 0, 2, lo, hi
    _, el = _difference(t, lo)
    _, eh = _difference(hi, t)
    if max(el,eh) > tolerance:
        return 0, 2, lo, hi
    if not (curvature*k > math.sqrt(np.finfo(np.float64).eps)) or abs(score)/curvature > tolerance:
        return 0, 5, lo, hi
    # This guard only requests fallback; it never removes fold states from support.
    for i in range(3):
        yl, yu = _difference(x[i], t)
        sl, su = _square(yl, yu)
        if not math.isfinite(su):
            return 0, 3, lo, hi
        if sl <= k*(1+1e-8) and su >= k*(1-1e-8):
            return 0, 4, lo, hi
    a,b,c = x[0],x[1],x[2]
    median = max(min(a,b), min(max(a,b),c))
    # Entire allowed release window outside the unique-root interval: proved invalid.
    window_hi = np.nextafter(t+tolerance, np.inf)
    window_lo = np.nextafter(t-tolerance, -np.inf)
    yl,yu = _difference(median, window_hi)
    sl,_ = _square(yl,yu)
    if yl > 0. and sl > k:
        return -1, 1, lo, hi
    yl,yu = _difference(median, window_lo)
    sl,_ = _square(yl,yu)
    if yu < 0. and sl > k:
        return -1, 1, lo, hi
    for endpoint in (lo,hi):
        yl,yu = _difference(median, endpoint)
        _,su = _square(yl,yu)
        if not su < k:
            return 0, 6, lo, hi
    left_lo,left_hi = score_interval(x,lo,k)
    right_lo,right_hi = score_interval(x,hi,k)
    if not (math.isfinite(left_lo) and math.isfinite(right_hi)):
        return 0, 3, lo, hi
    if left_lo > 0. and right_hi < 0.:
        return 1, 0, lo, hi
    return 0, 7, lo, hi


@dataclass(frozen=True)
class RootCertificate:
    status: int  # 1: selected root bracketed; -1: excluded; 0: fallback required
    reason: str
    lower: float
    upper: float


def certify_endpoint(x, t, k, tolerance, curvature, score):
    x = np.asarray(x)
    if np.iscomplexobj(x) or x.shape != (3,) or not np.isfinite(x).all():
        raise ValueError('three finite real observations required')
    if not all(math.isfinite(v) for v in (t,k,tolerance,curvature,score)) or min(k,tolerance) <= 0:
        raise ValueError('finite inputs and positive shape/location tolerance required')
    status, reason, lo, hi = _certificate(np.asarray(x,dtype=float), float(t), float(k),
                                         float(tolerance), float(curvature), float(score))
    return RootCertificate(status, REASONS[reason], lo, hi)


@dataclass(frozen=True)
class CertifiedN3State:
    geometry: object
    selected_mle: float  # bracket midpoint approximation, NOT an exact computed MLE
    selector_fallback: bool
    root_certificate: RootCertificate
    selected_mle_error_bound: float
