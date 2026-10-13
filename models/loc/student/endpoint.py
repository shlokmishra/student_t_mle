"""Sufficient endpoint certificates. Inconclusive is never mathematical rejection."""
import math
import numpy as np
from numba import njit
from .n3_endpoint import certify_endpoint, RootCertificate, CertifiedN3State, _difference, _square


@njit(cache=True)
def score_bounds(x, t, k):
    low = high = 0.
    for value in x:
        a, b = _difference(value, t)
        c, d = _square(a, b)
        dl, dh = np.nextafter(k+c, -np.inf), np.nextafter(k+d, np.inf)
        if not math.isfinite(dh) or dl <= 0:
            return np.nan, np.nan
        terms = (a/dl, a/dh, b/dl, b/dh)
        low = np.nextafter(low+np.nextafter(min(terms), -np.inf), -np.inf)
        high = np.nextafter(high+np.nextafter(max(terms), np.inf), np.inf)
    return low, high


@njit(cache=True)
def _cluster_certificate(x, t, k, tol):
    dl, dh = _difference(np.max(x), np.min(x))
    _, diameter_sq = _square(dl, dh)
    if not diameter_sq < k:
        return 0, np.nan, np.nan
    lo, hi = np.nextafter(t-tol, t), np.nextafter(t+tol, t)
    if not lo < t < hi or max(_difference(t,lo)[1], _difference(hi,t)[1]) > tol:
        return 0, lo, hi
    a, _ = score_bounds(x, lo, k); _, b = score_bounds(x, hi, k)
    if math.isfinite(a) and math.isfinite(b) and a > 0 > b:
        return 1, lo, hi
    return 0, lo, hi


def endpoint_certificate(x, t, k, tolerance, geometry, mode='auto', scale=1.):
    # No rounded scaling is admitted into a binary-input certificate.
    if mode == 'selector' or scale != 1.:
        return None
    if len(x) == 3:
        return certify_endpoint(x, t, k, tolerance, geometry.curvature, geometry.score)
    if mode == 'n3':
        return None
    status, lo, hi = _cluster_certificate(np.asarray(x), t, k, tolerance)
    return RootCertificate(status, 'strict_concavity_bracket' if status else 'cluster_inconclusive', lo, hi)


def certified_state(geometry, release, cert):
    error = float(np.nextafter(max(release-cert.lower, cert.upper-release), np.inf))
    return CertifiedN3State(geometry, float(release), False, cert, error)
