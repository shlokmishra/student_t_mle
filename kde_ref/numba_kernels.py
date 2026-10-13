"""Numba kernels for the custom KDE/reference calculations."""

import math

import numpy as np
from numba import njit, prange


@njit(cache=True)
def student_log_kernel(u, nu, log_constant):
    return log_constant - 0.5 * (nu + 1.0) * math.log1p((u * u) / nu)


@njit(cache=True, parallel=True)
def fixed_student_density(eval_points, centers, bandwidth, nu):
    """Evaluate a fixed-bandwidth Student-t KDE stably at all points."""
    out = np.empty(eval_points.size, dtype=np.float64)
    log_constant = math.lgamma((nu + 1.0) / 2.0) - 0.5 * math.log(nu * math.pi) - math.lgamma(nu / 2.0)
    log_bandwidth = math.log(bandwidth)
    normalizer = math.log(centers.size)
    for i in prange(eval_points.size):
        maximum = -math.inf
        for j in range(centers.size):
            value = student_log_kernel((eval_points[i] - centers[j]) / bandwidth, nu, log_constant) - log_bandwidth
            if value > maximum:
                maximum = value
        total = 0.0
        for j in range(centers.size):
            value = student_log_kernel((eval_points[i] - centers[j]) / bandwidth, nu, log_constant) - log_bandwidth
            total += math.exp(value - maximum)
        out[i] = math.exp(maximum + math.log(total) - normalizer)
    return out


@njit(cache=True, parallel=True)
def adaptive_student_logpdf(eval_points, centers, bandwidths, nu, log_eps):
    """Evaluate a variable-bandwidth Student-t KDE in log space."""
    out = np.empty(eval_points.size, dtype=np.float64)
    log_constant = math.lgamma((nu + 1.0) / 2.0) - 0.5 * math.log(nu * math.pi) - math.lgamma(nu / 2.0)
    normalizer = math.log(centers.size)
    for i in prange(eval_points.size):
        maximum = -math.inf
        for j in range(centers.size):
            bandwidth = bandwidths[j]
            value = student_log_kernel((eval_points[i] - centers[j]) / bandwidth, nu, log_constant) - math.log(bandwidth)
            if value > maximum:
                maximum = value
        total = 0.0
        for j in range(centers.size):
            bandwidth = bandwidths[j]
            value = student_log_kernel((eval_points[i] - centers[j]) / bandwidth, nu, log_constant) - math.log(bandwidth)
            total += math.exp(value - maximum)
        out[i] = max(maximum + math.log(total) - normalizer, log_eps)
    return out


@njit(cache=True)
def normal_logpdf(values, loc, scale):
    constant = math.log(scale) + 0.5 * math.log(2.0 * math.pi)
    out = np.empty(values.size, dtype=np.float64)
    for i in range(values.size):
        standardized = (values[i] - loc) / scale
        out[i] = -0.5 * standardized * standardized - constant
    return out
