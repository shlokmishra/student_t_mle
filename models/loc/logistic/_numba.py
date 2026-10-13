"""Numba implementation of the constrained Logistic Gibbs sampler."""

import math

import numpy as np
from numba import njit

from models.shared.numba_math import normal_cdf, normal_ppf

EPS_U = 1e-12


@njit(cache=True)
def logistic_logpdf(y, loc):
    t = y - loc
    if t > 0.0:
        return -t - 2.0 * math.log1p(math.exp(-t))
    return -t - 2.0 * (-t + math.log1p(math.exp(t)))


@njit(cache=True)
def psi(y):
    return math.tanh(y / 2.0)


@njit(cache=True)
def psi_inverse(z):
    return 2.0 * math.atanh(z)


@njit(cache=True)
def log_psi_prime_abs(y):
    z = psi(y)
    return math.log(0.5) + math.log(max(1.0 - z * z, 1e-30))


@njit(cache=True)
def q_logpdf(z, mu_current, mu_star):
    if z <= -1.0 or z >= 1.0:
        return -math.inf
    y = psi_inverse(z)
    return logistic_logpdf(y, mu_current - mu_star) - log_psi_prime_abs(y)


@njit(cache=True)
def sample_truncated_normal(center, sigma, lower, upper):
    a = (lower - center) / sigma
    b = (upper - center) / sigma
    fa = normal_cdf(a)
    fb = normal_cdf(b)
    p = fa + np.random.random() * max(fb - fa, 0.0)
    return center + sigma * normal_ppf(p)


@njit(cache=True)
def update_pair(x_out, x_current, i, j, mu_current, mu_star, sigma_z):
    xi = x_current[i]
    xj = x_current[j]
    zi = psi(xi - mu_star)
    zj = psi(xj - mu_star)
    delta = zi + zj
    low = max(-1.0 + 1e-12, delta - (1.0 - 1e-12))
    high = min(1.0 - 1e-12, delta + (1.0 - 1e-12))
    if low >= high:
        x_out[i] = xi
        x_out[j] = xj
        return 0, 0
    zi_prop = sample_truncated_normal(zi, sigma_z, low, high)
    zj_prop = delta - zi_prop
    if zj_prop <= -1.0 or zj_prop >= 1.0:
        x_out[i] = xi
        x_out[j] = xj
        return 0, 0
    log_cur = q_logpdf(zi, mu_current, mu_star) + q_logpdf(zj, mu_current, mu_star)
    log_prop = q_logpdf(zi_prop, mu_current, mu_star) + q_logpdf(zj_prop, mu_current, mu_star)
    a = (low - zi) / sigma_z
    b = (high - zi) / sigma_z
    a_back = (low - zi_prop) / sigma_z
    b_back = (high - zi_prop) / sigma_z
    log_forward = -0.5 * ((zi_prop - zi) / sigma_z) ** 2 - math.log(sigma_z) - math.log(max(normal_cdf(b) - normal_cdf(a), 1e-300))
    log_backward = -0.5 * ((zi - zi_prop) / sigma_z) ** 2 - math.log(sigma_z) - math.log(max(normal_cdf(b_back) - normal_cdf(a_back), 1e-300))
    log_alpha = log_prop - log_cur + log_backward - log_forward
    if math.log(max(np.random.random(), EPS_U)) < log_alpha:
        x_out[i] = psi_inverse(zi_prop) + mu_star
        x_out[j] = psi_inverse(zj_prop) + mu_star
        return 1, 1
    x_out[i] = xi
    x_out[j] = xj
    return 0, 1


@njit(cache=True)
def run_gibbs_kernel(seed, x0, mu_star, iterations, proposal_std_mu, proposal_std_z, prior_mean, prior_std):
    np.random.seed(seed)
    n = x0.size
    mus = np.zeros(iterations + 1)
    xs = np.zeros((iterations + 1, n))
    xs[0] = x0
    mus[0] = mu_star
    x_current = x0.copy()
    x_out = np.empty(n)
    mu_accepts = 0
    pair_accepts = 0
    z_accepts = 0
    for t in range(1, iterations + 1):
        mu_current = mus[t - 1]
        mu_candidate = mu_current + proposal_std_mu * np.random.normal()
        log_current = mu_logpdf(mu_current, x_current, prior_mean, prior_std)
        log_candidate = mu_logpdf(mu_candidate, x_current, prior_mean, prior_std)
        if math.log(max(np.random.random(), EPS_U)) < log_candidate - log_current:
            mu_current = mu_candidate
            mu_accepts += 1
        mus[t] = mu_current
        x_out[:] = x_current
        permutation = np.random.permutation(n)
        for p in range(0, n - 1, 2):
            pair_ok, z_ok = update_pair(x_out, x_current, permutation[p], permutation[p + 1], mu_current, mu_star, proposal_std_z)
            pair_accepts += pair_ok
            z_accepts += z_ok
        x_current[:] = x_out
        xs[t] = x_current
    return mus, xs, mu_accepts, pair_accepts, z_accepts


@njit(cache=True)
def mu_logpdf(mu, x, prior_mean, prior_std):
    value = -0.5 * ((mu - prior_mean) / prior_std) ** 2 - math.log(prior_std) - 0.5 * math.log(2.0 * math.pi)
    for i in range(x.size):
        value += logistic_logpdf(x[i], mu)
    return value
