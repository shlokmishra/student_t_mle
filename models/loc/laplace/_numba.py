"""Numba implementation of the constrained Laplace Gibbs sampler."""

import math

import numpy as np
from numba import njit

EPS_U = 1e-12


@njit(cache=True)
def laplace_logpdf(x, loc, b):
    return -math.log(2.0 * b) - abs(x - loc) / b


@njit(cache=True)
def laplace_cdf(x, loc, b):
    if x < loc:
        return 0.5 * math.exp((x - loc) / b)
    return 1.0 - 0.5 * math.exp(-(x - loc) / b)


@njit(cache=True)
def laplace_ppf(u, loc, b):
    u = min(max(u, EPS_U), 1.0 - EPS_U)
    if u < 0.5:
        return loc + b * math.log(2.0 * u)
    return loc - b * math.log(2.0 * (1.0 - u))


@njit(cache=True)
def sample_left(mu_current, mu_star, b):
    upper = min(max(laplace_cdf(mu_star, mu_current, b), EPS_U), 1.0 - EPS_U)
    return laplace_ppf(np.random.uniform(EPS_U, upper), mu_current, b)


@njit(cache=True)
def sample_right(mu_current, mu_star, b):
    lower = min(max(laplace_cdf(mu_star, mu_current, b), EPS_U), 1.0 - EPS_U)
    return laplace_ppf(np.random.uniform(lower, 1.0 - EPS_U), mu_current, b)


@njit(cache=True)
def mu_logpdf(mu, x, prior_mean, prior_std, b):
    value = -0.5 * ((mu - prior_mean) / prior_std) ** 2 - math.log(prior_std) - 0.5 * math.log(2.0 * math.pi)
    for i in range(x.size):
        value += laplace_logpdf(x[i], mu, b)
    return value


@njit(cache=True)
def run_gibbs_kernel(seed, x0, mu_star, iterations, proposal_std_mu, prior_mean, prior_std, b):
    np.random.seed(seed)
    n = x0.size
    half = n // 2
    mus = np.zeros(iterations + 1)
    xs = np.zeros((iterations + 1, n))
    mus[0] = mu_star
    xs[0] = x0
    x_current = x0.copy()
    mu_accepts = 0
    for t in range(1, iterations + 1):
        mu_current = mus[t - 1]
        mu_candidate = mu_current + proposal_std_mu * np.random.normal()
        log_current = mu_logpdf(mu_current, x_current, prior_mean, prior_std, b)
        log_candidate = mu_logpdf(mu_candidate, x_current, prior_mean, prior_std, b)
        if math.log(max(np.random.random(), EPS_U)) < log_candidate - log_current:
            mu_current = mu_candidate
            mu_accepts += 1
        mus[t] = mu_current
        for i in range(half):
            x_current[i] = sample_left(mu_current, mu_star, b)
            x_current[n - half + i] = sample_right(mu_current, mu_star, b)
        if n % 2 == 1:
            x_current[half] = mu_star
        else:
            permutation = np.random.permutation(n)
            x_copy = x_current.copy()
            for i in range(n):
                x_current[i] = x_copy[permutation[i]]
        xs[t] = x_current
    return mus, xs, mu_accepts
