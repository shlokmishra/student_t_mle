"""Shared Numba kernels for the Student-t location-parameter MH update."""

from __future__ import annotations

import math

import numpy as np
from numba import njit


EPS_U = 1e-12


def seed_from_key(key) -> int:
    """Derive a stable Numba/NumPy seed from a two-word JAX-style key."""
    words = np.asarray(key, dtype=np.uint32).reshape(-1)
    if words.size < 2:
        raise ValueError("expected a two-word JAX PRNG key")
    return int((int(words[0]) ^ (int(words[1]) << 1)) & 0x7FFFFFFF)


def rng_from_key(key) -> np.random.Generator:
    """Create a local NumPy RNG without importing or calling JAX."""
    return np.random.default_rng(seed_from_key(key))


@njit(cache=True)
def unnormalized_mu_logpdf(mu, x, prior_loc, prior_scale, k):
    """Scalar log p(mu | x), omitting constants independent of mu."""
    total = -0.5 * ((mu - prior_loc) / prior_scale) ** 2
    for i in range(x.size):
        y = x[i] - mu
        total -= 0.5 * (k + 1.0) * math.log1p((y * y) / k)
    return total


@njit(cache=True)
def update_mu_mh_from_variates(
    mu_current,
    x_current,
    proposal_std,
    prior_loc,
    prior_scale,
    k,
    proposal_noise,
    accept_uniform,
):
    """Take one MH step with externally supplied standard-normal/uniform draws."""
    mu_candidate = mu_current + proposal_std * proposal_noise
    log_alpha = (
        unnormalized_mu_logpdf(mu_candidate, x_current, prior_loc, prior_scale, k)
        - unnormalized_mu_logpdf(mu_current, x_current, prior_loc, prior_scale, k)
    )
    if math.isfinite(log_alpha) and math.log(max(accept_uniform, EPS_U)) < log_alpha:
        return mu_candidate, True
    return mu_current, False


@njit(cache=True)
def update_mu_mh_random(mu_current, x_current, proposal_std, prior_loc, prior_scale, k):
    """Take one MH step using Numba's RNG; intended for compiled Gibbs loops."""
    return update_mu_mh_from_variates(
        mu_current,
        x_current,
        proposal_std,
        prior_loc,
        prior_scale,
        k,
        np.random.normal(),
        np.random.random(),
    )
