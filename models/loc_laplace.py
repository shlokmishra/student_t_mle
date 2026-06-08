"""
Laplace location model (scale b known).

For odd n, Gibbs pins the unique sample median at mu_star. For even n, Gibbs
keeps the older median-interval target with half the state below and half above
mu_star.
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random, jit, vmap
from jax.scipy.stats import norm
from tqdm import tqdm

EPS_U = 1e-12


def _initial_x(mu_star, n, params):
    init = str(params.get("initialization", "central"))
    half = n // 2
    if init == "random":
        rng = np.random.default_rng(int(params.get("initialization_seed", 0)))
        left = -np.exp(rng.normal(0.0, 1.0, size=half))
        right = np.exp(rng.normal(0.0, 1.0, size=half))
        vals = list(mu_star + left)
        if n % 2:
            vals.append(mu_star)
        vals.extend(list(mu_star + right))
        return jnp.asarray(vals, dtype=float)
    if init == "central":
        amp = 1.0
    elif init == "tail_heavy":
        amp = float(params.get("initialization_tail_amplitude", 5.0))
    else:
        raise ValueError(f"Unknown initialization: {init}")
    if n % 2 == 1:
        return jnp.concatenate([
            (mu_star - amp) * jnp.ones(half),
            jnp.asarray([mu_star]),
            (mu_star + amp) * jnp.ones(half),
        ])
    return jnp.concatenate([
        (mu_star - amp) * jnp.ones(half),
        (mu_star + amp) * jnp.ones(half),
    ])


def get_mle(data, params):
    """MLE for Laplace location is the median."""
    return float(np.median(np.asarray(data)))


def sample_data(key, params, loc=0.0):
    """Sample from Laplace(loc, b)."""
    b = params.get("b", 1.0)
    n = params["n"]
    u = random.uniform(key, shape=(n,), minval=EPS_U, maxval=1.0 - EPS_U)
    left = loc + b * jnp.log(2.0 * u)
    right = loc - b * jnp.log(2.0 * (1.0 - u))
    return jnp.where(u < 0.5, left, right)


def get_benchmark_mle_samples(key, params, num_simulations=10000, verbose=False):
    """Samples from p(hat_theta | theta=0): median of Laplace(0, b) samples."""
    b = params.get("b", 1.0)
    n = params["n"]
    rng = np.random.default_rng(0)
    mle_samples = np.zeros(num_simulations)
    keys = random.split(key, num_simulations)
    for i in range(num_simulations):
        u = random.uniform(keys[i], shape=(n,), minval=EPS_U, maxval=1.0 - EPS_U)
        left = 0.0 + b * jnp.log(2.0 * u)
        right = 0.0 - b * jnp.log(2.0 * (1.0 - u))
        x = jnp.where(u < 0.5, left, right)
        mle_samples[i] = jnp.median(x)
        if verbose and (i + 1) % 10000 == 0:
            print(f"  Benchmark: {i+1}/{num_simulations}")
    return mle_samples


# --- JAX: Laplace logpdf, CDF, PPF, truncated sampling ---

@jit
def _laplace_logpdf(x, loc, b):
    return -jnp.log(2.0 * b) - jnp.abs(x - loc) / b


@jit
def _laplace_cdf(x, loc, b):
    left = 0.5 * jnp.exp((x - loc) / b)
    right = 1.0 - 0.5 * jnp.exp(-(x - loc) / b)
    return jnp.where(x < loc, left, right)


@jit
def _laplace_ppf(u, loc, b):
    u = jnp.clip(u, EPS_U, 1.0 - EPS_U)
    left = loc + b * jnp.log(2.0 * u)
    right = loc - b * jnp.log(2.0 * (1.0 - u))
    return jnp.where(u < 0.5, left, right)


@jit
def _sample_trunc_left(key, loc, b, upper):
    Fu = jnp.clip(_laplace_cdf(upper, loc, b), EPS_U, 1.0 - EPS_U)
    u = random.uniform(key, shape=(), minval=EPS_U, maxval=Fu)
    return _laplace_ppf(u, loc, b)


@jit
def _sample_trunc_right(key, loc, b, lower):
    Fl = jnp.clip(_laplace_cdf(lower, loc, b), EPS_U, 1.0 - EPS_U)
    u = random.uniform(key, shape=(), minval=Fl, maxval=1.0 - EPS_U)
    return _laplace_ppf(u, loc, b)


_sample_trunc_left_batch = vmap(_sample_trunc_left, in_axes=(0, None, None, None))
_sample_trunc_right_batch = vmap(_sample_trunc_right, in_axes=(0, None, None, None))


@jit
def _update_x_full(key, x_current, mu_current, mu_star, b):
    n = x_current.shape[0]
    half = n // 2
    is_odd = n % 2 == 1
    key_perm, key_left, key_right = random.split(key, 3)
    keys_left = random.split(key_left, half)
    keys_right = random.split(key_right, half)
    x_left = _sample_trunc_left_batch(keys_left, mu_current, b, mu_star)
    x_right = _sample_trunc_right_batch(keys_right, mu_current, b, mu_star)
    if is_odd:
        return jnp.concatenate([x_left, jnp.asarray([mu_star]), x_right], axis=0)
    perm = random.permutation(key_perm, n)
    x_new_perm = jnp.concatenate([x_left, x_right], axis=0)
    return x_new_perm[jnp.argsort(perm)]


@jit
def _unnorm_posterior_mu_logpdf(mu, x, prior_loc, prior_scale, b):
    return jnp.sum(_laplace_logpdf(x, loc=mu, b=b)) + norm.logpdf(mu, loc=prior_loc, scale=prior_scale)


@jit
def _update_mu_mh(key, mu_current, x_current, sigma_mu, prior_loc, prior_scale, b):
    key_prop, key_u = random.split(key, 2)
    mu_cand = mu_current + sigma_mu * random.normal(key_prop)
    log_cur = _unnorm_posterior_mu_logpdf(mu_current, x_current, prior_loc, prior_scale, b)
    log_cand = _unnorm_posterior_mu_logpdf(mu_cand, x_current, prior_loc, prior_scale, b)
    log_alpha = jnp.where(jnp.isfinite(log_cand - log_cur), log_cand - log_cur, -jnp.inf)
    accept = jnp.log(random.uniform(key_u, shape=(), minval=EPS_U, maxval=1.0)) < log_alpha
    return jnp.where(accept, mu_cand, mu_current), accept


def run_gibbs(key, mu_star, params, verbose=True):
    """Two-step Gibbs for Laplace median conditioning.

    Odd n pins the unique median at mu_star. Even n keeps the median-interval
    conditioning with half below and half above mu_star.
    """
    T = int(params["num_iterations_T"])
    n = int(params["n"])
    b = params.get("b", 1.0)
    mus = jnp.zeros(T + 1)
    xs = jnp.zeros((T + 1, n))
    x0 = _initial_x(mu_star, n, params)
    mus = mus.at[0].set(mu_star)
    xs = xs.at[0, :].set(x0)
    mu_acc = 0

    iters = range(1, T + 1)
    if verbose:
        iters = tqdm(iters, desc="Gibbs (Laplace)")
    for t in iters:
        key, key_mu, key_x = random.split(key, 3)
        x_cur = xs[t - 1]
        mu_new, acc = _update_mu_mh(
            key_mu, mus[t - 1], x_cur,
            params["proposal_std_mu"], params["prior_mean"], params["prior_std"], b
        )
        mus = mus.at[t].set(mu_new)
        mu_acc += int(acc)
        x_new = _update_x_full(key_x, x_cur, mu_new, mu_star, b)
        xs = xs.at[t, :].set(x_new)

    return {
        "mu_chain": mus,
        "x_chain": xs,
        "mu_acceptance_rate": mu_acc / T,
    }


def run_full_data_mh(key, x, params, verbose=True):
    """MH sampler for p(mu | x) with fixed data x. Returns mu_chain."""
    x = jnp.asarray(x)
    T = int(params["num_iterations_T"])
    b = params.get("b", 1.0)
    mus = jnp.zeros(T + 1)
    mus = mus.at[0].set(jnp.median(x))
    mu_acc = 0
    iters = range(1, T + 1)
    if verbose:
        iters = tqdm(iters, desc="Full-data MH (Laplace)")
    for t in iters:
        key, key_mu = random.split(key)
        mu_new, acc = _update_mu_mh(
            key_mu, mus[t - 1], x,
            params["proposal_std_mu"], params["prior_mean"], params["prior_std"], b
        )
        mus = mus.at[t].set(mu_new)
        mu_acc += int(acc)
    return {"mu_chain": mus, "mu_acceptance_rate": mu_acc / T}
