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
from jax import random, jit
from jax.scipy.stats import norm
from tqdm import tqdm

from models.loc.laplace._numba import run_gibbs_kernel

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
        return np.asarray(vals, dtype=float)
    if init == "central":
        amp = 1.0
    elif init == "tail_heavy":
        amp = float(params.get("initialization_tail_amplitude", 5.0))
    else:
        raise ValueError(f"Unknown initialization: {init}")
    if n % 2 == 1:
        return np.concatenate([
            np.full(half, mu_star - amp),
            np.asarray([mu_star]),
            np.full(half, mu_star + amp),
        ])
    return np.concatenate([
        np.full(half, mu_star - amp),
        np.full(half, mu_star + amp),
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


# --- JAX: shared full-data mu update ---

@jit
def _laplace_logpdf(x, loc, b):
    return -jnp.log(2.0 * b) - jnp.abs(x - loc) / b


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
    seed = int(np.asarray(random.randint(key, (), minval=0, maxval=2**31 - 1)))
    x0 = np.asarray(_initial_x(mu_star, n, params), dtype=np.float64)
    if verbose:
        print(f"Gibbs (Laplace): running {T} iterations", flush=True)
    mus, xs, mu_acc = run_gibbs_kernel(
        seed, x0, float(mu_star), T, float(params["proposal_std_mu"]),
        float(params["prior_mean"]), float(params["prior_std"]), float(b)
    )
    if verbose:
        print("Gibbs (Laplace): complete", flush=True)

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
