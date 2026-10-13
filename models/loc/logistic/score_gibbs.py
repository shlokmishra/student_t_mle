"""
Logistic location model (scale=1). MLE from score equation sum tanh((x-mu)/2)=0; Gibbs via psi(y)=tanh(y/2).
"""

import numpy as np
from scipy.optimize import root_scalar
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random, jit
from jax.scipy.stats import norm
from jax.nn import softplus
from tqdm import tqdm

from models.loc.logistic._numba import run_gibbs_kernel

EPS_Z = 1e-12
EPS_U = 1e-12


def _initial_x(mu_star, n, params):
    init = str(params.get("initialization", "central"))
    if init == "central":
        return np.full(n, float(mu_star), dtype=float)
    if init == "tail_heavy":
        amp = float(params.get("initialization_tail_amplitude", 5.0))
        vals = [mu_star + amp, mu_star - amp] * (n // 2)
        if n % 2:
            vals.append(mu_star)
        return np.asarray(vals[:n], dtype=float)
    if init == "random":
        rng = np.random.default_rng(int(params.get("initialization_seed", 0)))
        vals = []
        for _ in range(n // 2):
            amp = float(np.exp(rng.normal(0.0, 1.0)))
            vals.extend([mu_star + amp, mu_star - amp])
        if n % 2:
            vals.append(mu_star)
        rng.shuffle(vals)
        return np.asarray(vals[:n], dtype=float)
    raise ValueError(f"Unknown initialization: {init}")


def get_mle(data, params):
    """MLE for logistic location: root of score sum_i tanh((x_i - mu)/2) = 0."""
    x = np.asarray(data)

    def score(mu):
        return float(np.sum(np.tanh((x - mu) / 2.0)))

    bracket = (float(x.min() - 10), float(x.max() + 10))
    result = root_scalar(score, bracket=bracket, method="brentq")
    if not result.converged:
        raise RuntimeError("Logistic MLE root finding did not converge.")
    return result.root


def sample_data(key, params, loc=0.0):
    """Sample from Logistic(loc, scale=1). Inverse CDF: x = loc + log(u/(1-u))."""
    n = params["n"]
    u = random.uniform(key, shape=(n,), minval=EPS_U, maxval=1.0 - EPS_U)
    return loc + jnp.log(u / (1.0 - u))


def get_benchmark_mle_samples(key, params, num_simulations=10000, verbose=False):
    """Samples from p(hat_theta | theta=0) for logistic."""
    n = params["n"]
    rng = np.random.default_rng(0)
    mle_samples = np.zeros(num_simulations)
    keys = random.split(key, num_simulations)
    for i in range(num_simulations):
        u = random.uniform(keys[i], shape=(n,), minval=EPS_U, maxval=1.0 - EPS_U)
        x = jnp.log(u / (1.0 - u))
        mle_samples[i] = get_mle(x, params)
        if verbose and (i + 1) % 10000 == 0:
            print(f"  Benchmark: {i+1}/{num_simulations}")
    return mle_samples


# --- JAX: shared full-data mu update ---

@jit
def _logistic_logpdf(y, loc, scale=1.0):
    t = (y - loc) / scale
    return -t - jnp.log(scale) - 2.0 * softplus(-t)


def _unnorm_posterior_mu_logpdf(mu, x, prior_loc, prior_scale):
    mu = jnp.asarray(mu)
    x = jnp.asarray(x)
    if mu.ndim == 0:
        loglik = jnp.sum(_logistic_logpdf(x, loc=mu, scale=1.0))
    else:
        loglik = jnp.sum(_logistic_logpdf(x[:, None], loc=mu[None, :], scale=1.0), axis=0)
    return loglik + norm.logpdf(mu, loc=prior_loc, scale=prior_scale)


@jit
def _update_mu_mh(key, mu_current, x_current, sigma_mu, prior_loc, prior_scale):
    key_prop, key_u = random.split(key)
    mu_cand = mu_current + sigma_mu * random.normal(key_prop)
    log_cur = _unnorm_posterior_mu_logpdf(mu_current, x_current, prior_loc, prior_scale)
    log_cand = _unnorm_posterior_mu_logpdf(mu_cand, x_current, prior_loc, prior_scale)
    log_alpha = jnp.where(jnp.isfinite(log_cand - log_cur), log_cand - log_cur, -jnp.inf)
    accept = jnp.log(random.uniform(key_u, minval=EPS_U, maxval=1.0)) < log_alpha
    return jnp.where(accept, mu_cand, mu_current), accept


def run_gibbs(key, mu_star, params, verbose=True):
    """Two-step Gibbs: (1) mu | x MH, (2) x | mu, MLE=mu_star (psi(y)=tanh(y/2))."""
    T = int(params["num_iterations_T"])
    n = int(params["n"])
    seed = int(np.asarray(random.randint(key, (), minval=0, maxval=2**31 - 1)))
    x0 = np.asarray(_initial_x(mu_star, n, params), dtype=np.float64)
    total_pairs = T * (n // 2)
    if verbose:
        print(f"Gibbs (Logistic): running {T} iterations", flush=True)
    mus, xs, mu_acc, pair_acc, z_acc = run_gibbs_kernel(
        seed, x0, float(mu_star), T, float(params["proposal_std_mu"]),
        float(params["proposal_std_z"]), float(params["prior_mean"]), float(params["prior_std"])
    )
    if verbose:
        print("Gibbs (Logistic): complete", flush=True)

    return {
        "mu_chain": mus,
        "x_chain": xs,
        "mu_acceptance_rate": mu_acc / T,
        "pair_acceptance_rate": pair_acc / total_pairs,
        "z_acceptance_rate": z_acc / total_pairs,
    }


def run_full_data_mh(key, x, params, verbose=True):
    """MH sampler for p(mu | x) with fixed data x. Returns mu_chain."""
    x = jnp.asarray(x)
    T = int(params["num_iterations_T"])
    mus = jnp.zeros(T + 1)
    mus = mus.at[0].set(jnp.median(x))
    mu_acc = 0
    iters = range(1, T + 1)
    if verbose:
        iters = tqdm(iters, desc="Full-data MH (Logistic)")
    for t in iters:
        key, key_mu = random.split(key)
        mu_new, acc = _update_mu_mh(
            key_mu, mus[t - 1], x,
            params["proposal_std_mu"], params["prior_mean"], params["prior_std"]
        )
        mus = mus.at[t].set(mu_new)
        mu_acc += int(acc)
    return {"mu_chain": mus, "mu_acceptance_rate": mu_acc / T}
