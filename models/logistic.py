"""
Logistic location model (scale=1). MLE from score equation sum tanh((x-mu)/2)=0; Gibbs via psi(y)=tanh(y/2).
"""

import numpy as np
from scipy.optimize import root_scalar
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random, vmap, jit
from jax.scipy.stats import norm, truncnorm
from jax.nn import softplus
from tqdm import tqdm

EPS_Z = 1e-12
EPS_U = 1e-12


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


def get_benchmark_mle_samples(params, num_simulations=10000, verbose=False):
    """Samples from p(hat_theta | theta=0) for logistic."""
    n = params["n"]
    rng = np.random.default_rng(0)
    mle_samples = np.zeros(num_simulations)
    for i in range(num_simulations):
        u = rng.uniform(EPS_U, 1.0 - EPS_U, size=n)
        x = np.log(u / (1.0 - u))
        mle_samples[i] = get_mle(x, params)
        if verbose and (i + 1) % 10000 == 0:
            print(f"  Benchmark: {i+1}/{num_simulations}")
    return mle_samples


# --- JAX: logistic logpdf, psi = tanh(y/2), single inverse ---

@jit
def _logistic_logpdf(y, loc, scale=1.0):
    t = (y - loc) / scale
    return -t - jnp.log(scale) - 2.0 * softplus(-t)


def _z_support():
    return (-1.0 + EPS_Z, 1.0 - EPS_Z)


@jit
def _psi(y):
    return jnp.tanh(y / 2.0)


@jit
def _psi_inverse(z):
    return 2.0 * jnp.arctanh(z)


@jit
def _log_psi_prime_abs(y):
    z = _psi(y)
    return jnp.log(0.5) + jnp.log(jnp.maximum(1.0 - z * z, 1e-30))


@jit
def _fy_logpdf(y, mu_current, mu_star):
    loc = mu_current - mu_star
    return _logistic_logpdf(y, loc=loc, scale=1.0)


@jit
def _q_logpdf(z, mu_current, mu_star):
    z_min, z_max = _z_support()
    in_supp = (z > z_min) & (z < z_max)
    y = _psi_inverse(z)
    log_q = _fy_logpdf(y, mu_current, mu_star) - _log_psi_prime_abs(y)
    return jnp.where(in_supp, log_q, -jnp.inf)


@jit
def _q_tilde_logpdf(z, delta, mu_current, mu_star):
    return _q_logpdf(z, mu_current, mu_star) + _q_logpdf(delta - z, mu_current, mu_star)


def _update_z_one(key, z_current, delta, mu_current, mu_star, sigma_z):
    key_prop, key_u = random.split(key, 2)
    low, high = _z_support()
    low2, high2 = delta - high, delta - low
    low_int = jnp.maximum(low, low2)
    high_int = jnp.minimum(high, high2)
    valid = low_int < high_int

    def do_reject(_):
        return z_current, False

    def do_update(_):
        a = (low_int - z_current) / sigma_z
        b = (high_int - z_current) / sigma_z
        z_prop = z_current + sigma_z * random.truncated_normal(key_prop, shape=(), lower=a, upper=b)
        log_k_cur = truncnorm.logpdf(z_prop, a=a, b=b, loc=z_current, scale=sigma_z)
        a_back = (low_int - z_prop) / sigma_z
        b_back = (high_int - z_prop) / sigma_z
        log_k_back = truncnorm.logpdf(z_current, a=a_back, b=b_back, loc=z_prop, scale=sigma_z)
        log_cur = _q_tilde_logpdf(z_current, delta, mu_current, mu_star)
        log_prop = _q_tilde_logpdf(z_prop, delta, mu_current, mu_star)
        log_alpha = jnp.where(jnp.isfinite(log_prop - log_cur), log_prop - log_cur + log_k_back - log_k_cur, -jnp.inf)
        accept = jnp.log(random.uniform(key_u, minval=EPS_U, maxval=1.0)) < log_alpha
        return jnp.where(accept, z_prop, z_current), accept

    return jax.lax.cond(valid, do_update, do_reject, operand=None)


def _update_xi_xj_one(key, xi, xj, mu_current, mu_star, sigma_z):
    key_z = key
    yi, yj = xi - mu_star, xj - mu_star
    zi, zj = _psi(yi), _psi(yj)
    delta = zi + zj
    zi_tilde, z_acc = _update_z_one(key_z, zi, delta, mu_current, mu_star, sigma_z)
    zj_tilde = delta - zi_tilde
    z_min, z_max = _z_support()
    in_supp_j = (zj_tilde > z_min) & (zj_tilde < z_max)

    def reject_pair(_):
        return xi, xj, False, z_acc

    def accept_pair(_):
        yi_tilde = _psi_inverse(zi_tilde)
        yj_tilde = _psi_inverse(zj_tilde)
        return yi_tilde + mu_star, yj_tilde + mu_star, True, z_acc

    return jax.lax.cond(in_supp_j, accept_pair, reject_pair, operand=None)


@jit
def _update_x_full(key, x_current, mu_current, mu_star, sigma_z):
    n = x_current.shape[0]
    key_perm, key_pairs = random.split(key)
    perm = random.permutation(key_perm, n)
    x_perm = x_current[perm]
    xis, xjs = x_perm[0::2], x_perm[1::2]
    keys = random.split(key_pairs, xis.shape[0])
    batch = vmap(_update_xi_xj_one, in_axes=(0, 0, 0, None, None, None))
    xis_new, xjs_new, pair_acc, z_acc = batch(keys, xis, xjs, mu_current, mu_star, sigma_z)
    x_new_perm = jnp.stack([xis_new, xjs_new], axis=1).reshape(-1)
    return x_new_perm[jnp.argsort(perm)], jnp.sum(pair_acc), jnp.sum(z_acc)


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
    mus = jnp.zeros(T + 1)
    xs = jnp.zeros((T + 1, n))
    x0 = jnp.ones(n) * mu_star
    mus = mus.at[0].set(mu_star)
    xs = xs.at[0, :].set(x0)
    total_pairs = T * (n // 2)
    mu_acc, pair_acc, z_acc = 0, 0, 0

    iters = range(1, T + 1)
    if verbose:
        iters = tqdm(iters, desc="Gibbs (Logistic)")
    for t in iters:
        key, key_mu, key_x = random.split(key, 3)
        x_cur = xs[t - 1]
        mu_new, acc_mu = _update_mu_mh(
            key_mu, mus[t - 1], x_cur,
            params["proposal_std_mu"], params["prior_mean"], params["prior_std"]
        )
        mus = mus.at[t].set(mu_new)
        mu_acc += int(acc_mu)
        x_new, npairs, nz = _update_x_full(key_x, x_cur, mu_new, mu_star, params["proposal_std_z"])
        xs = xs.at[t, :].set(x_new)
        pair_acc += int(npairs)
        z_acc += int(nz)

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
