"""
Student-t location model (scale=1, df=k). MLE from score equation; Gibbs via psi(y)=y/(k+y^2).
"""

import numpy as np
import scipy.stats as stats
from scipy.optimize import root_scalar
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random, vmap, jit
from jax.scipy.stats import t, norm, truncnorm
from jax.scipy.special import logsumexp
from tqdm import tqdm

EPS_Z = 1e-12
EPS_U = 1e-12
EPS_DIV = 1e-12


def get_mle(data, params):
    """MLE for location of t-distribution (fixed df k, scale 1). Solves score equation."""
    k = params["k"]
    x = np.asarray(data)
    def score(mu):
        return np.sum((x - mu) / (k + (x - mu) ** 2))
    bracket = (float(x.min() - 10), float(x.max() + 10))
    result = root_scalar(score, bracket=bracket, method="brentq")
    if not result.converged:
        raise RuntimeError("MLE root finding did not converge.")
    return result.root


def sample_data(key, params, loc=0.0):
    """Sample data from Student-t(loc, scale=1, df=k)."""
    k = params["k"]
    n = params["n"]
    return random.t(key, df=k, shape=(n,)) + loc


def get_benchmark_mle_samples(key, params, num_simulations=10000, verbose=False):
    """Samples from p(hat_theta | theta=0)."""
    k, n = params["k"], params["n"]
    all_data = np.array(random.t(key, df=k, shape=(num_simulations, n)))
    return np.array([get_mle(data, params) for data in all_data])

# --- JAX Gibbs (z = psi(y), pairwise updates) ---

def _z_support(k):
    low = -1.0 / (2.0 * jnp.sqrt(k))
    high = 1.0 / (2.0 * jnp.sqrt(k))
    return low + EPS_Z, high - EPS_Z


def _psi(y, k):
    return y / (k + y ** 2)


def _psi_inverse(z, k):
    z_min, z_max = _z_support(k)
    z = jnp.clip(z, z_min, z_max)
    tval = 2.0 * jnp.sqrt(k) * z
    discr = jnp.clip(1.0 - tval * tval, 0.0)
    sqrt_discr = jnp.sqrt(discr)
    denom = 2.0 * z
    denom_safe = jnp.where(jnp.abs(denom) < EPS_DIV, jnp.sign(denom) * EPS_DIV + EPS_DIV, denom)
    y_plus = (1.0 + sqrt_discr) / denom_safe
    y_minus = (1.0 - sqrt_discr) / denom_safe
    y_plus = jnp.where(jnp.abs(z) < EPS_DIV, 0.0, y_plus)
    y_minus = jnp.where(jnp.abs(z) < EPS_DIV, 0.0, y_minus)
    y_lo = jnp.minimum(y_minus, y_plus)
    y_hi = jnp.maximum(y_minus, y_plus)
    return jnp.where(jnp.isfinite(y_lo), y_lo, 0.0), jnp.where(jnp.isfinite(y_hi), y_hi, 0.0)


def _log_psi_prime_abs(y, k):
    return jnp.log(jnp.abs(k - y ** 2) + 1e-30) - 2.0 * jnp.log(k + y ** 2)


def _fy_logpdf(y, mu_current, mu_star, k):
    loc = mu_current - mu_star
    return t.logpdf(y, df=k, loc=loc, scale=1.0)


def _q_logpdf(z, mu_current, mu_star, k):
    z_min, z_max = _z_support(k)
    in_supp = (z > z_min) & (z < z_max)
    y_lo, y_hi = _psi_inverse(z, k)
    y_vals = jnp.stack([y_lo, y_hi])
    log_fy = _fy_logpdf(y_vals, mu_current, mu_star, k)
    log_jac = _log_psi_prime_abs(y_vals, k)
    log_q = logsumexp(log_fy - log_jac)
    return jnp.where(in_supp, log_q, -jnp.inf)


def _q_tilde_logpdf(z, delta, mu_current, mu_star, k):
    return _q_logpdf(z, mu_current, mu_star, k) + _q_logpdf(delta - z, mu_current, mu_star, k)


def _update_z_one(key, z_current, delta, mu_current, mu_star, k, sigma_z):
    key_prop, key_u = random.split(key, 2)
    low, high = _z_support(k)
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
        log_cur = _q_tilde_logpdf(z_current, delta, mu_current, mu_star, k)
        log_prop = _q_tilde_logpdf(z_prop, delta, mu_current, mu_star, k)
        log_alpha = log_prop - log_cur + log_k_back - log_k_cur
        log_alpha = jnp.where(jnp.isfinite(log_alpha), log_alpha, -jnp.inf)
        accept = jnp.log(random.uniform(key_u, minval=EPS_U, maxval=1.0)) < log_alpha
        return jnp.where(accept, z_prop, z_current), accept

    return jax.lax.cond(valid, do_update, do_reject, operand=None)


def _safe_choice_2(key, candidates, log_w):
    log_w = jnp.where(jnp.isfinite(log_w), log_w, -jnp.inf)
    logZ = logsumexp(log_w)

    def fallback(_):
        return candidates[0]

    def sample(_):
        probs = jnp.exp(log_w - logZ)
        return candidates[random.choice(key, 2, p=probs)]

    return jax.lax.cond(jnp.isfinite(logZ), sample, fallback, operand=None)


def _update_xi_xj_one(key, xi, xj, mu_current, mu_star, k, sigma_z):
    key_z, key_i, key_j = random.split(key, 3)
    yi, yj = xi - mu_star, xj - mu_star
    zi, zj = _psi(yi, k), _psi(yj, k)
    delta = zi + zj
    zi_tilde, z_acc = _update_z_one(key_z, zi, delta, mu_current, mu_star, k, sigma_z)
    zj_tilde = delta - zi_tilde
    z_min, z_max = _z_support(k)
    in_supp_j = (zj_tilde > z_min) & (zj_tilde < z_max)

    def reject_pair(_):
        return xi, xj, False, z_acc

    def accept_pair(_):
        yi_lo, yi_hi = _psi_inverse(zi_tilde, k)
        yj_lo, yj_hi = _psi_inverse(zj_tilde, k)
        yi_cand = jnp.array([yi_lo, yi_hi])
        yj_cand = jnp.array([yj_lo, yj_hi])
        log_wi = _fy_logpdf(yi_cand, mu_current, mu_star, k) - _log_psi_prime_abs(yi_cand, k)
        log_wj = _fy_logpdf(yj_cand, mu_current, mu_star, k) - _log_psi_prime_abs(yj_cand, k)
        yi_new = _safe_choice_2(key_i, yi_cand, log_wi)
        yj_new = _safe_choice_2(key_j, yj_cand, log_wj)
        return yi_new + mu_star, yj_new + mu_star, True, z_acc

    return jax.lax.cond(in_supp_j, accept_pair, reject_pair, operand=None)


@jit
def _update_x_full(key, x_current, mu_current, mu_star, k, sigma_z):
    n = x_current.shape[0]
    key_perm, key_pairs = random.split(key)
    perm = random.permutation(key_perm, n)
    x_perm = x_current[perm]
    xis, xjs = x_perm[0::2], x_perm[1::2]
    n_pairs = xis.shape[0]
    keys = random.split(key_pairs, n_pairs)
    batch = vmap(_update_xi_xj_one, in_axes=(0, 0, 0, None, None, None, None))
    xis_new, xjs_new, pair_acc, z_acc = batch(keys, xis, xjs, mu_current, mu_star, k, sigma_z)
    x_new_perm = jnp.stack([xis_new, xjs_new], axis=1).reshape(-1)
    x_new = x_new_perm[jnp.argsort(perm)]
    return x_new, jnp.sum(pair_acc), jnp.sum(z_acc)


def _unnorm_posterior_mu_logpdf(mu, x, prior_loc, prior_scale, k):
    mu = jnp.asarray(mu)
    x = jnp.asarray(x)
    if mu.ndim == 0:
        loglik = jnp.sum(t.logpdf(x, df=k, loc=mu, scale=1.0))
    else:
        loglik = jnp.sum(t.logpdf(x[:, None], df=k, loc=mu[None, :], scale=1.0), axis=0)
    return loglik + norm.logpdf(mu, loc=prior_loc, scale=prior_scale)


@jit
def _update_mu_mh(key, mu_current, x_current, sigma_mu, prior_loc, prior_scale, k):
    key_prop, key_u = random.split(key)
    mu_cand = mu_current + sigma_mu * random.normal(key_prop)
    log_cur = _unnorm_posterior_mu_logpdf(mu_current, x_current, prior_loc, prior_scale, k)
    log_cand = _unnorm_posterior_mu_logpdf(mu_cand, x_current, prior_loc, prior_scale, k)
    log_alpha = jnp.where(jnp.isfinite(log_cand - log_cur), log_cand - log_cur, -jnp.inf)
    accept = jnp.log(random.uniform(key_u, minval=EPS_U, maxval=1.0)) < log_alpha
    return jnp.where(accept, mu_cand, mu_current), accept


def run_gibbs(key, mu_star, params, verbose=True):
    """Two-step Gibbs: (1) mu | x MH, (2) x | mu, MLE=mu_star.

    Note: For Cauchy (k=1) with small n (< 20), instability when outliers appear
    in the augmented data is expected: the Cauchy has no finite variance and the
    full conditional for x can occasionally impute large values, which then
    influence the next mu update. Mitigations: use a less flat prior (smaller
    prior_std) to regularize mu; use larger n when possible; reduce
    proposal_std_mu to limit large mu jumps; or increase burn-in.
    """
    T = int(params["num_iterations_T"])
    n = int(params["n"])
    k = params["k"]
    mus = jnp.zeros(T + 1)
    xs = jnp.zeros((T + 1, n))
    x0 = jnp.ones(n) * mu_star
    mus = mus.at[0].set(mu_star)
    xs = xs.at[0, :].set(x0)
    total_pairs = T * (n // 2)
    mu_acc, pair_acc, z_acc = 0, 0, 0

    iters = range(1, T + 1)
    if verbose:
        iters = tqdm(iters, desc="Gibbs (Student)")
    for t in iters:
        key, key_mu, key_x = random.split(key, 3)
        x_cur = xs[t - 1]
        mu_new, acc_mu = _update_mu_mh(
            key_mu, mus[t - 1], x_cur,
            params["proposal_std_mu"], params["prior_mean"], params["prior_std"], k
        )
        mus = mus.at[t].set(mu_new)
        mu_acc += int(acc_mu)
        x_new, npairs, nz = _update_x_full(key_x, x_cur, mu_new, mu_star, k, params["proposal_std_z"])
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
    k = params["k"]
    mus = jnp.zeros(T + 1)
    mus = mus.at[0].set(jnp.median(x))
    mu_acc = 0
    iters = range(1, T + 1)
    if verbose:
        iters = tqdm(iters, desc="Full-data MH (Student)")
    for t in iters:
        key, key_mu = random.split(key)
        mu_new, acc = _update_mu_mh(
            key_mu, mus[t - 1], x,
            params["proposal_std_mu"], params["prior_mean"], params["prior_std"], k
        )
        mus = mus.at[t].set(mu_new)
        mu_acc += int(acc)
    return {"mu_chain": mus, "mu_acceptance_rate": mu_acc / T}
