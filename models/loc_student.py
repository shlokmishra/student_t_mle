"""
Student-t location model (scale=1, df=k). MLE from score equation; Gibbs via psi(y)=y/(k+y^2).
"""

import numpy as np
from scipy.optimize import root_scalar
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random, vmap, jit, lax
from jax.scipy.stats import t, norm, truncnorm
from jax.scipy.special import logsumexp
from numba import njit, prange
from tqdm import tqdm
import math
from functools import partial

EPS_Z = 1e-12
EPS_U = 1e-12
EPS_DIV = 1e-12


def _initial_x(mu_star, n, k, params):
    init = str(params.get("initialization", "central"))
    if init == "central":
        return jnp.ones(n) * mu_star
    if init == "tail_heavy":
        amp = float(params.get("initialization_tail_amplitude", max(5.0, 5.0 * float(np.sqrt(k)))))
        vals = [mu_star + amp, mu_star - amp] * (n // 2)
        if n % 2:
            vals.append(mu_star)
        return jnp.asarray(vals[:n], dtype=float)
    if init == "random":
        rng = np.random.default_rng(int(params.get("initialization_seed", 0)))
        vals = []
        for _ in range(n // 2):
            amp = float(np.exp(rng.normal(0.0, 1.0)))
            vals.extend([mu_star + amp, mu_star - amp])
        if n % 2:
            vals.append(mu_star)
        rng.shuffle(vals)
        return jnp.asarray(vals[:n], dtype=float)
    raise ValueError(f"Unknown initialization: {init}")


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


def _gibbs_transition(key, mu_current, x_current, mu_star, k, sigma_mu, sigma_z, prior_loc, prior_scale):
    key, key_mu, key_x = random.split(key, 3)
    mu_new, acc_mu = _update_mu_mh(key_mu, mu_current, x_current, sigma_mu, prior_loc, prior_scale, k)
    x_new, pair_acc, z_acc = _update_x_full(key_x, x_current, mu_new, mu_star, k, sigma_z)
    return key, mu_new, x_new, acc_mu, pair_acc, z_acc


@jit
def _gibbs_scan_step(carry, _):
    key, mu_current, x_current, mu_star, k, sigma_mu, sigma_z, prior_loc, prior_scale = carry
    key, mu_new, x_new, acc_mu, pair_acc, z_acc = _gibbs_transition(
        key, mu_current, x_current, mu_star, k, sigma_mu, sigma_z, prior_loc, prior_scale
    )
    next_carry = (key, mu_new, x_new, mu_star, k, sigma_mu, sigma_z, prior_loc, prior_scale)
    return next_carry, (mu_new, x_new, acc_mu, pair_acc, z_acc)


@partial(jit, static_argnames=("T",))
def _run_gibbs_jax_scan_kernel(key, x0, mu_star, k, sigma_mu, sigma_z, prior_loc, prior_scale, T):
    carry = (
        key,
        jnp.asarray(mu_star, dtype=float),
        x0,
        jnp.asarray(mu_star, dtype=float),
        jnp.asarray(k, dtype=float),
        jnp.asarray(sigma_mu, dtype=float),
        jnp.asarray(sigma_z, dtype=float),
        jnp.asarray(prior_loc, dtype=float),
        jnp.asarray(prior_scale, dtype=float),
    )
    _, draws = lax.scan(_gibbs_scan_step, carry, xs=None, length=T)
    mus_tail, xs_tail, mu_acc, pair_acc, z_acc = draws
    mus = jnp.concatenate([jnp.asarray([mu_star], dtype=float), mus_tail])
    xs = jnp.concatenate([x0[None, :], xs_tail], axis=0)
    return mus, xs, jnp.sum(mu_acc), jnp.sum(pair_acc), jnp.sum(z_acc)


def _run_gibbs_jax_scan(key, mu_star, params):
    T = int(params["num_iterations_T"])
    n = int(params["n"])
    k = float(params["k"])
    x0 = _initial_x(mu_star, n, k, params)
    mus, xs, mu_acc, pair_acc, z_acc = _run_gibbs_jax_scan_kernel(
        key,
        x0,
        float(mu_star),
        k,
        float(params["proposal_std_mu"]),
        float(params["proposal_std_z"]),
        float(params["prior_mean"]),
        float(params["prior_std"]),
        T,
    )
    return {
        "mu_chain": mus,
        "x_chain": xs,
        "mu_acceptance_count": mu_acc,
        "pair_acceptance_count": pair_acc,
        "z_acceptance_count": z_acc,
    }


@njit(cache=True)
def _numba_z_support(k):
    bound = 1.0 / (2.0 * math.sqrt(k))
    return -bound + EPS_Z, bound - EPS_Z


@njit(cache=True)
def _numba_psi(y, k):
    return y / (k + y * y)


@njit(cache=True)
def _numba_student_logpdf(y, loc, k):
    half = 0.5
    return (
        math.lgamma((k + 1.0) * half)
        - math.lgamma(k * half)
        - half * math.log(k * math.pi)
        - ((k + 1.0) * half) * math.log1p(((y - loc) * (y - loc)) / k)
    )


@njit(cache=True)
def _numba_norm_logpdf(x, loc, scale):
    z = (x - loc) / scale
    return -0.5 * z * z - math.log(scale) - 0.5 * math.log(2.0 * math.pi)


@njit(cache=True)
def _numba_norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@njit(cache=True)
def _numba_truncnorm_logpdf(x, loc, scale, low, high):
    a = (low - loc) / scale
    b = (high - loc) / scale
    normalizer = _numba_norm_cdf(b) - _numba_norm_cdf(a)
    if normalizer <= 0.0 or not math.isfinite(normalizer):
        return -math.inf
    return _numba_norm_logpdf(x, loc, scale) - math.log(normalizer)


@njit(cache=True)
def _numba_psi_inverse(z, k):
    z_min, z_max = _numba_z_support(k)
    zc = min(max(z, z_min), z_max)
    if abs(zc) < EPS_DIV:
        return 0.0, 0.0
    tval = 2.0 * math.sqrt(k) * zc
    discr = max(1.0 - tval * tval, 0.0)
    root = math.sqrt(discr)
    denom = 2.0 * zc
    y_plus = (1.0 + root) / denom
    y_minus = (1.0 - root) / denom
    if not math.isfinite(y_plus):
        y_plus = 0.0
    if not math.isfinite(y_minus):
        y_minus = 0.0
    return min(y_minus, y_plus), max(y_minus, y_plus)


@njit(cache=True)
def _numba_log_psi_prime_abs(y, k):
    return math.log(abs(k - y * y) + 1e-30) - 2.0 * math.log(k + y * y)


@njit(cache=True)
def _numba_q_logpdf(z, mu_current, mu_star, k):
    z_min, z_max = _numba_z_support(k)
    if not (z > z_min and z < z_max):
        return -math.inf
    loc = mu_current - mu_star
    y_lo, y_hi = _numba_psi_inverse(z, k)
    a = _numba_student_logpdf(y_lo, loc, k) - _numba_log_psi_prime_abs(y_lo, k)
    b = _numba_student_logpdf(y_hi, loc, k) - _numba_log_psi_prime_abs(y_hi, k)
    m = max(a, b)
    if not math.isfinite(m):
        return -math.inf
    return m + math.log(math.exp(a - m) + math.exp(b - m))


@njit(cache=True)
def _numba_q_tilde_logpdf(z, delta, mu_current, mu_star, k):
    return _numba_q_logpdf(z, mu_current, mu_star, k) + _numba_q_logpdf(delta - z, mu_current, mu_star, k)


@njit(cache=True)
def _numba_sample_truncated_normal(loc, scale, low, high):
    for _ in range(10000):
        cand = loc + scale * np.random.normal()
        if cand >= low and cand <= high:
            return cand
    return min(max(loc, low), high)


@njit(parallel=True, cache=True)
def _numba_update_pairs(x_current, perm, mu_current, mu_star, k, sigma_z, x_out):
    n_pairs = perm.shape[0] // 2
    pair_acc = 0
    z_acc = 0
    for p in prange(n_pairs):
        idx_i = perm[2 * p]
        idx_j = perm[2 * p + 1]
        xi = x_current[idx_i]
        xj = x_current[idx_j]
        yi = xi - mu_star
        yj = xj - mu_star
        zi = _numba_psi(yi, k)
        zj = _numba_psi(yj, k)
        delta = zi + zj
        low, high = _numba_z_support(k)
        low_int = max(low, delta - high)
        high_int = min(high, delta - low)
        if low_int >= high_int:
            x_out[idx_i] = xi
            x_out[idx_j] = xj
            continue

        z_prop = _numba_sample_truncated_normal(zi, sigma_z, low_int, high_int)
        log_k_cur = _numba_truncnorm_logpdf(z_prop, zi, sigma_z, low_int, high_int)
        log_k_back = _numba_truncnorm_logpdf(zi, z_prop, sigma_z, low_int, high_int)
        log_cur = _numba_q_tilde_logpdf(zi, delta, mu_current, mu_star, k)
        log_prop = _numba_q_tilde_logpdf(z_prop, delta, mu_current, mu_star, k)
        log_alpha = log_prop - log_cur + log_k_back - log_k_cur
        if math.isfinite(log_alpha) and math.log(max(np.random.random(), EPS_U)) < log_alpha:
            zi_tilde = z_prop
            z_acc += 1
        else:
            zi_tilde = zi
        zj_tilde = delta - zi_tilde
        if not (zj_tilde > low and zj_tilde < high):
            x_out[idx_i] = xi
            x_out[idx_j] = xj
            continue

        yi_lo, yi_hi = _numba_psi_inverse(zi_tilde, k)
        yj_lo, yj_hi = _numba_psi_inverse(zj_tilde, k)
        loc = mu_current - mu_star
        wi_lo = _numba_student_logpdf(yi_lo, loc, k) - _numba_log_psi_prime_abs(yi_lo, k)
        wi_hi = _numba_student_logpdf(yi_hi, loc, k) - _numba_log_psi_prime_abs(yi_hi, k)
        wj_lo = _numba_student_logpdf(yj_lo, loc, k) - _numba_log_psi_prime_abs(yj_lo, k)
        wj_hi = _numba_student_logpdf(yj_hi, loc, k) - _numba_log_psi_prime_abs(yj_hi, k)
        mi = max(wi_lo, wi_hi)
        mj = max(wj_lo, wj_hi)
        if math.isfinite(mi):
            pi_lo = math.exp(wi_lo - mi)
            pi_hi = math.exp(wi_hi - mi)
            yi_new = yi_lo if np.random.random() < pi_lo / (pi_lo + pi_hi) else yi_hi
        else:
            yi_new = yi_lo
        if math.isfinite(mj):
            pj_lo = math.exp(wj_lo - mj)
            pj_hi = math.exp(wj_hi - mj)
            yj_new = yj_lo if np.random.random() < pj_lo / (pj_lo + pj_hi) else yj_hi
        else:
            yj_new = yj_lo
        x_out[idx_i] = yi_new + mu_star
        x_out[idx_j] = yj_new + mu_star
        pair_acc += 1
    return pair_acc, z_acc


def _unnorm_posterior_mu_logpdf(mu, x, prior_loc, prior_scale, k):
    mu = jnp.asarray(mu)
    x = jnp.asarray(x)
    if mu.ndim == 0:
        loglik = jnp.sum(t.logpdf(x, df=k, loc=mu, scale=1.0))
    else:
        loglik = jnp.sum(t.logpdf(x[:, None], df=k, loc=mu[None, :], scale=1.0), axis=0)
    return loglik + norm.logpdf(mu, loc=prior_loc, scale=prior_scale)


def _numba_mu_logpdf(mu, x, prior_loc, prior_scale, k):
    x = np.asarray(x, dtype=float)
    k = float(k)
    log_const = math.lgamma((k + 1.0) / 2.0) - math.lgamma(k / 2.0) - 0.5 * math.log(k * math.pi)
    y = x - float(mu)
    loglik = float(np.sum(log_const - ((k + 1.0) / 2.0) * np.log1p((y * y) / k)))
    prior = -0.5 * ((mu - prior_loc) / prior_scale) ** 2 - math.log(prior_scale) - 0.5 * math.log(2.0 * math.pi)
    return float(loglik + prior)


def _run_gibbs_numba(key, mu_star, params):
    T = int(params["num_iterations_T"])
    n = int(params["n"])
    k = float(params["k"])
    sigma_mu = float(params["proposal_std_mu"])
    sigma_z = float(params["proposal_std_z"])
    prior_loc = float(params["prior_mean"])
    prior_scale = float(params["prior_std"])
    seed = int(np.asarray(random.randint(key, (), minval=0, maxval=2**31 - 1)))
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    mus = np.zeros(T + 1, dtype=float)
    xs = np.zeros((T + 1, n), dtype=float)
    mus[0] = float(mu_star)
    xs[0, :] = np.asarray(_initial_x(mu_star, n, k, params), dtype=float)
    mu_acc = 0
    pair_acc = 0
    z_acc = 0

    for t_idx in range(1, T + 1):
        x_cur = xs[t_idx - 1]
        mu_cur = mus[t_idx - 1]
        mu_cand = mu_cur + sigma_mu * rng.normal()
        log_cur = _numba_mu_logpdf(mu_cur, x_cur, prior_loc, prior_scale, k)
        log_cand = _numba_mu_logpdf(mu_cand, x_cur, prior_loc, prior_scale, k)
        log_alpha = log_cand - log_cur if np.isfinite(log_cand - log_cur) else -np.inf
        if np.log(max(rng.random(), EPS_U)) < log_alpha:
            mu_new = mu_cand
            mu_acc += 1
        else:
            mu_new = mu_cur
        mus[t_idx] = mu_new

        perm = rng.permutation(n).astype(np.int64)
        x_out = x_cur.copy()
        completed, accepted_z = _numba_update_pairs(x_cur, perm, mu_new, float(mu_star), k, sigma_z, x_out)
        xs[t_idx, :] = x_out
        pair_acc += int(completed)
        z_acc += int(accepted_z)

    return {
        "mu_chain": mus,
        "x_chain": xs,
        "mu_acceptance_count": mu_acc,
        "pair_acceptance_count": pair_acc,
        "z_acceptance_count": z_acc,
    }


@jit
def _update_mu_mh(key, mu_current, x_current, sigma_mu, prior_loc, prior_scale, k):
    key_prop, key_u = random.split(key)
    mu_cand = mu_current + sigma_mu * random.normal(key_prop)
    log_cur = _unnorm_posterior_mu_logpdf(mu_current, x_current, prior_loc, prior_scale, k)
    log_cand = _unnorm_posterior_mu_logpdf(mu_cand, x_current, prior_loc, prior_scale, k)
    log_alpha = jnp.where(jnp.isfinite(log_cand - log_cur), log_cand - log_cur, -jnp.inf)
    accept = jnp.log(random.uniform(key_u, minval=EPS_U, maxval=1.0)) < log_alpha
    return jnp.where(accept, mu_cand, mu_current), accept


def _record_gibbs_costs(cost_ledger, T, n, mu_acc, pair_acc, backend):
    if cost_ledger is None:
        return
    attempted = T * (n // 2)
    q_logpdf_evals = 4 * attempted
    cost_ledger.inc("iterations", T)
    cost_ledger.inc("sweep_count", T)
    cost_ledger.inc("mu_mh_proposals", T)
    cost_ledger.inc("student_logpdf_evals", 2 * T * n)
    cost_ledger.inc("prior_logpdf_evals", 2 * T)
    cost_ledger.inc("mu_mh_accepts", int(mu_acc))
    cost_ledger.inc("pair_updates_attempted", attempted)
    cost_ledger.inc("pair_updates_completed", int(pair_acc))
    cost_ledger.inc("pair_rejections", attempted - int(pair_acc))
    cost_ledger.inc("constraint_evals", attempted)
    cost_ledger.inc("pair_grid_evals", q_logpdf_evals)
    cost_ledger.inc("pair_inverse_branch_evals", 2 * q_logpdf_evals + 4 * int(pair_acc))
    cost_ledger.inc("pair_weight_evals", 4 * int(pair_acc))
    cost_ledger.inc("student_logpdf_evals", 2 * q_logpdf_evals + 4 * int(pair_acc))
    cost_ledger.set("iterations", T)
    cost_ledger.set("mu_mh_accepts", int(mu_acc))
    cost_ledger.set("pair_updates_completed", int(pair_acc))
    cost_ledger.set("gibbs_backend", backend)


def run_gibbs(key, mu_star, params, verbose=True, cost_ledger=None):
    """Two-step Gibbs: (1) mu | x MH, (2) x | mu, MLE=mu_star.

    Note: For Cauchy (k=1) with small n (< 20), instability when outliers appear
    in the augmented data is expected: the Cauchy has no finite variance and the
    full conditional for x can occasionally impute large values, which then
    influence the next mu update. Mitigations: use a less flat prior (smaller
    prior_std) to regularize mu; use larger n when possible; reduce
    proposal_std_mu to limit large mu jumps; or increase burn-in.
    """
    backend = str(params.get("gibbs_backend", "jax_loop"))
    if backend not in {"jax_loop", "jax_scan", "numba"}:
        raise ValueError(f"Unknown Student Gibbs backend: {backend}")
    T = int(params["num_iterations_T"])
    n = int(params["n"])
    k = params["k"]
    total_pairs = T * (n // 2)

    if backend == "jax_scan":
        out = _run_gibbs_jax_scan(key, mu_star, params)
        mu_acc = int(out["mu_acceptance_count"])
        pair_acc = int(out["pair_acceptance_count"])
        z_acc = int(out["z_acceptance_count"])
        _record_gibbs_costs(cost_ledger, T, n, mu_acc, pair_acc, backend)
        return {
            "mu_chain": out["mu_chain"],
            "x_chain": out["x_chain"],
            "mu_acceptance_rate": mu_acc / T,
            "pair_acceptance_rate": pair_acc / total_pairs,
            "z_acceptance_rate": z_acc / total_pairs,
            "gibbs_backend": backend,
        }

    if backend == "numba":
        out = _run_gibbs_numba(key, mu_star, params)
        mu_acc = int(out["mu_acceptance_count"])
        pair_acc = int(out["pair_acceptance_count"])
        z_acc = int(out["z_acceptance_count"])
        _record_gibbs_costs(cost_ledger, T, n, mu_acc, pair_acc, backend)
        return {
            "mu_chain": out["mu_chain"],
            "x_chain": out["x_chain"],
            "mu_acceptance_rate": mu_acc / T,
            "pair_acceptance_rate": pair_acc / total_pairs,
            "z_acceptance_rate": z_acc / total_pairs,
            "gibbs_backend": backend,
        }

    mus = jnp.zeros(T + 1)
    xs = jnp.zeros((T + 1, n))
    x0 = _initial_x(mu_star, n, k, params)
    mus = mus.at[0].set(mu_star)
    xs = xs.at[0, :].set(x0)
    mu_acc, pair_acc, z_acc = 0, 0, 0

    iters = range(1, T + 1)
    if verbose:
        iters = tqdm(iters, desc="Gibbs (Student)")
    for t in iters:
        key, key_mu, key_x = random.split(key, 3)
        x_cur = xs[t - 1]
        if cost_ledger is not None:
            cost_ledger.inc("iterations")
            cost_ledger.inc("sweep_count")
            cost_ledger.inc("mu_mh_proposals")
            # Current/candidate log posterior for mu: two Student-t sums over n plus two prior logpdfs.
            cost_ledger.inc("student_logpdf_evals", 2 * n)
            cost_ledger.inc("prior_logpdf_evals", 2)
        mu_new, acc_mu = _update_mu_mh(
            key_mu, mus[t - 1], x_cur,
            params["proposal_std_mu"], params["prior_mean"], params["prior_std"], k
        )
        mus = mus.at[t].set(mu_new)
        mu_acc += int(acc_mu)
        if cost_ledger is not None:
            cost_ledger.inc("mu_mh_accepts", int(acc_mu))
        x_new, npairs, nz = _update_x_full(key_x, x_cur, mu_new, mu_star, k, params["proposal_std_z"])
        xs = xs.at[t, :].set(x_new)
        pair_acc += int(npairs)
        z_acc += int(nz)
        if cost_ledger is not None:
            attempted = n // 2
            completed = int(npairs)
            q_logpdf_evals = 4 * attempted
            cost_ledger.inc("pair_updates_attempted", attempted)
            cost_ledger.inc("pair_updates_completed", completed)
            cost_ledger.inc("pair_rejections", attempted - completed)
            cost_ledger.inc("constraint_evals", attempted)
            cost_ledger.inc("pair_grid_evals", q_logpdf_evals)
            # Each q(z) evaluates two inverse branches and two Student-t branch densities.
            cost_ledger.inc("pair_inverse_branch_evals", 2 * q_logpdf_evals + 4 * completed)
            cost_ledger.inc("pair_weight_evals", 4 * completed)
            cost_ledger.inc("student_logpdf_evals", 2 * q_logpdf_evals + 4 * completed)

    result = {
        "mu_chain": mus,
        "x_chain": xs,
        "mu_acceptance_rate": mu_acc / T,
        "pair_acceptance_rate": pair_acc / total_pairs,
        "z_acceptance_rate": z_acc / total_pairs,
        "gibbs_backend": backend,
    }
    if cost_ledger is not None:
        cost_ledger.set("iterations", T)
        cost_ledger.set("mu_mh_accepts", mu_acc)
        cost_ledger.set("pair_updates_completed", pair_acc)
        cost_ledger.set("gibbs_backend", backend)
    return result


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
