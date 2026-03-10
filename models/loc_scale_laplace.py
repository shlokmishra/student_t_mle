"""
Laplace LOCATION–SCALE model with UNKNOWN (mu,b).

We target the "insufficient-statistic posterior" via data augmentation:
    π(mu,b,x | MLE(x)=(mu_star,b_star))
      ∝  prior(mu) prior(b)  *  ∏_i Laplace(x_i | mu,b)
         *  1{ median(x)=mu_star } * 1{ (1/n)∑|x_i-mu_star| = b_star }.

Algorithm (MH-within-Gibbs):
1) (mu,b) | x : MH on (mu, log b) targeting prior(mu)prior(b) * ∏ Laplace(x_i|mu,b)
2) x | (mu,b), constraints : pairwise MH updates that preserve the constraints exactly:
     - keep each coordinate on its side of mu_star  (=> preserves median constraint if init_x correct)
     - keep delta = |xi-mu_star|+|xj-mu_star| fixed (=> preserves ∑|x-mu_star| = n*b_star)

You asked:
- uniformize the whole code
- add init_x(mu_star,b_star,n) producing x0 with MLE exactly (mu_star,b_star)
- run_gibbs uses x0=init_x, includes a b_chain, and x-update calls _update_x_full
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random, jit, vmap
from jax.scipy.stats import norm
from tqdm import tqdm

EPS_U = 1e-12
EPS_B = 1e-12
EPS_DELTA = 1e-30


# ============================================================
# MLE + init
# ============================================================

def get_mle(data):
    """Laplace loc-scale MLE: (median, mean abs dev from median)."""
    x = np.asarray(data)
    mu_hat = float(np.median(x))
    b_hat = float(np.mean(np.abs(x - mu_hat)))
    b_hat = max(b_hat, EPS_B)
    return mu_hat, b_hat


def init_x(mu_star, b_star, n):
    """
    Build x0 with EXACT MLE (median, mean abs dev) = (mu_star, b_star).

    - n even: n/2 at mu*-b*, n/2 at mu*+b*  -> median=mu*, mean abs dev=b*
    - n odd : one at mu*, remaining split with radius r = n*b*/(n-1)
              -> sum abs dev = (n-1)r = n b* -> mean abs dev=b*, median=mu*
    """
    mu_star = float(mu_star)
    b_star = float(b_star)
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive.")
    if b_star <= 0:
        raise ValueError("b_star must be > 0.")

    if n % 2 == 0:
        half = n // 2
        return jnp.concatenate(
            [(mu_star - b_star) * jnp.ones((half,)),
             (mu_star + b_star) * jnp.ones((half,))],
            axis=0
        )

    half = (n - 1) // 2
    r = (n * b_star) / (n - 1)
    return jnp.concatenate(
        [(mu_star - r) * jnp.ones((half,)),
         jnp.array([mu_star]),
         (mu_star + r) * jnp.ones((half,))],
        axis=0
    )


# ============================================================
# Laplace primitives
# ============================================================

@jit
def _laplace_logpdf(x, loc, b):
    return -jnp.log(2.0 * b) - jnp.abs(x - loc) / b


def _log_prior_mu(mu, prior_mean, prior_std):
    return float(norm.logpdf(mu, loc=prior_mean, scale=prior_std))


def _log_prior_logb(logb, prior_logb_mean, prior_logb_std):
    """Prior on log b (Gaussian)."""
    return float(norm.logpdf(logb, loc=prior_logb_mean, scale=prior_logb_std))


# ============================================================
# x-step (constraints) : pairwise partial resampling
# ============================================================

@jit
def _safe_sign(y):
    # keep side of mu_star fixed; if exactly 0, choose +1
    return jnp.where(y >= 0.0, 1.0, -1.0)

@jit
def _logit(u):
    u = jnp.clip(u, EPS_U, 1.0 - EPS_U)
    return jnp.log(u) - jnp.log1p(-u)

@jit
def _sigmoid(z):
    return jax.nn.sigmoid(z)

@jit
def _log_dr_dz(delta, z):
    # r = delta * sigmoid(z), dr/dz = delta*s*(1-s)
    s = _sigmoid(z)
    return jnp.log(jnp.maximum(delta, EPS_DELTA)) + jnp.log(jnp.maximum(s * (1.0 - s), 1e-30))


@jit
def _pair_log_target_from_z(z, xi, xj, mu_current, b_current, mu_star):
    """
    Log target for z (parametrizing r_i in [0,delta]) with Jacobian.
    IMPORTANT: likelihood uses (mu_current, b_current),
               constraints are around mu_star via delta and fixed signs.
    """
    yi = xi - mu_star
    yj = xj - mu_star
    si = _safe_sign(yi)
    sj = _safe_sign(yj)
    ri = jnp.abs(yi)
    rj = jnp.abs(yj)
    delta = jnp.maximum(ri + rj, EPS_DELTA)

    u = _sigmoid(z)
    ri_new = delta * u
    rj_new = delta - ri_new
    xi_new = mu_star + si * ri_new
    xj_new = mu_star + sj * rj_new

    loglik = _laplace_logpdf(xi_new, loc=mu_current, b=b_current) + _laplace_logpdf(xj_new, loc=mu_current, b=b_current)
    logjac = _log_dr_dz(delta, z)
    return loglik + logjac


@jit
def _update_xi_xj_one(key, xi, xj, mu_current, b_current, mu_star, sigma_z):
    """
    One pair MH update preserving:
      - side of mu_star for each coordinate (=> median constraint if init_x correct)
      - delta = |xi-mu_star|+|xj-mu_star| (=> preserves sum abs dev constraint)
    Returns: xi_new, xj_new, pair_acc, z_acc
    """
    key_prop, key_u = random.split(key, 2)

    yi = xi - mu_star
    yj = xj - mu_star
    ri = jnp.abs(yi)
    rj = jnp.abs(yj)
    delta = ri + rj

    def do_nop(_):
        return xi, xj, False, False

    def do_mh(_):
        u_cur = jnp.clip(ri / jnp.maximum(delta, EPS_DELTA), EPS_U, 1.0 - EPS_U)
        z_cur = _logit(u_cur)
        z_prop = z_cur + sigma_z * random.normal(key_prop)

        log_cur = _pair_log_target_from_z(z_cur, xi, xj, mu_current, b_current, mu_star)
        log_prop = _pair_log_target_from_z(z_prop, xi, xj, mu_current, b_current, mu_star)

        log_alpha = jnp.where(jnp.isfinite(log_prop - log_cur), log_prop - log_cur, -jnp.inf)
        accept = jnp.log(random.uniform(key_u, shape=(), minval=EPS_U, maxval=1.0)) < log_alpha

        si = _safe_sign(yi)
        sj = _safe_sign(yj)
        u = _sigmoid(z_prop)
        ri_new = delta * u
        rj_new = delta - ri_new
        xi_new = mu_star + si * ri_new
        xj_new = mu_star + sj * rj_new

        xi_out = jnp.where(accept, xi_new, xi)
        xj_out = jnp.where(accept, xj_new, xj)
        return xi_out, xj_out, accept, accept

    return jax.lax.cond(delta > EPS_DELTA, do_mh, do_nop, operand=None)


@jit
def _update_x_full(key, x_current, mu_current, b_current, mu_star, sigma_z):
    """
    Pairwise MH updates over a random permutation.
    If n odd, last unpaired element is left unchanged.
    """
    n = x_current.shape[0]
    key_perm, key_pairs = random.split(key)
    perm = random.permutation(key_perm, n)
    x_perm = x_current[perm]

    n_pairs = n // 2
    xis = x_perm[: 2 * n_pairs : 2]
    xjs = x_perm[1 : 2 * n_pairs : 2]

    keys = random.split(key_pairs, n_pairs)
    batch = vmap(_update_xi_xj_one, in_axes=(0, 0, 0, None, None, None, None))
    xis_new, xjs_new, pair_acc, z_acc = batch(keys, xis, xjs, mu_current, b_current, mu_star, sigma_z)

    x_new_perm = x_perm
    x_new_perm = x_new_perm.at[: 2 * n_pairs : 2].set(xis_new)
    x_new_perm = x_new_perm.at[1 : 2 * n_pairs : 2].set(xjs_new)

    x_new = x_new_perm[jnp.argsort(perm)]
    return x_new, jnp.sum(pair_acc), jnp.sum(z_acc)


# ============================================================
# (mu,b) step: joint MH on (mu, log b) given full x
# ============================================================

@jit
def _unnorm_post_mulogb(mu, logb, x, prior_mean, prior_std, prior_logb_mean, prior_logb_std):
    """
    Unnormalized log posterior in (mu,logb):
      loglik(x|mu,b) + log prior(mu) + log prior(logb) + log|db/dlogb|
    with b = exp(logb), Jacobian term is +logb.
    """
    b = jnp.exp(logb)
    loglik = jnp.sum(_laplace_logpdf(x, loc=mu, b=b))
    logprior_mu = norm.logpdf(mu, loc=prior_mean, scale=prior_std)
    logprior_logb = norm.logpdf(logb, loc=prior_logb_mean, scale=prior_logb_std)
    return loglik + logprior_mu + logprior_logb + logb


@jit
def _update_mulogb_mh(key, mu_cur, logb_cur, x, sigma_mu, sigma_logb,
                      prior_mean, prior_std, prior_logb_mean, prior_logb_std):
    key_mu, key_lb, key_u = random.split(key, 3)
    mu_cand = mu_cur + sigma_mu * random.normal(key_mu)
    logb_cand = logb_cur + sigma_logb * random.normal(key_lb)

    log_cur = _unnorm_post_mulogb(mu_cur, logb_cur, x, prior_mean, prior_std, prior_logb_mean, prior_logb_std)
    log_cand = _unnorm_post_mulogb(mu_cand, logb_cand, x, prior_mean, prior_std, prior_logb_mean, prior_logb_std)

    log_alpha = jnp.where(jnp.isfinite(log_cand - log_cur), log_cand - log_cur, -jnp.inf)
    accept = jnp.log(random.uniform(key_u, shape=(), minval=EPS_U, maxval=1.0)) < log_alpha

    mu_out = jnp.where(accept, mu_cand, mu_cur)
    logb_out = jnp.where(accept, logb_cand, logb_cur)
    return mu_out, logb_out, accept


# ============================================================
# Main constrained Gibbs
# ============================================================

def run_gibbs(key, mu_star, b_star, params, verbose=True):
    """
    Gibbs / MH-within-Gibbs for π(mu,b,x | MLE(x)=(mu_star,b_star)).

    params:
      - n, num_iterations_T
      - prior_mean, prior_std
      - prior_logb_mean, prior_logb_std
      - proposal_std_mu, proposal_std_logb
      - proposal_std_z   (RW std in z for pairwise x updates)
    """
    T = int(params["num_iterations_T"])
    n = int(params["n"])

    prior_mean = float(params.get("prior_mean", 0.0))
    prior_std = float(params.get("prior_std", 5.0))
    prior_logb_mean = float(params.get("prior_logb_mean", 0.0))
    prior_logb_std = float(params.get("prior_logb_std", 1.0))

    sigma_mu = float(params.get("proposal_std_mu", 0.3))
    sigma_logb = float(params.get("proposal_std_logb", 0.2))
    sigma_z = float(params.get("proposal_std_z", 0.7))

    mu_star = float(mu_star)
    b_star = float(b_star)

    # chains
    mus = jnp.zeros((T + 1,))
    bs = jnp.zeros((T + 1,))
    xs = jnp.zeros((T + 1, n))

    # init x exactly on constraints
    x0 = init_x(mu_star, b_star, n)

    # init parameters (reasonable default: start at MLE; b init at b_star)
    mus = mus.at[0].set(mu_star)
    bs = bs.at[0].set(max(b_star, EPS_B))
    xs = xs.at[0, :].set(x0)

    mu_b_acc = 0
    pair_acc = 0
    z_acc = 0
    total_pairs = T * (n // 2)

    iters = range(1, T + 1)
    if verbose:
        iters = tqdm(iters, desc="Gibbs (Laplace loc-scale | MLE)")

    for t in iters:
        key, key_par, key_x = random.split(key, 3)

        x_cur = xs[t - 1]
        mu_cur = mus[t - 1]
        b_cur = bs[t - 1]
        logb_cur = jnp.log(jnp.maximum(b_cur, EPS_B))

        # 1) update (mu,logb) | x
        mu_new, logb_new, acc_par = _update_mulogb_mh(
            key_par, mu_cur, logb_cur, x_cur,
            sigma_mu, sigma_logb,
            prior_mean, prior_std, prior_logb_mean, prior_logb_std
        )
        b_new = jnp.exp(logb_new)

        mus = mus.at[t].set(mu_new)
        bs = bs.at[t].set(b_new)
        mu_b_acc += int(acc_par)

        # 2) update x | (mu,b), constraints
        x_new, npairs, nz = _update_x_full(key_x, x_cur, mu_new, b_new, mu_star, sigma_z)
        xs = xs.at[t, :].set(x_new)
        pair_acc += int(npairs)
        z_acc += int(nz)

    return {
        "mu_chain": mus,
        "b_chain": bs,
        "x_chain": xs,
        "param_acceptance_rate": mu_b_acc / T,
        "pair_acceptance_rate": (pair_acc / total_pairs) if total_pairs > 0 else 0.0,
        "z_acceptance_rate": (z_acc / total_pairs) if total_pairs > 0 else 0.0,
    }