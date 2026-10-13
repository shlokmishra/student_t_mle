"""
Student-t location model (scale=1, df=k). Global MLE; Gibbs via psi(y)=y/(k+y^2).
"""

import numpy as np
from jax import random
from numba import njit
from tqdm import tqdm
import math

from models.loc.student.mu_mh import seed_from_key, update_mu_mh_from_variates, update_mu_mh_random
from models.loc.student.mle import global_mle as _global_location_mle

EPS_Z = 1e-12
EPS_U = 1e-12


def _initial_x(mu_star, n, k, params):
    init = str(params.get("initialization", "central"))
    if init == "central":
        return np.full(n, float(mu_star), dtype=float)
    if init == "tail_heavy":
        amp = float(params.get("initialization_tail_amplitude", max(5.0, 5.0 * float(np.sqrt(k)))))
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
    """Selected/global location MLE for fixed Student df and unit scale."""
    return _global_location_mle(data, params["k"])


def sample_data(key, params, loc=0.0):
    """Sample data from Student-t(loc, scale=1, df=k)."""
    k = params["k"]
    n = params["n"]
    return random.t(key, df=k, shape=(n,)) + loc


def get_benchmark_mle_samples(key, params, num_simulations=10000, verbose=False):
    """Samples from p(hat_theta | theta=0)."""
    k, n = params["k"], params["n"]
    all_data = np.array(random.t(key, df=k, shape=(num_simulations, n)))
    data_iter = tqdm(all_data, desc="Benchmark MLE samples") if verbose else all_data
    return np.array([get_mle(data, params) for data in data_iter])

@njit(cache=True)
def z_support(k):
    bound = 1.0 / (2.0 * math.sqrt(k))
    return -bound + EPS_Z, bound - EPS_Z


@njit(cache=True)
def psi(y, k):
    return y / (k + y * y)


@njit(cache=True)
def student_logpdf_no_const(y, loc, k):
    return -0.5 * (k + 1.0) * math.log1p(((y - loc) * (y - loc)) / k)


@njit(cache=True)
def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@njit(cache=True)
def norm_ppf(p):
    # Acklam's rational approximation for the inverse standard normal CDF.
    p = min(max(p, EPS_U), 1.0 - EPS_U)
    a1 = -3.969683028665376e01
    a2 = 2.209460984245205e02
    a3 = -2.759285104469687e02
    a4 = 1.383577518672690e02
    a5 = -3.066479806614716e01
    a6 = 2.506628277459239e00
    b1 = -5.447609879822406e01
    b2 = 1.615858368580409e02
    b3 = -1.556989798598866e02
    b4 = 6.680131188771972e01
    b5 = -1.328068155288572e01
    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e00
    c4 = -2.549732539343734e00
    c5 = 4.374664141464968e00
    c6 = 2.938163982698783e00
    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e00
    d4 = 3.754408661907416e00
    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)


@njit(cache=True)
def truncnorm_log_normalizer(loc, scale, low, high):
    a = (low - loc) / scale
    b = (high - loc) / scale
    normalizer = norm_cdf(b) - norm_cdf(a)
    if normalizer <= 0.0 or not math.isfinite(normalizer):
        return -math.inf
    return math.log(normalizer)


@njit(cache=True)
def psi_inverse(z, k):
    z_min, z_max = z_support(k)
    zc = min(max(z, z_min), z_max)
    if zc == 0.0:
        return 0.0, 0.0
    tval = 2.0 * math.sqrt(k) * zc
    discr = max(1.0 - tval * tval, 0.0)
    root = math.sqrt(discr)
    # The direct ``1 - root`` formula catastrophically cancels near z=0.
    # These are the same quadratic roots, written stably for the inner branch.
    y_inner = 2.0 * k * zc / (1.0 + root)
    y_outer = (1.0 + root) / (2.0 * zc)
    return min(y_inner, y_outer), max(y_inner, y_outer)


@njit(cache=True)
def log_psi_prime_abs(y, k):
    return math.log(abs(k - y * y) + 1e-30) - 2.0 * math.log(k + y * y)


@njit(cache=True)
def inverse_branch_log_weight(y, loc, k):
    """Log density with respect to dz on one inverse branch."""
    return student_logpdf_no_const(y, loc, k) - log_psi_prime_abs(y, k)


@njit(cache=True)
def q_logpdf_no_const(z, mu_current, mu_star, k, z_min, z_max):
    if not (z > z_min and z < z_max):
        return -math.inf
    loc = mu_current - mu_star
    y_lo, y_hi = psi_inverse(z, k)
    a = inverse_branch_log_weight(y_lo, loc, k)
    b = inverse_branch_log_weight(y_hi, loc, k)
    m = max(a, b)
    if not math.isfinite(m):
        return -math.inf
    return m + math.log(math.exp(a - m) + math.exp(b - m))


@njit(cache=True)
def q_tilde_logpdf_no_const(z, delta, mu_current, mu_star, k, z_min, z_max):
    return q_logpdf_no_const(z, mu_current, mu_star, k, z_min, z_max) + q_logpdf_no_const(
        delta - z, mu_current, mu_star, k, z_min, z_max
    )


@njit(cache=True)
def sample_truncated_normal(loc, scale, low, high):
    a = (low - loc) / scale
    b = (high - loc) / scale
    cdf_low = norm_cdf(a)
    cdf_high = norm_cdf(b)
    mass = cdf_high - cdf_low
    if mass <= 0.0 or not math.isfinite(mass):
        return min(max(loc, low), high)
    u = cdf_low + np.random.random() * mass
    z = norm_ppf(u)
    return min(max(loc + scale * z, low), high)


@njit(cache=True)
def sample_inverse_branch(y_lo, y_hi, loc, k, u01):
    """Sample one of the two inverse branches for a fixed transformed value."""
    w_lo = inverse_branch_log_weight(y_lo, loc, k)
    w_hi = inverse_branch_log_weight(y_hi, loc, k)
    max_weight = max(w_lo, w_hi)
    if not math.isfinite(max_weight):
        return y_lo
    p_lo = math.exp(w_lo - max_weight)
    p_hi = math.exp(w_hi - max_weight)
    return y_lo if u01 < p_lo / (p_lo + p_hi) else y_hi


@njit(cache=True)
def update_pairs(x_current, perm, mu_current, mu_star, k, sigma_z, z_min, z_max, x_out):
    n_pairs = perm.shape[0] // 2
    pair_acc = 0
    z_acc = 0
    for p in range(n_pairs):
        idx_i = perm[2 * p]
        idx_j = perm[2 * p + 1]
        xi = x_current[idx_i]
        xj = x_current[idx_j]
        yi = xi - mu_star
        yj = xj - mu_star
        zi = psi(yi, k)
        zj = psi(yj, k)
        delta = zi + zj
        low_int = max(z_min, delta - z_max)
        high_int = min(z_max, delta - z_min)
        if low_int >= high_int:
            x_out[idx_i] = xi
            x_out[idx_j] = xj
            continue

        z_prop = sample_truncated_normal(zi, sigma_z, low_int, high_int)
        log_norm_cur = truncnorm_log_normalizer(zi, sigma_z, low_int, high_int)
        log_norm_prop = truncnorm_log_normalizer(z_prop, sigma_z, low_int, high_int)
        log_cur = q_tilde_logpdf_no_const(zi, delta, mu_current, mu_star, k, z_min, z_max)
        log_prop = q_tilde_logpdf_no_const(z_prop, delta, mu_current, mu_star, k, z_min, z_max)
        log_alpha = log_prop - log_cur + log_norm_cur - log_norm_prop
        if math.isfinite(log_alpha) and math.log(max(np.random.random(), EPS_U)) < log_alpha:
            zi_tilde = z_prop
            z_acc += 1
        else:
            zi_tilde = zi
        zj_tilde = delta - zi_tilde
        if not (zj_tilde > z_min and zj_tilde < z_max):
            x_out[idx_i] = xi
            x_out[idx_j] = xj
            continue

        yi_lo, yi_hi = psi_inverse(zi_tilde, k)
        yj_lo, yj_hi = psi_inverse(zj_tilde, k)
        loc = mu_current - mu_star
        yi_new = sample_inverse_branch(yi_lo, yi_hi, loc, k, np.random.random())
        yj_new = sample_inverse_branch(yj_lo, yj_hi, loc, k, np.random.random())
        x_out[idx_i] = yi_new + mu_star
        x_out[idx_j] = yj_new + mu_star
        pair_acc += 1
    return pair_acc, z_acc


@njit(cache=True)
def run_gibbs_kernel(
    seed,
    x0,
    mu_star,
    T,
    k,
    sigma_mu,
    sigma_z,
    prior_loc,
    prior_scale,
    store_x_chain,
):
    n = x0.shape[0]
    np.random.seed(seed)
    mus = np.zeros(T + 1, dtype=np.float64)
    if store_x_chain:
        xs = np.zeros((T + 1, n), dtype=np.float64)
    else:
        xs = np.zeros((0, 0), dtype=np.float64)
    x_cur = np.empty(n, dtype=np.float64)
    x_out = np.empty(n, dtype=np.float64)
    for i in range(n):
        x_cur[i] = x0[i]
        if store_x_chain:
            xs[0, i] = x0[i]
    mus[0] = mu_star

    mu_acc = 0
    pair_acc = 0
    z_acc = 0
    z_min, z_max = z_support(k)

    for t_idx in range(1, T + 1):
        mu_cur = mus[t_idx - 1]
        mu_new, accepted_mu = update_mu_mh_random(
            mu_cur, x_cur, sigma_mu, prior_loc, prior_scale, k
        )
        if accepted_mu:
            mu_acc += 1
        mus[t_idx] = mu_new

        perm = np.random.permutation(n)
        for i in range(n):
            x_out[i] = x_cur[i]
        completed, accepted_z = update_pairs(
            x_cur, perm, mu_new, mu_star, k, sigma_z, z_min, z_max, x_out
        )
        pair_acc += int(completed)
        z_acc += int(accepted_z)
        for i in range(n):
            x_cur[i] = x_out[i]
            if store_x_chain:
                xs[t_idx, i] = x_out[i]

    x_final = np.empty(n, dtype=np.float64)
    for i in range(n):
        x_final[i] = x_cur[i]
    return mus, xs, x_final, mu_acc, pair_acc, z_acc


def run_gibbs_kernel_dispatch(key, mu_star, params):
    T = int(params["num_iterations_T"])
    n = int(params["n"])
    k = float(params["k"])
    seed = seed_from_key(key)
    store_x_chain = bool(params.get("store_x_chain", True))
    x0 = np.asarray(_initial_x(mu_star, n, k, params), dtype=np.float64)
    mus, xs, x_final, mu_acc, pair_acc, z_acc = run_gibbs_kernel(
        seed,
        x0,
        float(mu_star),
        T,
        k,
        float(params["proposal_std_mu"]),
        float(params["proposal_std_z"]),
        float(params["prior_mean"]),
        float(params["prior_std"]),
        store_x_chain,
    )
    out = {
        "mu_chain": mus,
        "x_final": x_final,
        "mu_acceptance_count": mu_acc,
        "pair_acceptance_count": pair_acc,
        "z_acceptance_count": z_acc,
        "store_x_chain": store_x_chain,
    }
    if store_x_chain:
        out["x_chain"] = xs
    return out


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
    backend = str(params.get("gibbs_backend", "numba_full"))
    if backend != "numba_full":
        raise ValueError("Student Gibbs uses the production 'numba_full' backend")
    T = int(params["num_iterations_T"])
    n = int(params["n"])
    k = params["k"]
    total_pairs = T * (n // 2)

    if verbose:
        print(f"Gibbs (Student): running {T} iterations", flush=True)
    out = run_gibbs_kernel_dispatch(key, mu_star, params)
    if verbose:
        print("Gibbs (Student): complete", flush=True)
    mu_acc = int(out["mu_acceptance_count"])
    pair_acc = int(out["pair_acceptance_count"])
    z_acc = int(out["z_acceptance_count"])
    _record_gibbs_costs(cost_ledger, T, n, mu_acc, pair_acc, backend)
    result = {
        "mu_chain": out["mu_chain"],
        "mu_acceptance_rate": mu_acc / T,
        "pair_acceptance_rate": pair_acc / total_pairs,
        "z_acceptance_rate": z_acc / total_pairs,
        "gibbs_backend": backend,
        "store_x_chain": out["store_x_chain"],
    }
    if "x_chain" in out:
        result["x_chain"] = out["x_chain"]
    if "x_final" in out:
        result["x_final"] = out["x_final"]
    if cost_ledger is not None:
        cost_ledger.set("store_x_chain", bool(out["store_x_chain"]))
    return result

def run_full_data_mh(key, x, params, verbose=True):
    """MH sampler for p(mu | x) with fixed data x. Returns mu_chain."""
    x = np.asarray(x, dtype=np.float64)
    T = int(params["num_iterations_T"])
    k = float(params["k"])
    mus = np.empty(T + 1, dtype=np.float64)
    mus[0] = float(np.median(x))
    rng = np.random.default_rng(seed_from_key(key))
    mu_acc = 0
    iters = range(1, T + 1)
    if verbose:
        iters = tqdm(iters, desc="Full-data MH (Student)")
    for t in iters:
        mu_new, acc = update_mu_mh_from_variates(
            mus[t - 1],
            x,
            params["proposal_std_mu"],
            params["prior_mean"],
            params["prior_std"],
            k,
            float(rng.normal()),
            float(rng.uniform(EPS_U, 1.0)),
        )
        mus[t] = mu_new
        mu_acc += int(acc)
    return {"mu_chain": mus, "mu_acceptance_rate": mu_acc / T}
