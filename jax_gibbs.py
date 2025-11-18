import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random, vmap, jit
from jax.scipy.stats import t, norm, truncnorm

def psi_jax(y: jnp.ndarray, k: jnp.int32) -> jnp.ndarray:
    return y / (k + y**2)

def psi_inverse_jax(z: jnp.ndarray, k: jnp.int32) -> jnp.ndarray:
    # Handle the discriminant carefully to avoid NaNs
    discr = 1.0 - 4.0 * k * z**2
    discr = jnp.clip(discr, a_min=0.0)
    sqrt_discr = jnp.sqrt(discr)

    def non_zero_branch(z_local):
        y_plus  = (1.0 + sqrt_discr) / (2.0 * z_local)
        y_minus = (1.0 - sqrt_discr) / (2.0 * z_local)
        y1 = jnp.minimum(y_minus, y_plus)
        y2 = jnp.maximum(y_minus, y_plus)
        return y1, y2

    def zero_branch(z_local):
        # when z is close to 0, use the approximation y ≈ k * z
        y0 = k * z_local # approximation
        return y0, y0

    # when z is close to 0, use the approximation
    eps = 1e-8  
    return jax.lax.cond(jnp.abs(z) < eps,
                        zero_branch,
                        non_zero_branch,
                        z)

def psi_prime_abs_jax(y: jnp.ndarray, k: jnp.int32) -> jnp.ndarray:
    num = k - y**2
    den = (k + y**2)**2
    return jnp.abs(num) / den

def student_t_logpdf(y: jnp.ndarray, df: jnp.int32, loc: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    # log pdf du Student-t :
    # log Γ((ν+1)/2) - log Γ(ν/2) - 0.5 log(νπ) - log(scale)
    # - ((ν+1)/2) * log(1 + ((y-loc)/scale)^2 / ν)
    y_std = (y - loc) / scale
    nu = df
    log_norm = (
        jax.scipy.special.gammaln((nu + 1.) / 2.)
        - jax.scipy.special.gammaln(nu / 2.)
        - 0.5 * jnp.log(nu * jnp.pi)
        - jnp.log(scale)
    )
    log_kernel = - (nu + 1.) / 2. * jnp.log1p((y_std**2) / nu)
    return log_norm + log_kernel

def fy_logpdf_jax(y: jnp.ndarray, mu_current: jnp.ndarray, mu_star: jnp.ndarray, k: jnp.int32) -> jnp.ndarray:
    loc = mu_current - mu_star
    return t.logpdf(y, df=k, loc=loc, scale=1.0)



def q_logpdf_jax(z: jnp.ndarray, mu_current: jnp.ndarray, mu_star: jnp.ndarray, k: jnp.int32):
    z_min, z_max = (-1/(2*jnp.sqrt(k)), 1/(2*jnp.sqrt(k)))
    
    in_supp = (z > z_min) & (z < z_max)

    y_minus, y_plus = psi_inverse_jax(z, k)

    # log f(y) + log |psi'(y)|^{-1}
    y_vals = jnp.stack([y_minus, y_plus])  # shape (2,)
    log_fy_vals = fy_logpdf_jax(y_vals, mu_current, mu_star, k)
    log_psi_prime_vals = jnp.log(psi_prime_abs_jax(y_vals, k) + 1e-30)

    log_terms = log_fy_vals - log_psi_prime_vals  # shape (2,)

    log_q = jax.scipy.special.logsumexp(log_terms)

    log_q = jnp.where(in_supp, log_q, -jnp.inf)
    return log_q

def q_tilde_logpdf_jax(z: jnp.ndarray, delta: jnp.ndarray, mu_current: jnp.ndarray, mu_star: jnp.ndarray, k: jnp.int32) -> jnp.ndarray:
    log_q_z       = q_logpdf_jax(z,           mu_current, mu_star, k)
    log_q_partner = q_logpdf_jax(delta - z,   mu_current, mu_star, k)
    return log_q_z + log_q_partner


def update_z_one(key: jax.random.PRNGKey, z_current: jnp.ndarray, delta: jnp.ndarray, mu_current: jnp.ndarray, mu_star: jnp.ndarray, k : jnp.int32, sigma_z: float) -> tuple:
    key_prop, key_u = random.split(key, 2)
    low = -1/(2*jnp.sqrt(k))
    high = 1/(2*jnp.sqrt(k))
    z_prop = z_current + sigma_z * random.truncated_normal(key_prop, shape=(), lower=(low - z_current)/sigma_z, upper=(high - z_current)/sigma_z)
    
    log_kernel_current_to_prop = truncnorm.logpdf(z_prop, a=(low - z_current)/sigma_z, b=(high - z_current)/sigma_z, loc=z_current, scale=sigma_z)
    log_kernel_prop_to_current = truncnorm.logpdf(z_current, a=(low - z_prop)/sigma_z, b=(high - z_prop)/sigma_z, loc=z_prop, scale=sigma_z)

    log_post_current  = q_tilde_logpdf_jax(z_current, delta, mu_current, mu_star, k)
    log_post_proposal = q_tilde_logpdf_jax(z_prop,     delta, mu_current, mu_star, k)

    log_alpha = log_post_proposal - log_post_current + log_kernel_prop_to_current - log_kernel_current_to_prop

    u = random.uniform(key_u)
    log_u = jnp.log(u)
    accept = log_u < log_alpha

    z_new = jnp.where(accept, z_prop, z_current)
    return z_new, accept


def update_xi_xj_one(key, xi, xj, mu_current, mu_star, k, sigma_z):

    key_z, key_choice = random.split(key)

    # --- Step 1: delta ---
    yi, yj = xi - mu_star, xj - mu_star
    zi, zj = psi_jax(yi, k), psi_jax(yj, k)
    delta  = zi + zj

    # --- Step 2: update_z_one ---
    zi_tilde, z_accepted = update_z_one(
        key_z, zi, delta, mu_current, mu_star, k, sigma_z
    )
    zj_tilde = delta - zi_tilde

    z_min, z_max = (-1/(2*jnp.sqrt(k)), 1/(2*jnp.sqrt(k)))
    in_supp_partner = (zj_tilde > z_min) & (zj_tilde < z_max)

    def reject_pair(_):
        return xi, xj, False, z_accepted

    def accept_pair(_):
        yi_minus, yi_plus = psi_inverse_jax(zi_tilde, k)
        yj_minus, yj_plus = psi_inverse_jax(zj_tilde, k)

        ys_candidate = jnp.array([yi_minus, yi_plus, yj_minus, yj_plus])
        w_single = jnp.exp(fy_logpdf_jax(ys_candidate, mu_current, mu_star, k))

        w1 = w_single[0] * w_single[2]  # (yi-, yj-)
        w2 = w_single[0] * w_single[3]  # (yi-, yj+)
        w3 = w_single[1] * w_single[2]  # (yi+, yj-)
        w4 = w_single[1] * w_single[3]  # (yi+, yj+)
        weights = jnp.array([w1, w2, w3, w4])

        sum_w = jnp.sum(weights)
        probs = weights / sum_w
        idx = random.choice(key_choice, 4, p=probs)
        yi_candidates = jnp.array([yi_minus, yi_minus, yi_plus, yi_plus])
        yj_candidates = jnp.array([yj_minus, yj_plus, yj_minus, yj_plus])

        y_i_new = yi_candidates[idx]
        y_j_new = yj_candidates[idx]

        x_i_new = y_i_new + mu_star
        x_j_new = y_j_new + mu_star

        return x_i_new, x_j_new, True, z_accepted

        # def reject_from_weights(_):
        #     return xi, xj, False, z_accepted

        # def sample_from_weights(_):
        #     probs = weights / sum_w
        #     idx = random.choice(key_choice, 4, p=probs)
        #     yi_candidates = jnp.array([yi_minus, yi_minus, yi_plus, yi_plus])
        #     yj_candidates = jnp.array([yj_minus, yj_plus, yj_minus, yj_plus])

        #     y_i_new = yi_candidates[idx]
        #     y_j_new = yj_candidates[idx]

        #     x_i_new = y_i_new + mu_star
        #     x_j_new = y_j_new + mu_star

        #     return x_i_new, x_j_new, True, z_accepted

        # return jax.lax.cond(sum_w <= 0.0, reject_from_weights, sample_from_weights, operand=None)

    return jax.lax.cond(in_supp_partner, accept_pair, reject_pair, operand=None)

@jit
def delta_from_xi_xj(xi, xj, mu_star, k):
    yi, yj = xi - mu_star, xj - mu_star
    zi, zj = psi_jax(yi, k), psi_jax(yj, k)
    delta  = zi + zj
    return delta

delta_from_xi_xj_batch = vmap(delta_from_xi_xj, in_axes=(0,0,None,None))
@jit
def update_x_full_jax(key, x_current, mu_current, mu_star, k, sigma_z):
    m = x_current.shape[0]
    assert m % 2 == 0

    key_perm, key_pairs = random.split(key)
    perm = random.permutation(key_perm, m)
    x_perm = x_current[perm]

    xis = x_perm[0::2]
    xjs = x_perm[1::2]
    n_pairs = xis.shape[0]

    keys_pairs = random.split(key_pairs, n_pairs)

    update_xi_xj_batch = vmap(
        update_xi_xj_one,
        in_axes=(0, 0, 0, None, None, None, None)
    )

    xis_new, xjs_new, pair_accepted_vec, z_accepted_vec = update_xi_xj_batch(
        keys_pairs, xis, xjs, mu_current, mu_star, k, sigma_z
    )
    
    deltas = delta_from_xi_xj_batch( xis, xjs, mu_star, k)
    deltas_new = delta_from_xi_xj_batch( xis_new, xjs_new, mu_star, k)
    
    
    x_updated_pairs = jnp.stack([xis_new, xjs_new], axis=1).reshape(-1)
    x_new = x_perm.at[0:m].set(x_updated_pairs)  #

    pair_accepted_count = jnp.sum(pair_accepted_vec)
    z_accepted_count    = jnp.sum(z_accepted_vec)

    # x_new = jnp.round(x_new, decimals=6)  # avoid numerical issues
    return x_new, pair_accepted_count, z_accepted_count, deltas, deltas_new


def unnormalized_posterior_mu_logpdf_jax(mu: float, x: jnp.ndarray, prior_loc: float, prior_scale: float, k: int) -> float:
    """
    Log of the unnormalized posterior p(mu | x) (log-likelihood + log-prior).
    Accepts scalar mu or array-like mu (returns array of log-posteriors).
    """
    x = jnp.asarray(x)
    mu = jnp.asarray(mu)
    k = jnp.asarray(k)

    if mu.ndim == 0:
        log_likelihood = jnp.sum(t.logpdf(x, df=k, loc=mu, scale=1))
        log_prior = norm.logpdf(mu, loc=prior_loc, scale=prior_scale)
    else:
        # vectorized mu -> compute per-mu likelihood: shape (n_data, n_mu) -> sum over data -> (n_mu,)
        log_likelihood = jnp.sum(t.logpdf(x[:, None], df=k, loc=mu[None, :], scale=1), axis=0)
        log_prior = norm.logpdf(mu, loc=prior_loc, scale=prior_scale)

    return log_likelihood + log_prior





@jit
def update_mu_metropolis_jax(key: jax.random.PRNGKey, mu_current: float, x_current: jnp.ndarray, sigma_mu : float, prior_loc: float, prior_scale: float, k : int) -> tuple:
    """
    Performs one Metropolis-Hastings step to get a new sample for mu.
    """
    key_prop, key_u = random.split(key)

    mu_candidate = mu_current + sigma_mu * random.normal(key_prop)

    log_post_current = unnormalized_posterior_mu_logpdf_jax(
        mu=mu_current,
        x=x_current,
        prior_loc=prior_loc,
        prior_scale=prior_scale,
        k=k
    )
    log_post_candidate = unnormalized_posterior_mu_logpdf_jax(
        mu=mu_candidate,
        x=x_current,
        prior_loc=prior_loc,
        prior_scale=prior_scale,
        k=k
    )

    log_acceptance_ratio = log_post_candidate - log_post_current
    # avoid log(0)
    u = random.uniform(key_u, minval=1e-12, maxval=1.0)
    log_u = jnp.log(u)
    accept = log_u < log_acceptance_ratio
    mu_new = jnp.where(accept, mu_candidate, mu_current)
    return mu_new, accept

    
    
from tqdm import tqdm

def run_gibbs_sampler_mle_jax(key: jax.random.PRNGKey, mu_star : float, params: dict) -> dict:
    T = params['num_iterations_T']
    m = params['m']
    mus = jnp.zeros(T+1)
    xs = jnp.zeros((T+1, m))
    
    x_0 = jnp.ones(m) * mu_star
    xs = xs.at[0, :].set(x_0)
    
    mle_current = mu_star
    mu_0 = mu_star
    mus = mus.at[0].set(mu_0)
    
    num_z_moves = params.get('num_z_moves', m//2)
    params['num_z_moves'] = num_z_moves
    x_current = x_0.copy() # Use a copy to avoid modifying the original
    mu_acceptance_count = 0
    z_i_acceptance_count = 0
    pair_acceptance_count = 0
    
    total_z_moves = T * num_z_moves 

    # The main loop
    for t in tqdm(range(1, T+1), desc="Running Gibbs Sampler"):
        # --- Step (a): Sample mu(t) ---
        
        key, key_mu, key_x = random.split(key, 3)
        mu_new, accept_mu = update_mu_metropolis_jax(key_mu, mus[t-1], x_current, params['proposal_std_mu'], params['prior_mean'], params['prior_std'], params['k'])
        mus = mus.at[t].set(mu_new)
        x_current = xs[t-1].copy()  # Update current x for the next
        if accept_mu: 
            mu_acceptance_count += 1
            
        # --- Step (b): Sample x(t) ---
        x_new, accepted_pairs, accepted_z_is, deltas, deltas_new = update_x_full_jax(key_x, xs[t-1], mus[t], mu_star, params['k'], params['proposal_std_z'])
        xs = xs.at[t, :].set(x_new)
        mle_new = get_mle(x_new, params['k'])
        # print("MLE difference:", mle_new - mle_current)
        mle_current = mle_new
        # print("Difference in deltas (should be small):", jnp.max(jnp.abs(deltas - deltas_new)))
        z_i_acceptance_count += accepted_z_is
        pair_acceptance_count += accepted_pairs


    # Calculate final rates
    mu_acceptance_rate = mu_acceptance_count / T
    z_i_acceptance_rate = z_i_acceptance_count / total_z_moves 
    pair_acceptance_rate = pair_acceptance_count / total_z_moves

    print(f"\n--- Sampling Complete ---")
    print(f"Mu Acceptance Rate: {mu_acceptance_rate:.4f}")
    print(f"Z_i Acceptance Rate: {z_i_acceptance_rate:.4f}")
    
    results = {
        "mu_acceptance_rate": mu_acceptance_rate,
        "pair_acceptance_rate": pair_acceptance_rate,
        "z_i_acceptance_rate": z_i_acceptance_rate,
        "mu_chain": mus,
        "x_chain": xs,
    }
    
    return results


from scipy.optimize import root_scalar

def get_mle(x, k):
    """Find the MLE for the location parameter μ of a t-distribution
    with fixed dof and fixed scale=1, for the given data array."""

    x = jnp.asarray(x)
    # print(f"Calculating MLE for data with {len(x)} points and k={k}...")
    # Define the MLE equation: sum((x - mu) / (k + (x - mu)**2)) = 0
    def mle_equation(mu):
        return jnp.sum((x - mu) / (k + (x - mu)**2))

    # Use the median as a robust initial guess and bracket for root finding
    bracket = (x.min() - 10, x.max() + 10)

    # Find the root
    result = root_scalar(mle_equation, bracket=bracket, method='brentq')
    if not result.converged:
        raise RuntimeError("MLE root finding did not converge.")

    mu_star = result.root
    # mu_star = jnp.round(mu_star, decimals=6)  # avoid numerical issues
    return mu_star