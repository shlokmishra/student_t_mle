import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random, vmap, jit
from jax.scipy.stats import t, norm, truncnorm
from jax.scipy.special import logsumexp, softmax

def sum_psi_jax(y: jnp.ndarray, k: jnp.int32) -> jnp.ndarray:
    return jnp.sum(psi_jax(y, k))

def psi_jax(y: jnp.ndarray, k: jnp.int32) -> jnp.ndarray:
    return y / (k + y**2)

def psi_inverse_jax(z: jnp.ndarray, k: jnp.int32):
    discr = 1.0 - 4.0 * k * z**2
    discr = jnp.maximum(discr, 0.0)
    sqrt_discr = jnp.sqrt(discr)

    eps = 1e-12
    denom = 2.0 * z
    denom_safe = jnp.where(jnp.abs(denom) < eps, jnp.sign(denom) * eps + eps, denom)

    y_plus  = (1.0 + sqrt_discr) / denom_safe
    y_minus = (1.0 - sqrt_discr) / denom_safe

    y_plus  = jnp.where(jnp.abs(z) < eps, 0.0, y_plus)
    y_minus = jnp.where(jnp.abs(z) < eps, 0.0, y_minus)

    return jnp.minimum(y_minus, y_plus), jnp.maximum(y_minus, y_plus)


def psi_prime_abs_jax(y: jnp.ndarray, k: jnp.int32) -> jnp.ndarray:
    num = k - y**2
    den = (k + y**2)**2
    return jnp.abs(num) / den

def log_psi_prime_abs_jax(y: jnp.ndarray, k: jnp.int32) -> jnp.ndarray:
    num = k - y**2
    den = (k + y**2)**2
    return jnp.log(jnp.abs(num)) - 2.0 * jnp.log(den)

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
    log_psi_prime_vals = log_psi_prime_abs_jax(y_vals, k)

    log_terms = log_fy_vals - log_psi_prime_vals  # shape (2,)

    log_q = logsumexp(log_terms)

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

    key_z, key_i, key_j = random.split(key, 3)


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

        yi_candidates = jnp.array([yi_minus, yi_plus])
        yj_candidates = jnp.array([yj_minus, yj_plus])
        
        log_wi = fy_logpdf_jax(yi_candidates, mu_current, mu_star, k) - log_psi_prime_abs_jax(yi_candidates, k) # try without the second term
        pi = softmax(log_wi)
        yi_tilde = yi_candidates[random.choice(key_i, 2, p=pi)]
        
        
        log_wj = fy_logpdf_jax(yj_candidates, mu_current, mu_star, k) - log_psi_prime_abs_jax(yj_candidates, k) # try without the second term
        pj = softmax(log_wj)
        yj_tilde = yj_candidates[random.choice(key_j, 2, p=pj)]
        
    
        xi_tilde = yi_tilde + mu_star
        xj_tilde = yj_tilde + mu_star

        return xi_tilde, xj_tilde, True, z_accepted

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
    x_current = x_0.copy() 
    mu_acceptance_count = 0
    z_i_acceptance_count = 0
    pair_acceptance_count = 0
    grad_likelihood_checks = jnp.zeros(T)
    
    total_z_moves = T * num_z_moves 

    # The main loop
    for t in tqdm(range(1, T+1), desc="Running Gibbs Sampler"):
        # --- Step (a): Sample mu(t) ---
        
        key, key_mu, key_x = random.split(key, 3)
        mu_new, accept_mu = update_mu_metropolis_jax(key_mu, mus[t-1], x_current, params['proposal_std_mu'], params['prior_mean'], params['prior_std'], params['k'])
        mus = mus.at[t].set(mu_new)
        x_current = xs[t-1].copy()  
        if accept_mu: 
            mu_acceptance_count += 1
            
        # --- Step (b): Sample x(t) ---
        x_new, accepted_pairs, accepted_z_is, deltas, deltas_new = update_x_full_jax(key_x, xs[t-1], mus[t], mu_star, params['k'], params['proposal_std_z'])
        xs = xs.at[t, :].set(x_new)
        # print("MLE difference:", mle_new - mle_current)
        grad_likelihood_checks = grad_likelihood_checks.at[t-1].set(sum_psi_jax(x_new - mu_star, params['k']))
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
        "grad_likelihood_checks": grad_likelihood_checks,
    }
    
    return results

def run_metropolis_x_jax(key: jax.random.PRNGKey, x : jnp.ndarray, params: dict) -> dict:
    T = params['num_iterations_T']
    m = x.shape[0]
    mus = jnp.zeros(T+1)
    mus = mus.at[0].set(jnp.median(x))
    
    num_z_moves = params.get('num_z_moves', m//2)
    params['num_z_moves'] = num_z_moves
    mu_acceptance_count = 0
    
    # The main loop
    for t in tqdm(range(1, T+1), desc="Running Gibbs Sampler"):
        # --- Step (a): Sample mu(t) ---
        key, key_mu, key_x = random.split(key, 3)
        mu_new, accept_mu = update_mu_metropolis_jax(key_mu, mus[t-1], x, params['proposal_std_mu'], params['prior_mean'], params['prior_std'], params['k'])
        mus = mus.at[t].set(mu_new)
        if accept_mu: 
            mu_acceptance_count += 1
    # Calculate final rates
    mu_acceptance_rate = mu_acceptance_count / T
  
    print(f"\n--- Sampling Complete ---")
    print(f"Mu Acceptance Rate: {mu_acceptance_rate:.4f}")
    
    results = {
        "mu_acceptance_rate": mu_acceptance_rate,
        "mu_chain": mus,
    }
    return results