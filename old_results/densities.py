# import numpy as np
# import scipy.stats as stats
# from scipy.special import logsumexp


# def unnormalized_posterior_mu_logpdf(mu, x, params):
#     """
#     Log of the unnormalized posterior p(mu | x) (log-likelihood + log-prior).
#     Accepts scalar mu or array-like mu (returns array of log-posteriors).
#     """
#     x = np.asarray(x)
#     mu = np.asarray(mu)
#     k = params['k']


#     if mu.ndim == 0:
#         # scalar mu -> sum over data
#         log_likelihood = np.sum(stats.t.logpdf(x, df=k, loc=mu, scale=1))
#         log_prior = stats.norm.logpdf(mu, loc=params['prior_mean'], scale=params['prior_std'])
#     else:
#         # vectorized mu -> compute per-mu likelihood: shape (n_data, n_mu) -> sum over data -> (n_mu,)
#         log_likelihood = np.sum(stats.t.logpdf(x[:, None], df=k, loc=mu[None, :], scale=1), axis=0)
#         log_prior = stats.norm.logpdf(mu, loc=params['prior_mean'], scale=params['prior_std'])

#     return log_likelihood + log_prior


# def psi(y, k):
#     return y / (k + y**2)

# def psi_inverse(z, k):
#     if np.isclose(z, 0.0):
#         return 0.0, 0.0

#     discriminant = 1 - 4 * k * z**2
#     if discriminant < 0:
#         print(f"Warning: Negative discriminant for z={z:.4f}")
#         return np.nan, np.nan

#     sqrt_disc = np.sqrt(discriminant)
#     y_plus  = (1 + sqrt_disc) / (2 * z)
#     y_minus = (1 - sqrt_disc) / (2 * z)
#     return tuple(sorted((y_minus, y_plus)))

# def psi_prime_abs(y, k):
#     num = k - y**2
#     den = (k + y**2)**2
#     return np.abs(num) / den

# def fy_pdf(y, mu, mu_star, k):
#     loc = mu - mu_star
#     return stats.t.pdf(y, df=k, loc=loc, scale=1)

# def fy_logpdf(y, mu, mu_star, k):
#     loc = mu - mu_star
#     return stats.t.logpdf(y, df=k, loc=loc, scale=1)

# def q_logpdf(z, mu, mu_star, k):
#     # support of psi
#     supp_q = (-1/(2*np.sqrt(k)), 1/(2*np.sqrt(k)))
#     if not (supp_q[0] < z < supp_q[1]):
#         return -np.inf

#     y_minus, y_plus = psi_inverse(z, k)
#     if np.isnan(y_minus):
#         return -np.inf

#     if np.isclose(y_minus, y_plus):
#         # cas spécial "une seule racine"
#         log_fy        = fy_logpdf(y_minus, mu, mu_star, k)
#         log_psi_prime = np.log(psi_prime_abs(y_minus, k) + 1e-30)
#         return log_fy - log_psi_prime

#     y_vals = np.array([y_minus, y_plus])
#     log_fy_vals        = fy_logpdf(y_vals, mu, mu_star, k)
#     log_psi_prime_vals = np.log(psi_prime_abs(y_vals, k) + 1e-30)

#     log_terms = log_fy_vals - log_psi_prime_vals
#     return logsumexp(log_terms)

# def q_tilde_logpdf(z, delta, mu_current, mu_star, k):
#     supp_q = (-1/(2*np.sqrt(k)), 1/(2*np.sqrt(k)))
    
#     supp_tilde_q = (max(supp_q[0], delta - supp_q[1]),
#                     min(supp_q[1], delta - supp_q[0]))
    
    
#     if not (supp_tilde_q[0] < z < supp_tilde_q[1]):
#         return -np.inf

#     log_q_z       = q_logpdf(z,         mu_current, mu_star, k)
#     log_q_partner = q_logpdf(delta - z, mu_current, mu_star, k)
#     return log_q_z + log_q_partner



# from utils import get_benchmark_mle_samples
# from scipy.integrate import quad

# def get_unnormalized_posterior_mle_logpdf(mu_star, params, num_simulations=10000):
#     """
#     Returns a callable log f(mu | mu_hat = mu_star) up to a multiplicative constant.
    
#     p(mu | mu_hat = mu_star) ∝ p(mu) * p(mu_hat = mu_star | mu)
#     ≈ N(mu ; prior_mean, prior_std^2) * kde(mu_star - mu)
#     where kde is a Gaussian KDE fitted on benchmark MLE error samples.
#     """

#     # Samples that approximate the distribution of (mu_hat - mu)
#     mles = get_benchmark_mle_samples(params, num_simulations=num_simulations)
#     prior_mean = params['prior_mean']
#     prior_std = params['prior_std']
#     kde = stats.gaussian_kde(mles)

#     def log_unnorm(mu):
#         # prior on mu
#         log_prior = stats.norm.logpdf(mu, loc=prior_mean, scale=prior_std)
#         # approximate likelihood p(mu_hat = mu_star | mu)
#         log_like = kde.logpdf(mu_star - mu)
#         return log_prior + log_like

#     return log_unnorm


# def get_normalized_posterior_mle_pdf(mu_star, params, num_simulations=10000):
#     """
#     Returns a callable for the *normalized* posterior density p(mu | mu_hat = mu_star),
#     using numerical integration to compute the normalizing constant.
#     """

#     # Same KDE and prior as above to ensure consistency
#     mles = get_benchmark_mle_samples(params, num_simulations=num_simulations)
#     prior_mean = params['prior_mean']
#     prior_std = params['prior_std']
#     kde = stats.gaussian_kde(mles)

#     def log_unnorm(mu):
#         log_prior = stats.norm.logpdf(mu, loc=prior_mean, scale=prior_std)
#         log_like = kde.logpdf(mu_star - mu)
#         return log_prior + log_like

#     # Normalizing constant Z = ∫ exp(log_unnorm(mu)) dmu
#     def integrand(t):
#         return np.exp(log_unnorm(t))

#     integral, _ = quad(integrand, -np.inf, np.inf)
#     normalization_constant = integral

#     def posterior_pdf(mu):
#         return np.exp(log_unnorm(mu)) / normalization_constant

#     return posterior_pdf
