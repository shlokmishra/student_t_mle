# outlier_experiment_functions.py
# Helper functions for the outlier detection experiment.

import numpy as np
import scipy.stats as stats
from tqdm import tqdm
from scipy.optimize import root_scalar


# We will need to import our existing sampler functions
import sampler_functions as sf

# ==============================================================================
# --- Phase 1: Data Generation
# ==============================================================================

# ==============================================================================
# --- Phase 2: Posterior Generation
# ==============================================================================

def get_full_data_posterior_samples(data, params):
    """Runs a simple M-H sampler to get p(μ|x)."""
    # Use the existing function from our main sampler_functions file
    start_mu = np.median(data) # Median is a robust starting point
    mu_chain = sf.run_full_data_sampler(params, data, start_mu)
    return mu_chain

def get_mle_conditional_posterior_samples(mu_star, params):
    """Runs the full Insufficient Gibbs Sampler to get p(μ|μ*)."""
    # Use the existing function from our main sampler_functions file
    temp_params = params.copy()
    temp_params['mu_star'] = mu_star
    mu_0, x_0 = sf.initialize_sampler(temp_params)
    results = sf.run_main_gibbs_sampler(temp_params, mu_0, x_0)
    return results['mu_chain']

# ==============================================================================
# --- Phase 3: Outlier Probability Calculation
# ==============================================================================

def generate_predictive_samples(mu_chain, params, num_samples=50000):
    """
    Generates posterior predictive samples of x̃ given a chain of mu samples.
    """
    burn_in = int(len(mu_chain) * 0.2)
    posterior_mus = mu_chain[burn_in:]
    
    # Re-sample from the posterior chain to get i.i.d. draws
    mus_for_prediction = np.random.choice(posterior_mus, size=num_samples, replace=True)
    
    # For each mu, generate ONE new x_tilde value.
    x_tilde_samples = stats.t.rvs(
        df=params['k'],
        loc=mus_for_prediction, # Pass the whole array of mus at once
        scale=1,
        size=num_samples
    )
    return x_tilde_samples

def calculate_outlier_probability(x_tilde_samples, L):
    """
    Calculates the probability of an outlier from a set of predictive samples.
    """
    # The probability is the proportion of samples that exceed the threshold
    prob = np.mean(x_tilde_samples > L)
    return prob