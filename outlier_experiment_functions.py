# outlier_experiment_functions.py
# Helper functions for the outlier detection experiment.

import numpy as np
import scipy.stats as stats
from tqdm import tqdm

# We will need to import our existing sampler functions
import sampler_functions as sf

# ==============================================================================
# --- Phase 1: Data Generation
# ==============================================================================

def generate_datasets(params, outlier_percentile=0.995):
    """
    Generates two datasets: x1 (clean) and x2 (with a guaranteed outlier).

    Args:
        params (dict): Requires 'k', 'm', 'mu_true'.
        outlier_percentile (float): The percentile threshold to define an outlier.

    Returns:
        (np.array, np.array, float): A tuple containing:
                                     - x1: The clean dataset.
                                     - x2: The dataset with an outlier.
                                     - L: The calculated outlier threshold.
    """
    k = params['k']
    m = params['m']
    mu_true = params['mu_true']
    
    # 1. Define the outlier threshold L
    L = stats.t.ppf(outlier_percentile, df=k, loc=mu_true, scale=1)
    
    # 2. Generate x1 (clean dataset)
    #    We loop to ensure by random chance we don't get an outlier.
    while True:
        x1 = stats.t.rvs(df=k, loc=mu_true, scale=1, size=m)
        if np.max(x1) < L:
            break # We have a clean sample
            
    # 3. Generate x2 (dataset with a guaranteed outlier)
    #    First, generate m-1 normal samples
    x2_base = stats.t.rvs(df=k, loc=mu_true, scale=1, size=m-1)
    #    Then, generate one sample specifically from the tail of the distribution
    tail_prob = np.random.uniform(outlier_percentile, 1.0)
    outlier = stats.t.ppf(tail_prob, df=k, loc=mu_true, scale=1)
    #    Combine them
    x2 = np.append(x2_base, outlier)
    np.random.shuffle(x2) # Shuffle to make its position random
    
    print(f"Outlier threshold L (>{outlier_percentile*100:.1f}th percentile) = {L:.4f}")
    print(f"Max value in clean dataset (x1): {np.max(x1):.4f}")
    print(f"Max value in outlier dataset (x2): {np.max(x2):.4f}")
    
    return x1, x2, L

def get_mle(data, params):
    """A simple helper to find the MLE for a given dataset."""
    _, mu_star, _ = stats.t.fit(data, fdf=params['k'])
    return mu_star

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

def calculate_outlier_probability(mu_chain, L, params, num_samples=50000):
    """
    Calculates the posterior predictive probability of generating an outlier (x > L),
    given a chain of posterior samples for mu.
    
    Args:
        mu_chain (np.array): The MCMC chain of posterior mu samples.
        L (float): The outlier threshold.
        params (dict): Requires 'k' and 'm'.
        num_samples (int): The number of predictive samples to generate.
        
    Returns:
        float: The probability p(x̃ > L | evidence).
    """
    burn_in = int(len(mu_chain) * 0.2)
    posterior_mus = mu_chain[burn_in:]
    
    # Re-sample from the posterior chain to get i.i.d. draws
    mus_for_prediction = np.random.choice(posterior_mus, size=num_samples, replace=True)
    
    # For each mu, generate ONE new x_tilde value.
    # This is more efficient than generating full vectors.
    x_tilde_samples = stats.t.rvs(
        df=params['k'],
        loc=mus_for_prediction, # Pass the whole array of mus at once
        scale=1,
        size=num_samples
    )
    
    # The probability is the proportion of samples that exceed the threshold
    prob = np.mean(x_tilde_samples > L)
    
    return prob