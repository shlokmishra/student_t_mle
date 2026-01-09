import numpy as np
import scipy.stats as stats
from scipy.optimize import root_scalar


def get_mle(data, params):
    """Find the MLE for the location parameter μ of a t-distribution
    with fixed dof and fixed scale=1, for the given data array."""

    k = params['k']
    x = np.asarray(data)
    # print(f"Calculating MLE for data with {len(x)} points and k={k}...")
    # Define the MLE equation: sum((x - mu) / (k + (x - mu)**2)) = 0
    def mle_equation(mu):
        return np.sum((x - mu) / (k + (x - mu)**2))

    # # Use the median as a robust initial guess and bracket for root finding
    # median = np.median(x)
    bracket = (x.min() - 10, x.max() + 10)

    # Find the root
    result = root_scalar(mle_equation, bracket=bracket, method='brentq')
    if not result.converged:
        raise RuntimeError("MLE root finding did not converge.")

    mu_star = result.root
    return mu_star

def get_benchmark_mle_samples(params, num_simulations=10000):
    """
    Performs a large simulation to get the raw samples of the MLE distribution p(μ̂ | μ=0).
    This is the computationally expensive part.

    Args:
        params (dict): Needs 'k' and 'm'.
        num_simulations (int): The number of datasets to generate.

    Returns:
        np.array: An array containing the calculated MLE for each simulated dataset.
    """
    print(f"\n--- Computing Benchmark KDE from {num_simulations} simulations ---")
    
    k = params['k']
    m = params['m']
    mle_samples = np.zeros(num_simulations)

    # It's faster to generate all random numbers at once
    # We assume mu=0 for the benchmark distribution
    all_data = stats.t.rvs(df=k, loc=0, scale=1, size=(num_simulations, m))

    for i in range(num_simulations):
        if (i + 1) % 10000 == 0:
            print(f"  Processing simulation {i+1}/{num_simulations}...")
        # Use the pre-generated data for this simulation
        sample_data = all_data[i, :]
        
        # Calculate the MLE for this dataset
        # Note: This assumes you have a get_mle function available
        mle_samples[i] = get_mle(sample_data, params)
        
        
    return mle_samples


from scipy.integrate import quad


def get_normalized_posterior_mle_pdf(mu_star, params, num_simulations=10000):
    """
    Returns a callable for the *normalized* posterior density p(mu | mu_hat = mu_star),
    using numerical integration to compute the normalizing constant.
    """

    # Same KDE and prior as above to ensure consistency
    mles = get_benchmark_mle_samples(params, num_simulations=num_simulations)
    prior_mean = params['prior_mean']
    prior_std = params['prior_std']
    bw_method = params.get('kde_bw_method', 'scott')
    print("Fitting KDE to MLE samples using bw_method =", bw_method)
    kde = stats.gaussian_kde(mles, bw_method=bw_method)

    def log_unnorm(mu):
        log_prior = stats.norm.logpdf(mu, loc=prior_mean, scale=prior_std)
        log_like = kde.logpdf(mu_star - mu)
        return log_prior + log_like

    # Compute normalizing constant via numerical integration
    integral, _ = quad(lambda mu: np.exp(log_unnorm(mu)), -np.inf, np.inf)

    def normalized_pdf(mu):
        return np.exp(log_unnorm(mu)) / integral

    return normalized_pdf   