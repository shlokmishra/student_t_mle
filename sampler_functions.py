import numpy as np
from scipy import stats
from scipy.special import logsumexp
from tqdm import tqdm
from scipy.optimize import root_scalar


def get_mle(data, params):
    """Find the MLE for the location parameter μ of a t-distribution
    with fixed dof and fixed scale=1, for the given data array."""

    k = params['k']
    x = np.asarray(data)
    print(f"Calculating MLE for data with {len(x)} points and k={k}...")
    # Define the MLE equation: sum((x - mu) / (k + (x - mu)**2)) = 0
    def mle_equation(mu):
        return np.sum((x - mu) / (k + (x - mu)**2))

    # Use the median as a robust initial guess and bracket for root finding
    median = np.median(x)
    bracket = (x.min() - 10, x.max() + 10)

    # Find the root
    result = root_scalar(mle_equation, bracket=bracket, method='brentq')
    if not result.converged:
        raise RuntimeError("MLE root finding did not converge.")

    mu_star = result.root
    return mu_star

def mle_score(mu, x, k):
    """Calculates the score function for the MLE of a t-distribution."""
    return np.sum((x - mu) / (k + (x - mu)**2))

def negative_log_likelihood(mu, data, k):
    """Calculates the negative of the log-likelihood for Student's t data."""
    # logpdf is log(PDF), so we sum them up for the log-likelihood
    log_likelihood = np.sum(stats.t.logpdf(data, df=k, loc=mu, scale=1))
    return -log_likelihood

def log_posterior_mu(mu, data_x, k, prior_mean, prior_std):
    """
    Calculates the log of the posterior probability of mu, p(mu | x).
    This is proportional to log(likelihood) + log(prior).
    """
    # Calculate the log-likelihood: log( p(x | mu) )
    # This is the sum of the log-PDF of the t-distribution for each data point.
    log_likelihood = np.sum(stats.t.logpdf(data_x, df=k, loc=mu, scale=1))

    # Calculate the log-prior: log( p(mu) )
    # This is the log-PDF of our Normal prior for mu.
    log_prior = stats.norm.logpdf(mu, loc=prior_mean, scale=prior_std)

    # The log-posterior is the sum of the two.
    return log_likelihood + log_prior

def update_mu_metropolis(mu_current, x_current, params):
    """
    Performs one Metropolis-Hastings step to get a new sample for mu.
    """
    # 1. Propose a new candidate for mu from a symmetric proposal distribution.
    #    We use a Normal distribution centered at the current mu.
    mu_candidate = np.random.normal(loc=mu_current, scale=params['proposal_std_mu'])

    # 2. Calculate the log-posterior for the current and candidate mu values.
    log_post_current = log_posterior_mu(
        mu=mu_current,
        data_x=x_current,
        k=params['k'],
        prior_mean=params['prior_mean'],
        prior_std=params['prior_std']
    )
    log_post_candidate = log_posterior_mu(
        mu=mu_candidate,
        data_x=x_current,
        k=params['k'],
        prior_mean=params['prior_mean'],
        prior_std=params['prior_std']
    )

    # 3. Calculate the acceptance probability (alpha) in log-space.
    #    alpha = min(1, p(candidate)/p(current))
    #    log(alpha) = min(0, log(p(candidate)) - log(p(current)))
    log_acceptance_ratio = log_post_candidate - log_post_current
    acceptance_prob = min(1.0, np.exp(log_acceptance_ratio))

    # 4. Accept or reject the candidate.
    if np.random.uniform(0, 1) < acceptance_prob:
        return mu_candidate
    else:
        return mu_current

def psi(y, k):
    """The forward transformation from y-space to z-space."""
    return y / (k + y**2)

def psi_inverse(z, k):
    """
    The inverse transformation from z-space to y-space.
    Returns a tuple of the two possible y values (y_minus, y_plus).
    """
    # edge case where z is very close to zero
    if np.isclose(z, 0):
        return 0.0, 0.0 # The only inverse is y=0

    # Ensure the value under the square root is non-negative
    discriminant = 1 - 4 * k * z**2

    if discriminant < 0:
        # This case should not be reached if z is in the valid domain
        print(f"Warning: Negative discriminant for z={z:.4f}")
        return np.nan, np.nan

    sqrt_discriminant = np.sqrt(discriminant)
    y_plus = (1 + sqrt_discriminant) / (2 * z)
    y_minus = (1 - sqrt_discriminant) / (2 * z)

    return tuple(sorted((y_minus, y_plus)))

def psi_prime_abs(y, k):
    """
    The absolute value of the derivative of the psi function.
    This is the "stretching factor" or Jacobian for the 1D transformation.
    """
    numerator = k - y**2
    denominator = (k + y**2)**2
    # The denominator is always non-negative, so we only need the abs of the numerator.
    return np.abs(numerator) / denominator

def f_y(y, mu_current, mu_star, k):

    loc = mu_current - mu_star
    return stats.t.pdf(y, df=k, loc=loc, scale=1)

def log_q_density(z, mu_current, mu_star, k, z_domain):
    """
    Calculates the log of the pushforward probability density, log(q(z)),
    using vectorization and the log-sum-exp trick.
    """
    if not (z_domain[0] < z < z_domain[1]):
        return -np.inf

    y_minus, y_plus = psi_inverse(z, k)

    if np.isnan(y_minus):
        return -np.inf

    # Handle the special case where the two branches are the same (e.g., z=0)
    # In this case, there's only one term in the sum, no vectorization needed.
    if np.isclose(y_minus, y_plus):
        log_fy = stats.t.logpdf(y_minus, df=k, loc=mu_current - mu_star, scale=1)
        log_psi_prime = np.log(psi_prime_abs(y_minus, k) + 1e-30)
        return log_fy - log_psi_prime


    # Create a single array of the two y values
    y_vals = np.array([y_minus, y_plus])

    # Call logpdf and psi_prime_abs only ONCE on the array
    log_fy_vals = stats.t.logpdf(y_vals, df=k, loc=mu_current - mu_star, scale=1)
    log_psi_prime_vals = np.log(psi_prime_abs(y_vals, k) + 1e-30)

    # Calculate both log terms at once with element-wise subtraction
    log_terms = log_fy_vals - log_psi_prime_vals

    # Combine the two log terms using the stable log-sum-exp trick
    return logsumexp(log_terms)

def log_q_tilde(z, delta, mu_current, mu_star, k, z_domain):
    """
    Calculates the log of the target density log(q_tilde(z)).
    log(q_tilde(z)) = log(q(z)) + log(q(delta - z))

    """
    # Calculate the log-density of the z component
    log_q_z = log_q_density(z, mu_current, mu_star, k, z_domain)

    # Calculate the log-density of the partner component
    log_q_partner = log_q_density(delta - z, mu_current, mu_star, k, z_domain)

    # The log of the product is the sum of the logs.
    # If either term is -inf, the sum will correctly be -inf.
    return log_q_z + log_q_partner

def sample_z_i(z_start, delta, mu_current, mu_star, k, z_domain, proposal_std_z):
    """
    Performs one Metropolis-Hastings step to get a new sample for z_i.
    (This function's logic remains unchanged, but it now benefits from the
     more stable log_q_tilde function.)

    Returns:
        (float, bool): A tuple containing:
                       - The new z_i value.
                       - A boolean indicating if the proposal was accepted.
    """
    # 1. Propose a new candidate for z, ensuring it's in the valid domain.
    while True:
        z_candidate = np.random.normal(loc=z_start, scale=proposal_std_z)
        if z_domain[0] < z_candidate < z_domain[1]:
            break

    # 2. Calculate the log-posterior (log_q_tilde) for current and candidate z.
    log_post_current = log_q_tilde(z_start, delta, mu_current, mu_star, k, z_domain)
    log_post_candidate = log_q_tilde(z_candidate, delta, mu_current, mu_star, k, z_domain)

    # 3. Calculate the acceptance probability.
    log_acceptance_ratio = log_post_candidate - log_post_current
    if np.isneginf(log_acceptance_ratio):
        acceptance_prob = 0.0
    else:
        acceptance_prob = min(1.0, np.exp(log_acceptance_ratio))

    # 4. Accept or reject the candidate.
    if np.random.uniform(0, 1) < acceptance_prob:
        return z_candidate, True
    else:
        return z_start, False
    
def update_one_pair(x_current, indices, mu_current, params):
    """
    Performs the full 6-step constrained pairwise update for indices (i, j).

    Returns:
        (np.array, bool, bool): A tuple containing:
                                - The new data vector.
                                - A boolean for whether the PAIR was updated.
                                - A boolean for whether the Z_I proposal was accepted.
    """
    i, j = indices
    k = params['k']
    mu_star = params['mu_star']

    # --- Step 1: Calculate delta ---
    y_i, y_j = x_current[i] - mu_star, x_current[j] - mu_star
    z_i, z_j = psi(y_i, k), psi(y_j, k)
    delta = z_i + z_j

    # --- Step 2: Sample z_tilde_i ---
    z_tilde_i, z_i_accepted = sample_z_i(
        z_start=z_i, delta=delta, mu_current=mu_current, mu_star=mu_star, k=k,
        z_domain=params['z_domain'], proposal_std_z=params['proposal_std_z']
    )

    # --- Step 3: Set Partner and Check Domain ---
    z_tilde_j = delta - z_tilde_i
    if not (params['z_domain'][0] < z_tilde_j < params['z_domain'][1]):
        return x_current, False, z_i_accepted

    # --- Step 4: Compute inverse branches ---
    y_i_minus, y_i_plus = psi_inverse(z_tilde_i, k)
    y_j_minus, y_j_plus = psi_inverse(z_tilde_j, k)

    # --- Step 5: Assign weights to each of the 4 pairs ---

    # create two arrays.
    # One for all the y_i candidates and one for all the y_j candidates.
    y_i_cands = np.array([y_i_minus, y_i_minus, y_i_plus, y_i_plus])
    y_j_cands = np.array([y_j_minus, y_j_plus, y_j_minus, y_j_plus])

    # Now call f_y only twice on the entire arrays.
    weights_i = f_y(y_i_cands, mu_current, mu_star, k)
    weights_j = f_y(y_j_cands, mu_current, mu_star, k)

    # The final weights are the element-wise product of the two arrays.
    weights = weights_i * weights_j

    # --- Step 6: Sample a pair  ---
    sum_of_weights = np.sum(weights)
    if sum_of_weights <= 0 or np.isnan(sum_of_weights):
        return x_current, False, z_i_accepted

    probs = weights / sum_of_weights

    # We need the original list of tuples to select the winner from
    candidate_y_pairs = [(y_i_minus, y_j_minus), (y_i_minus, y_j_plus), (y_i_plus, y_j_minus), (y_i_plus, y_j_plus)]
    chosen_index = np.random.choice(4, p=probs)
    y_i_new, y_j_new = candidate_y_pairs[chosen_index]

    x_i_new = y_i_new + mu_star
    x_j_new = y_j_new + mu_star

    x_new = np.copy(x_current)
    x_new[i], x_new[j] = x_i_new, x_j_new

    return x_new, True, z_i_accepted

def update_x_full(x_current, mu_current, params):
    """
    Performs a full systematic scan to update the entire x vector.

    Returns:
        (np.array, int, int): A tuple containing:
                              - The new, fully updated data vector.
                              - The number of accepted PAIRS.
                              - The number of accepted Z_I proposals.
    """
    x_to_update = np.copy(x_current)
    m = params['m']
    pair_accepted_count = 0
    z_i_accepted_count = 0

    shuffled_indices = np.random.permutation(m)

    for i in range(0, m, 2):
        if i + 1 < m:
            indices = (shuffled_indices[i], shuffled_indices[i+1])

            x_to_update, pair_accepted, z_i_accepted = update_one_pair(
                x_to_update, indices, mu_current, params
            )
            if pair_accepted:
                pair_accepted_count += 1
            if z_i_accepted:
                z_i_accepted_count += 1

    return x_to_update, pair_accepted_count, z_i_accepted_count

def run_main_gibbs_sampler(params, mu_0, x_0):
    """
    Runs the entire Insufficient Gibbs Sampler loop and returns the results.

    Args:
        params (dict): Dictionary of all model and sampler parameters.
        mu_0 (float): The starting value for the mu chain.
        x_0 (np.array): The starting vector for the latent data x.

    Returns:
        dict: A dictionary containing all the output chains and final statistics.
    """
    print("\n--- Starting the Main Gibbs Sampler ---")
    T = params['num_iterations_T']

    # --- Containers for MCMC chains and statistics ---
    mu_chain = np.zeros(T)
    mu_chain[0] = mu_0

    # x_mean_chain = np.zeros(T)
    # x_std_chain = np.zeros(T)
    # x_quartiles_chain = np.zeros((T, 3))

    # Initialize stats at t=0 from the provided x_0
    x_current = x_0.copy() # Use a copy to avoid modifying the original
    # x_mean_chain[0] = np.mean(x_current)
    # x_std_chain[0] = np.std(x_current)
    # x_quartiles_chain[0, :] = np.percentile(x_current, [25, 50, 75])

    # Counters for acceptance rates
    mu_acceptance_count = 0
    # x_pair_acceptance_count = 0
    z_i_acceptance_count = 0
    total_pairs_attempted = (T - 1) * (params['m'] // 2)

    # The main loop
    for t in tqdm(range(1, T), desc="Running Gibbs Sampler"):
        # --- Step (a): Sample mu(t) ---
        mu_previous = mu_chain[t-1]
        mu_new = update_mu_metropolis(mu_previous, x_current, params)
        mu_chain[t] = mu_new
        if mu_new != mu_previous:
            mu_acceptance_count += 1

        # --- Step (b): Sample x(t) ---
        x_new, accepted_pairs, accepted_z_is = update_x_full(x_current, mu_new, params)
        x_current = x_new # Update x_current for the next iteration

        # x_pair_acceptance_count += accepted_pairs
        z_i_acceptance_count += accepted_z_is

        # --- Record statistics for the new x_current ---
        # x_mean_chain[t] = np.mean(x_current)
        # x_std_chain[t] = np.std(x_current)
        # x_quartiles_chain[t, :] = np.percentile(x_current, [25, 50, 75])

    # Calculate final rates
    mu_acceptance_rate = mu_acceptance_count / (T - 1)
    # x_pair_acceptance_rate = x_pair_acceptance_count / total_pairs_attempted if total_pairs_attempted > 0 else 0
    z_i_acceptance_rate = z_i_acceptance_count / total_pairs_attempted if total_pairs_attempted > 0 else 0

    print(f"\n--- Sampling Complete ---")
    print(f"Mu Acceptance Rate: {mu_acceptance_rate:.4f}")
    print(f"Z_i Acceptance Rate: {z_i_acceptance_rate:.4f}")
    
    results = {
        "mu_acceptance_rate": mu_acceptance_rate,
        # "x_pair_acceptance_rate": x_pair_acceptance_rate,
        "z_i_acceptance_rate": z_i_acceptance_rate,
        "mu_chain": mu_chain,
        # "x_mean_chain": x_mean_chain,
        # "x_std_chain": x_std_chain,
        # "x_quartiles_chain": x_quartiles_chain,
    }
    
    return results

def generate_initial_data(params):
    """
    Generates the initial "ground-truth" dataset and finds its MLE, μ*.
    """
    print("--- Generating initial data and finding MLE (μ*) ---")
    np.random.seed(42)
    x_original = stats.t.rvs(
        df=params['k'],
        loc=params['mu_true'],
        scale=1,
        size=params['m']
    )
    mu_star = get_mle(x_original, params)
    print(f"Generated {params['m']} data points with true μ = {params['mu_true']}.")
    print(f"Calculated MLE μ* = {mu_star:.4f}")
    return x_original, mu_star

def initialize_sampler(params):
    """
    Initializes the starting state (μ_0, x_0) for the Gibbs Sampler.

    Args:
        params (dict): Must contain 'mu_star' and 'm'.

    Returns:
        (float, np.array): A tuple containing the starting mu and starting x vector.
    """
    print("--- Initializing sampler state ---")
    mu_0 = params['mu_star']
    x_0 = np.full(shape=params['m'], fill_value=params['mu_star'])
    return mu_0, x_0

def compute_benchmark_mle_samples(params, num_simulations=10000):
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
    print("(This is computationally intensive and will take some time...)")
    
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
        
    print("Benchmark MLE samples computed successfully.")
    return mle_samples

def generate_all_predictive_samples(mu_chain_mle, mu_chain_full_data, params):
    """
    Generates two sets of posterior predictive samples for x:
    1. Based on the posterior of mu given the MLE (from Gibbs Sampler)
    2. Based on the posterior of mu given the full data

    Returns:
        (np.array, np.array): A tuple containing the two predictive datasets:
                              (x_pred_mle, x_pred_full_data)
    """
    print("\n--- Generating Posterior Predictive Datasets ---")
    
    num_x_samples = 10000  # Number of posterior predictive draws
    burn_in = int(params['num_iterations_T'] * 0.2)

    # --- Generate samples for x_tilde | mu* ---
    print(f"Generating {num_x_samples} samples from p(x|μ*)...")
    posterior_mle_mus = mu_chain_mle[burn_in:]
    mus_for_mle_pred = np.random.choice(posterior_mle_mus, size=num_x_samples, replace=True)
    x_pred_mle = np.array([
        stats.t.rvs(df=params['k'], loc=mu, scale=1, size=1)
        for mu in tqdm(mus_for_mle_pred, desc="Sampling x|μ*")
    ]).flatten()

    # --- Generate samples for x_tilde | x_original ---
    print(f"Generating {num_x_samples} samples from p(x|x_original)...")
    posterior_full_data_mus = mu_chain_full_data[burn_in:]
    mus_for_full_data_pred = np.random.choice(posterior_full_data_mus, size=num_x_samples, replace=True)
    x_pred_full_data = np.array([
        stats.t.rvs(df=params['k'], loc=mu, scale=1, size=1)
        for mu in tqdm(mus_for_full_data_pred, desc="Sampling x|x")
    ]).flatten()
    
    return x_pred_mle, x_pred_full_data

def run_full_data_sampler(params, x_data, start_mu):
    """
    Runs a simple M-H sampler for mu given a fixed, full dataset.

    This generates samples from the "gold standard" posterior p(μ|x) for
    comparison against the Insufficient Gibbs Sampler.

    Args:
        params (dict): Dictionary of model and sampler parameters.
        x_data (np.array): The fixed, complete dataset to condition on.
        start_mu (float): The starting value for the mu chain.

    Returns:
        np.array: The MCMC chain of mu samples.
    """
    print("\n--- Running Sampler for Full-Data Posterior p(μ|x) ---")
    T = params['num_iterations_T']

    # Create a container for this chain's samples
    mu_chain = np.zeros(T)
    mu_chain[0] = start_mu

    # This is a simple loop that repeatedly calls the mu-updater
    for t in tqdm(range(1, T), desc="Sampling p(μ|x)"):
        mu_previous = mu_chain[t-1]
        
        # At each step, we update mu based on the same, fixed, full dataset
        mu_next = update_mu_metropolis(
            mu_current=mu_previous,
            x_current=x_data, # The key difference: x is fixed to the full data
            params=params
        )
        
        mu_chain[t] = mu_next

    print("Full-data posterior sampling complete.")
    return mu_chain

def generate_datasets(params, num_outliers=1, outlier_percentile=0.995, epsilon=1e-4):
    """
    Generates two datasets: x1 (clean) and x2 (with a guaranteed number of outliers).
    x1 and x2 will be identical except for a specified number of elements.
    All non-outlier samples are constrained to be within the 25th to 75th percentile range.
    Outliers are generated to be very close to the defined threshold.

    Args:
        params (dict): Requires 'k', 'm', 'mu_true'.
                       k: Degrees of freedom for the t-distribution.
                       m: Number of samples in each dataset.
                       mu_true: True mean (location) of the t-distribution.
        num_outliers (int): The number of outliers to include in the x2 dataset.
        outlier_percentile (float): The percentile threshold to define an outlier.
                                    A value above this percentile is considered an outlier.
        epsilon (float): A small value added to the outlier threshold to ensure the
                         generated outlier is just slightly above it.

    Returns:
        (np.array, np.array, float): A tuple containing:
                                     - x1: The clean dataset.
                                     - x2: The dataset with outliers.
                                     - L: The calculated outlier threshold.
    """
    k = params['k']
    m = params['m']
    mu_true = params['mu_true']
    
    if num_outliers >= m:
        raise ValueError("The number of outliers must be less than the total number of samples.")

    # 1. Define the outlier threshold L
    L = stats.t.ppf(outlier_percentile, df=k, loc=mu_true, scale=1)

    # 2. Define the 25th and 75th percentiles for the "clean" range
    Q1 = stats.t.ppf(0.25, df=k, loc=mu_true, scale=1)
    Q3 = stats.t.ppf(0.75, df=k, loc=mu_true, scale=1)
    
    # 3. Generate m - num_outliers common samples for both datasets
    common_samples = []
    while len(common_samples) < m - num_outliers:
        sample = stats.t.rvs(df=k, loc=mu_true, scale=1, size=1)[0]
        if Q1 <= sample <= Q3: 
            common_samples.append(sample)
    common_samples = np.array(common_samples)
    
    # 4. Generate num_outliers non-outlier samples for x1
    non_outlier_samples = []
    while len(non_outlier_samples) < num_outliers:
        non_outlier_sample = stats.t.rvs(df=k, loc=mu_true, scale=1, size=1)[0]
        if Q1 <= non_outlier_sample <= Q3:
            non_outlier_samples.append(non_outlier_sample)
            
    # 5. Generate num_outliers outlier samples for x2
    outlier_samples = [L + epsilon for _ in range(num_outliers)]
            
    # 6. Construct x1 (the clean dataset)
    x1 = np.append(common_samples, non_outlier_samples)
    np.random.shuffle(x1)
            
    # 7. Construct x2 (the dataset with outliers)
    x2 = np.append(common_samples, outlier_samples)
    np.random.shuffle(x2)
    
    print(f"Outlier threshold L (>{outlier_percentile*100:.1f}th percentile) = {L:.4f}")
    print(f"25th percentile (Q1): {Q1:.4f}, 75th percentile (Q3): {Q3:.4f}")
    print(f"Max value in clean dataset (x1): {np.max(x1):.4f}")
    print(f"Min value in clean dataset (x1): {np.min(x1):.4f}")
    print(f"Max value in outlier dataset (x2): {np.max(x2):.4f}")
    
    return x1, x2, L

def generate_clean_dataset_centered(params):
    """
    Generates a 'clean' dataset where the MLE is guaranteed to be mu_true.

    Args:
        params (dict): Requires 'k', 'm', 'mu_true'.
                       k: Degrees of freedom for the t-distribution.
                       m: Number of samples in the dataset.
                       mu_true: The desired MLE for the final dataset.

    Returns:
        np.array: The generated clean dataset with MLE = mu_true.
    """
    k = params['k']
    m = params['m']
    mu_true = params['mu_true']

    # 1. Define the 25th and 75th percentiles for the "clean" range
    Q1 = stats.t.ppf(0.25, df=k, loc=mu_true, scale=1)
    Q3 = stats.t.ppf(0.75, df=k, loc=mu_true, scale=1)

    # 2. Generate m initial samples within the interquartile range
    initial_samples = []
    while len(initial_samples) < m:
        sample = stats.t.rvs(df=k, loc=mu_true, scale=1, size=1)[0]
        if Q1 <= sample <= Q3:
            initial_samples.append(sample)
    initial_samples = np.array(initial_samples)

    # 3. Find the initial MLE of this dataset using the provided function
    initial_mle = get_mle(initial_samples, params)

    # 4. Calculate the shift required to move the MLE to mu_true
    shift = mu_true - initial_mle
    
    # 5. Apply the shift to the entire dataset
    centered_dataset = initial_samples + shift
    
    return centered_dataset

def generate_constrained_outlier_pair(clean_dataset, indices, mu_star, k, params):
    """
    Takes a dataset and two indices (i, j), sets x[i] to a random outlier value,
    and calculates a new x[j] to keep the MLE constant. The outlier is sampled
    from either the high tail or the low tail with 50% probability.
    """
    i, j = indices
    x_i, x_j = clean_dataset[i], clean_dataset[j]

    # 1. Transform original pair to y-space and then z-space
    y_i = x_i - mu_star
    y_j = x_j - mu_star
    z_i = psi(y_i, k)
    z_j = psi(y_j, k)
    
    # 2. Calculate the invariant sum delta
    delta = z_i + z_j

    # --- MODIFIED SECTION ---
    # 3. Loop to find a valid outlier value
    max_attempts = 100000
    for attempt in range(max_attempts):
        # With 50% chance, sample from the upper tail, otherwise sample from the lower tail.
        if np.random.rand() < 0.5:
            # Sample from the upper tail (99.5th to 99.9th percentile)
            p995 = stats.t.ppf(0.995, df=k, loc=mu_star, scale=1)
            p999 = stats.t.ppf(0.999, df=k, loc=mu_star, scale=1)
            outlier_value = np.random.uniform(p995, p999)
        else:
            # Sample from the lower tail (0.1th to 0.5th percentile)
            p001 = stats.t.ppf(0.001, df=k, loc=mu_star, scale=1)
            p005 = stats.t.ppf(0.005, df=k, loc=mu_star, scale=1)
            outlier_value = np.random.uniform(p001, p005)

        # Transform the new outlier point
        x_tilde_i = outlier_value
        y_tilde_i = x_tilde_i - mu_star
        z_tilde_i = psi(y_tilde_i, k)

        # Calculate the required z for the second point
        z_tilde_j = delta - z_tilde_i

        # Check if the new z is within the valid domain from params
        z_domain_limit = params['z_domain'][1]
        if -z_domain_limit < z_tilde_j < z_domain_limit:
            # Found a valid z_tilde_j, break the loop and proceed
            break
    else:
        # This block executes if the loop completes without a 'break'
        print(f"Warning: Failed to find a valid outlier pair after {max_attempts} attempts.")
        return None
    # --- END OF MODIFIED SECTION ---

    # 6. Invert z_tilde_j back to y-space
    y_minus, y_plus = psi_inverse(z_tilde_j, k)
    
    if np.isnan(y_minus):
        print("Warning: Could not find a valid inverse for z_tilde_j.")
        return None

    # 7. Choose the y-value closer to the original y_j
    if abs(y_minus - y_j) < abs(y_plus - y_j):
        y_tilde_j = y_minus
    else:
        y_tilde_j = y_plus
        
    # 8. Transform back to x-space
    x_tilde_j = y_tilde_j + mu_star

    # 9. Create and return the new dataset
    outlier_dataset = clean_dataset.copy()
    outlier_dataset[i] = x_tilde_i
    outlier_dataset[j] = x_tilde_j
    
    return outlier_dataset







