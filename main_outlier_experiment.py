# main_outlier_experiment.py
# This script is a command-line tool to run the outlier robustness experiment
# for a single, specified (k, m) pair.

import os
import numpy as np
import pandas as pd
import argparse
from IPython.display import display

# Import our custom function modules
import sampler_functions as sf
import outlier_experiment_functions as oef

def run_single_outlier_experiment(k, m):
    """
    Runs the full outlier experiment pipeline for one (k, m) pair.
    
    Args:
        k (float): Degrees of freedom.
        m (int): Sample size.
        
    Returns:
        dict: A dictionary containing the summary results of the run.
    """
    # --- Per-run Setup ---
    print(f"\n{'='*60}")
    print(f"--- Starting Outlier Experiment: k={k}, m={m} ---")
    print(f"{'='*60}")

    params = {
        'm': m, 'k': float(k), 'mu_true': 10.0,
        'prior_mean': 0.0, 'prior_std': 20.0,
        'num_iterations_T': 20000,
        'proposal_std_mu': 0.5, 'proposal_std_z': 0.05
    }
    z_domain_half_width = 1 / (2 * np.sqrt(k))
    params['z_domain'] = (-z_domain_half_width, z_domain_half_width)

    # --- Step 1: Generate Datasets ---
    x1_clean, x2_with_outlier, outlier_threshold_L = oef.generate_datasets(params)
    
    # --- Step 2: Calculate MLEs ---
    mu_star_1 = oef.get_mle(x1_clean, params)
    mu_star_2 = oef.get_mle(x2_with_outlier, params)
    print(f"MLE for clean data μ*(x1) = {mu_star_1:.4f}")
    print(f"MLE for outlier data μ*(x2) = {mu_star_2:.4f}")
    
    # --- Step 3: Generate the Four μ Posteriors ---
    print("\nGenerating the four posterior distributions for μ (this will take time)...")
    mu_chain_x1 = oef.get_full_data_posterior_samples(x1_clean, params)
    mu_chain_x2 = oef.get_full_data_posterior_samples(x2_with_outlier, params)
    mu_chain_mle1 = oef.get_mle_conditional_posterior_samples(mu_star_1, params)
    mu_chain_mle2 = oef.get_mle_conditional_posterior_samples(mu_star_2, params)

    # --- Step 4: Calculate Final Outlier Probabilities ---
    print("\nCalculating posterior predictive outlier probabilities...")
    prob_x1 = oef.calculate_outlier_probability(mu_chain_x1, outlier_threshold_L, params)
    prob_x2 = oef.calculate_outlier_probability(mu_chain_x2, outlier_threshold_L, params)
    prob_mle1 = oef.calculate_outlier_probability(mu_chain_mle1, outlier_threshold_L, params)
    prob_mle2 = oef.calculate_outlier_probability(mu_chain_mle2, outlier_threshold_L, params)

    # --- Step 5: Package Results ---
    summary_data = {
        'k': k, 'm': m,
        'mu_star_clean': mu_star_1, 'mu_star_outlier': mu_star_2,
        'prob_outlier_given_x1': prob_x1, 'prob_outlier_given_x2': prob_x2,
        'prob_outlier_given_mle1': prob_mle1, 'prob_outlier_given_mle2': prob_mle2,
    }
    return summary_data


if __name__ == '__main__':
    # --- Setup Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run the outlier robustness experiment for a given k and m.")
    parser.add_argument('--k', type=float, required=True, help="Degrees of freedom for the t-distribution.")
    parser.add_argument('--m', type=int, required=True, help="Sample size.")
    args = parser.parse_args()

    # --- Input Validation ---
    if args.k <= 2:
        print(f"Error: Degrees of freedom k must be greater than 2. Got k={args.k}. Aborting this run.")
        exit(1)

    # --- Run the Experiment ---
    summary = run_single_outlier_experiment(k=args.k, m=args.m)

    # --- Append results to a master CSV file ---
    master_results_file = "results/outlier_experiment_master_results.csv"
    df_summary = pd.DataFrame([summary])
    
    # Create the main results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # To handle parallel runs safely, we check if the file exists to write the header
    if not os.path.exists(master_results_file):
        df_summary.to_csv(master_results_file, index=False, mode='w', header=True)
    else: # If it exists, append without the header
        df_summary.to_csv(master_results_file, index=False, mode='a', header=False)
        
    print(f"\nResults for (k={args.k}, m={args.m}) appended to {master_results_file}")
    print("\n--- Experiment Run Complete ---")