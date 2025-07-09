# main_outlier_experiment.py
# This script is a command-line tool to run the outlier robustness experiment
# for a single, specified (k, m) pair.

import os
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

# Import our custom function modules
import sampler_functions as sf
import outlier_functions as oef
import analysis as an

def run_single_outlier_experiment(k, m):
    """
    Runs the full outlier experiment pipeline for one (k, m) pair.
    """
    # --- Per-run Setup ---
    print(f"\n{'='*60}")
    print(f"--- Starting Outlier Experiment: k={k}, m={m} ---")
    print(f"{'='*60}")
    
    base_results_dir = f"results/outlier/outlier_k_{k}_m_{m}"
    os.makedirs(base_results_dir, exist_ok=True)

    params = {
        'm': m, 'k': float(k), 'mu_true': 10.0,
        'prior_mean': 0.0, 'prior_std': 20.0,
        'num_iterations_T': 10000,
        'proposal_std_mu': 0.9, 'proposal_std_z': 0.07
    }
    z_domain_half_width = 1 / (2 * np.sqrt(k))
    params['z_domain'] = (-z_domain_half_width, z_domain_half_width)

    # --- Step 1: Generate Datasets ---
    x1_clean, x2_with_outlier, outlier_threshold_L = oef.generate_datasets(params)
    
    # --- Step 2: Calculate MLEs ---
    mu_star_1 = oef.get_mle(x1_clean, params)
    mu_star_2 = oef.get_mle(x2_with_outlier, params)
    
    # --- Step 3: Generate the Four μ Posteriors ---
    print("\nGenerating the four posterior distributions for μ...")
    mu_chain_x1 = oef.get_full_data_posterior_samples(x1_clean, params)
    mu_chain_x2 = oef.get_full_data_posterior_samples(x2_with_outlier, params)
    mu_chain_mle1 = oef.get_mle_conditional_posterior_samples(mu_star_1, params)
    mu_chain_mle2 = oef.get_mle_conditional_posterior_samples(mu_star_2, params)

    # --- Step 4: Generate the Four Posterior Predictive Distributions for x̃ ---
    print("\nGenerating posterior predictive samples for analysis...")
    x_pred_x1 = oef.generate_predictive_samples(mu_chain_x1, params)
    x_pred_x2 = oef.generate_predictive_samples(mu_chain_x2, params)
    x_pred_mle1 = oef.generate_predictive_samples(mu_chain_mle1, params)
    x_pred_mle2 = oef.generate_predictive_samples(mu_chain_mle2, params)
    
    # --- Step 5: Calculate Final Probabilities ---
    prob_x1 = oef.calculate_outlier_probability(x_pred_x1, outlier_threshold_L)
    prob_x2 = oef.calculate_outlier_probability(x_pred_x2, outlier_threshold_L)
    prob_mle1 = oef.calculate_outlier_probability(x_pred_mle1, outlier_threshold_L)
    prob_mle2 = oef.calculate_outlier_probability(x_pred_mle2, outlier_threshold_L)
    
    # --- Step 6: Generate and Save All Reports ---
    print("\n--- Generating Analysis Reports ---")
    
    # Save the plot of the predictive densities
    an.plot_outlier_predictive_distributions(
        x_pred_x1, x_pred_x2, x_pred_mle1, x_pred_mle2,
        outlier_threshold_L,
        filename_prefix=f"{base_results_dir}"
    )
    
    # Package results for the summary table
    summary_data = {
        'k': k, 'm': m,
        'mu_star_clean': mu_star_1, 'mu_star_outlier': mu_star_2,
        'prob_outlier_given_x1': prob_x1,
        'prob_outlier_given_x2': prob_x2,
        'prob_outlier_given_mle1': prob_mle1,
        'prob_outlier_given_mle2': prob_mle2,
    }
    
    # Save a human-readable summary text file
    report_filename = f"{base_results_dir}/outlier_summary.txt"
    with open(report_filename, 'w') as f:
        f.write(f"Summary for k={k}, m={m}\n{'='*30}\n")
        for key, val in summary_data.items():
            f.write(f"{key}: {val:.4%}\n" if "prob" in key else f"{key}: {val:.4f}\n")
    print(f"Summary report saved to {report_filename}")
        
    return summary_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the outlier robustness experiment.")
    parser.add_argument('--k', type=float, required=True, help="Degrees of freedom.")
    parser.add_argument('--m', type=int, required=True, help="Sample size.")
    args = parser.parse_args()

    summary = run_single_outlier_experiment(k=args.k, m=args.m)

    master_results_file = "results/outlier_experiment_master_results.csv"
    df_summary = pd.DataFrame([summary])
    os.makedirs("results", exist_ok=True)
    
    if not os.path.exists(master_results_file):
        df_summary.to_csv(master_results_file, index=False, mode='w', header=True)
    else:
        df_summary.to_csv(master_results_file, index=False, mode='a', header=False)
        
    print(f"\nResults for (k={args.k}, m={args.m}) appended to {master_results_file}")
    print("\n--- Experiment Run Complete ---")