# main.py
# This script orchestrates a single, complete run of the Insufficient Gibbs Sampler
# for a given k and m, controlled via command-line arguments.

import os
import numpy as np
import pandas as pd
import argparse  # The library for command-line arguments
import sampler_functions as sf
import analysis as an

def run_single_experiment(k, m):
    """
    Main function to run the entire experimental pipeline for one (k, m) pair.
    """
    # --- 1. SETUP PHASE ---
    print(f"--- Starting Experiment: k={k}, m={m} ---")

    # Create results directories if they don't exist
    base_results_dir = f"results/k_{k}_m_{m}"
    os.makedirs(base_results_dir, exist_ok=True)

    # Define parameters for this specific run
    params = {
        'm': m,
        'k': k,
        'mu_true': 10.0,
        'prior_mean': 0.0,
        'prior_std': 20.0,
        'num_iterations_T': 30000,
        'proposal_std_mu': 0.9,
        'proposal_std_z': 0.07
    }

    # Generate the initial "observed" data and its MLE
    x_original, mu_star = sf.generate_initial_data(params)
    params['mu_star'] = mu_star
    
    # Add z_domain to params, as it depends on k
    z_domain_half_width = 1 / (2 * np.sqrt(params['k']))
    params['z_domain'] = (-z_domain_half_width, z_domain_half_width)

    # Initialize the sampler's starting state
    mu_0, x_0 = sf.initialize_sampler(params)
    
    print(f"Setup complete. Observed μ* = {mu_star:.4f}")

    # --- 2. EXECUTION PHASE ---
    
    # Run the main Insufficient Gibbs Sampler to get p(μ|μ*)
    gibbs_results = sf.run_main_gibbs_sampler(params, mu_0, x_0)
    mu_chain_mle = gibbs_results['mu_chain']
    
    # Run a separate, simple MCMC to get the "gold standard" posterior p(μ|x)
    mu_chain_full_data = sf.run_full_data_sampler(params, x_original, mu_0)
    
    # Generate the two sets of posterior predictive samples for x
    x_pred_mle_flat, x_pred_full_data_flat = sf.generate_all_predictive_samples(
        mu_chain_mle, mu_chain_full_data, params
    )
    
    # --- 3. ANALYSIS & REPORTING PHASE ---
    print("\n--- Generating All Analysis Reports ---")

    # Generate the posterior comparison plot (Experiment 1)
    an.plot_posterior_comparison(
        mle_chain=mu_chain_mle, 
        x_original=x_original, 
        params=params, 
        filename=f"{base_results_dir}/posterior_comparison.png"
    )
    
    # Create and save the statistics comparison table (Experiment 2)
    an.create_x_comparison_table(
        x_original, 
        x_pred_full_data=x_pred_full_data_flat, 
        x_pred_mle=x_pred_mle_flat,
        filename=f"{base_results_dir}/stats_comparison_table.html"
    )

    # Generate the posterior predictive comparison plot (Experiment 2)
    an.plot_x_distribution_comparison(
        x_pred_full_data=x_pred_full_data_flat, 
        x_pred_mle=x_pred_mle_flat, 
        filename=f"{base_results_dir}/x_pred_comparison.png"
    )

    # Generate the difference plot
    an.plot_predictive_density_difference(
        x_pred_full_data=x_pred_full_data_flat,
        x_pred_mle=x_pred_mle_flat,
        filename=f"{base_results_dir}/x_pred_difference.png"
    )

    # --- 4. SAVE SUMMARY RESULTS ---
    # Calculate key summary statistics for the master results table
    burn_in = int(params['num_iterations_T'] * 0.2)
    std_dev_mle_only = np.std(mu_chain_mle[burn_in:])
    std_dev_full_data = np.std(mu_chain_full_data[burn_in:])
    info_loss_ratio = std_dev_mle_only / std_dev_full_data if std_dev_full_data > 0 else np.nan

    summary_data = {
        'k': k,
        'm': m,
        'mu_star': mu_star,
        'std_dev_full_data': std_dev_full_data,
        'std_dev_mle_only': std_dev_mle_only,
        'info_loss_ratio': info_loss_ratio,
        'mu_acceptance_rate': gibbs_results['mu_acceptance_rate'],
        'x_acceptance_rate': gibbs_results['x_pair_acceptance_rate'],
    }

    # Save human-readable summary to a text file
    summary_filename = f"{base_results_dir}/summary_report.txt"
    with open(summary_filename, 'w') as f:
        f.write(f"Summary for k={k}, m={m}\n")
        f.write("="*30 + "\n")
        for key, val in summary_data.items():
            f.write(f"{key}: {val:.4f}\n")
    print(f"Summary report saved to {summary_filename}")

    # Return summary data to be appended to the master CSV
    return summary_data


if __name__ == '__main__':
    # --- Setup Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run the Insufficient Gibbs Sampler for a given k and m.")
    parser.add_argument('--k', type=float, required=True, help="Degrees of freedom for the t-distribution.")
    parser.add_argument('--m', type=int, required=True, help="Sample size.")
    args = parser.parse_args()

    # --- Input Validation ---
    if args.k <= 2:
        print(f"Error: Degrees of freedom k must be greater than 2. Got k={args.k}.")
        exit(1)

    # --- Run the Experiment ---
    summary = run_single_experiment(k=args.k, m=args.m)

    # --- Append results to a master CSV file ---
    master_results_file = "results/master_results.csv"
    df_summary = pd.DataFrame([summary])
    
    # If the file doesn't exist, write the header first
    if not os.path.exists(master_results_file):
        df_summary.to_csv(master_results_file, index=False, mode='w', header=True)
    else: # Otherwise, append without the header
        df_summary.to_csv(master_results_file, index=False, mode='a', header=False)
        
    print(f"\nResults for (k={args.k}, m={args.m}) appended to {master_results_file}")
    print("\n--- Experiment Run Complete ---")