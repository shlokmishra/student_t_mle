import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import random
import pickle

# --- 1. Import Custom Function Modules ---
try:
    import sampler_functions as sf
    import outlier_functions as oef
    import analysis as an
except ImportError:
    print("Error: Make sure 'sampler_functions.py', 'outlier_functions.py', and 'analysis_functions.py' are in the same directory.")
    exit()

## ------------------------------------------------------------------
## Main Execution
## ------------------------------------------------------------------

def main(args):
    """
    Main function to run the outlier analysis experiment.
    """
    # --- 2. Define Experiment Parameters ---
    params = {
        'm': args.m,
        'k': args.k,
        'mu_true': 10.0,
        'num_iterations_T': 30000,
        'prior_mean': 0.0,
        'prior_std': 20.0,
        'proposal_std_mu': 0.9,
    }
    # Add z_domain to params, as it depends on k
    z_domain_half_width = 1 / (2 * np.sqrt(params['k']))
    params['z_domain'] = (-z_domain_half_width, z_domain_half_width)
    
    # print("--- Experiment Parameters ---")
    # for key, val in params.items():
    #     print(f"{key}: {val}")
    # print(f"num_outliers: {args.num_outliers}")

    # --- 3. Create Custom Results Directory ---
    # Structure: results_outlier/{num_outliers}_outliers/k_{k}_m_{m}/
    results_path = os.path.join(
        "results_outlier_new",
        f"k_{args.k}_m_{args.m}"
    )
    os.makedirs(results_path, exist_ok=True)
    print(f"\nResults will be saved in: '{results_path}'")
    print("-" * 30)
    
    print("--- Generating Clean, Centered Dataset ---")
    x1_clean = sf.generate_clean_dataset_centered(params)
    print("Clean dataset generated successfully.")
    print(f"Verified MLE of clean dataset: {sf.get_mle(x1_clean, params):.6f}\n")


    # --- NEW SEQUENTIAL GENERATION LOGIC ---
    print("\n--- Generating 5 Sequential Outlier Datasets ---")
    outlier_datasets = []
    num_outlier_sets_to_generate = 5

    # Start with the clean dataset as the base for the first modification
    current_dataset = x1_clean.copy() 
    # Keep track of indices that are still "clean" and can be modified
    available_indices = list(range(params['m']))

    for i in range(num_outlier_sets_to_generate):
        # Check if we have enough clean points left to form a pair
        if len(available_indices) < 2:
            print(f"Not enough available indices to continue. Generated {i} outlier sets.")
            break

        print(f"--- Generating Outlier Set {i+1} (Total non-clean points: {(i+1)*2}) ---")
        
        # Randomly select two available indices to form the new pair
        indices_to_update = tuple(random.sample(available_indices, 2))
        idx1, idx2 = indices_to_update
        
        # Generate the next outlier dataset based on the CURRENT one
        x_outlier = sf.generate_constrained_outlier_pair(
            clean_dataset=current_dataset,
            indices=indices_to_update,
            mu_star=params['mu_true'],
            k=params['k'],
            params=params
        )
        
        if x_outlier is not None:
            # Add the new dataset to our list
            outlier_datasets.append(x_outlier)
            
            # --- ADDED PRINT STATEMENTS ---
            original_val1 = current_dataset[idx1]
            original_val2 = current_dataset[idx2]
            new_val1 = x_outlier[idx1]
            new_val2 = x_outlier[idx2]
            print(f"  Modifying indices: ({idx1}, {idx2})")
            print(f"  Original values: x[{idx1}]={original_val1:.2f}, x[{idx2}]={original_val2:.2f}")
            print(f"  New values:      x[{idx1}]={new_val1:.2f}, x[{idx2}]={new_val2:.2f}")
            
            # --- CORRECTED VERIFICATION STEP ---
            # Instead of re-calculating the MLE, we verify that the score at mu_true is ~0.
            # This is more stable than root-finding on a complex distribution.
            score_at_mu_true = sf.mle_score(params['mu_true'], x_outlier, params['k'])
            print(f"  Score at mu_true={params['mu_true']:.1f}: {score_at_mu_true:.6e}\n")

            # Update the current dataset for the next iteration
            current_dataset = x_outlier.copy()
            
            # Remove the used indices from the available pool
            available_indices.remove(idx1)
            available_indices.remove(idx2)
        else:
            print(f"Failed to generate a valid pair for indices ({idx1}, {idx2}). Stopping generation.")
            break

    print(f"\nSuccessfully generated {len(outlier_datasets)} sequential outlier datasets.")




    plot_path = os.path.join(results_path, "mle_comparison_plot.png")
    an.plot_dataset_comparison(
        x1_clean, outlier_datasets, params, save_path=plot_path
    )
    print("\n--- Loading/Computing Benchmark KDE for p(μ̂ | μ=0) ---")
    
    kde_path = os.path.join(
        "results_outlier_new")

    # Define the path where the saved KDE object is stored
    kde_save_path = os.path.join(kde_path, "kde_0_benchmark.pkl")

    print("\n--- Loading/Computing Benchmark Data for p(μ̂ | μ=0) ---")

    # Define the path where the saved MLE samples will be stored
    mle_samples_save_path = os.path.join(kde_path, "benchmark_mle_samples.pkl")

    # Check if the file with the raw data already exists
    if os.path.exists(mle_samples_save_path):
        # If it exists, load the pre-computed samples
        print(f"Loading pre-computed benchmark samples from: {mle_samples_save_path}")
        with open(mle_samples_save_path, 'rb') as f:
            mle_samples = pickle.load(f)
    else:
        # If it doesn't exist, run the expensive simulation using the new function
        print("No pre-computed samples found. Running simulation...")
        mle_samples = sf.compute_benchmark_mle_samples(params, num_simulations=50000)
        
        # Save the newly computed samples for future runs
        with open(mle_samples_save_path, 'wb') as f:
            pickle.dump(mle_samples, f)
        print(f"Benchmark samples computed and saved to: {mle_samples_save_path}")

    print("\nBuilding Benchmark KDE from samples...")
    kde_0 = stats.gaussian_kde(mle_samples)
    print("Benchmark KDE is ready.")


    # --- Calculate Benchmark Posterior using the loaded/computed KDE ---
    print("\n--- Calculating Benchmark Posterior ---")
    mu_grid = np.linspace(params['mu_true'] - 8, params['mu_true'] + 8, 1000)
    # The likelihood p(μ̂ | μ) is equivalent to p(μ̂ - μ | 0).
    likelihood_kde = kde_0.pdf(params['mu_true'] - mu_grid)
    prior_on_grid = stats.norm.pdf(mu_grid, loc=params['prior_mean'], scale=params['prior_std'])
    unnormalized_posterior_kde = likelihood_kde * prior_on_grid
    area = np.trapz(unnormalized_posterior_kde, mu_grid)
    benchmark_posterior = unnormalized_posterior_kde / area
    print("Benchmark posterior calculated successfully.")

    
    print("\n--- Generating Full Data Posterior Samples for Each Dataset ---")
    all_datasets = [x1_clean] + outlier_datasets
    posterior_chains = []
    for i, dataset in enumerate(all_datasets):
        if i == 0:
            print(f"Running sampler for Clean Dataset...")
        else:
            print(f"Running sampler for Outlier Set {i}...")
        # This calls the function from outlier_functions.py
        chain = oef.get_full_data_posterior_samples(dataset, params)
        posterior_chains.append(chain)
    print("All posterior chains generated.")



    # --- 7. Generate All Four Predictive Distributions for x ---
    print("\n--- Generating Posterior Predictive Distributions ---")
    predictive_chains = []
    # Model parameters
    k = params['k']
    scale = 1
    m = params['m']

    for i, mu_chain in enumerate(posterior_chains):
        if i == 0:
            print(f"Generating predictive samples for Clean Dataset chain...")
        else:
            print(f"Generating predictive samples for Outlier Set {i} chain...")
        
        # For each mu in the posterior chain, generate a new dataset of size m
        predictive_datasets = [stats.t.rvs(df=k, loc=mu, scale=scale, size=m) for mu in mu_chain]
        
        # Flatten the list of datasets into a single array of predictive samples
        x_pred = np.array(predictive_datasets).flatten()
        predictive_chains.append(x_pred)

    print("All posterior predictive chains generated.")


    ## ------------------------------------------------------------------
    ## --- Detailed Analysis and Plotting ---
    ## ------------------------------------------------------------------
    print("Generating detailed analysis plots...")

    posterior_plot_path = os.path.join(results_path, "posterior_comparison.png")

    # Call the new function
    an.plot_posterior_comparison(
        mu_grid=mu_grid,
        benchmark_posterior=benchmark_posterior,
        posterior_chains=posterior_chains,
        params=params,
        save_path=posterior_plot_path
    )


# First, construct the full path for the output file
    predictive_plot_path = os.path.join(results_path, "posterior_predictive_comparison.png")

    # a) Generate a sample chain of mu values from the benchmark posterior distribution
    print("\n--- Generating Samples from Benchmark Posterior ---")
    # We sample from the mu_grid, with probabilities given by the benchmark density.
    num_samples = params['num_iterations_T'] # Match length of other chains

    # Ensure probabilities sum to 1 for np.random.choice
    probabilities = benchmark_posterior / np.sum(benchmark_posterior)

    mu_chain_from_mle = np.random.choice(
        mu_grid,
        size=num_samples,
        p=probabilities,
        replace=True # Allow sampling mu values more than once
    )
    print("Samples from benchmark posterior generated.")


    # b) Generate the predictive chain p(x|μ*) from the mu samples
    print("\n--- Generating Predictive Chain from MLE's Posterior ---")
    # Model parameters
    k = params['k']
    scale = 1
    m = params['m']

    predictive_datasets_from_mle = [stats.t.rvs(df=k, loc=mu, scale=scale, size=m) for mu in mu_chain_from_mle]
    x_pred_from_mle = np.array(predictive_datasets_from_mle).flatten()
    print("Predictive chain from MLE's posterior generated.")

    # Now, call the function with all the required data
    an.plot_predictive_comparison(
        predictive_chains=predictive_chains,
        x_pred_from_mle=x_pred_from_mle,
        params=params,
        save_path=predictive_plot_path
    )

    print("\n--- Saving All Generated Chains to Files ---")

    # 1. Save the list of posterior chains (mu chains)
    posterior_chains_path = os.path.join(results_path, "posterior_chains.pkl")
    with open(posterior_chains_path, 'wb') as f:
        pickle.dump(posterior_chains, f)
    print(f"Posterior chains saved to: {posterior_chains_path}")

    # 2. Save the list of predictive chains from the full data
    predictive_chains_path = os.path.join(results_path, "predictive_chains_from_data.pkl")
    with open(predictive_chains_path, 'wb') as f:
        pickle.dump(predictive_chains, f)
    print(f"Predictive chains from full data saved to: {predictive_chains_path}")

    # 3. Save the single predictive chain generated from the MLE
    predictive_mle_path = os.path.join(results_path, "predictive_chain_from_mle.pkl")
    with open(predictive_mle_path, 'wb') as f:
        pickle.dump(x_pred_from_mle, f)
    print(f"Predictive chain from MLE saved to: {predictive_mle_path}")

    print("All chains have been successfully saved.")










if __name__ == "__main__":
    # This block handles command-line argument parsing
    parser = argparse.ArgumentParser(description="Run an outlier sensitivity analysis experiment.")
    
    parser.add_argument('-k', type=float, default=2.0, 
                        help='Degrees of freedom for the t-distribution.')
    parser.add_argument('-m', type=int, default=20, 
                        help='Number of samples in each dataset.')
    
    args = parser.parse_args()
    
    # Run the main function with the parsed arguments
    main(args)



