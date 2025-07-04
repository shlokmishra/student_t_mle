# main_outlier_experiment.py
# This script runs the outlier robustness experiment for a single (k, m) pair.

import numpy as np
import pandas as pd
from IPython.display import display

# Import our custom function modules
import sampler_functions as sf
import outlier_experiment_functions as oef

def main():
    """Main function to run the outlier experiment pipeline."""
    
    # --- 1. SETUP PHASE ---
    print("--- Setting up Outlier Experiment ---")

    # Define parameters for this specific test run
    # Using a slightly smaller T for a quicker test run
    params = {
        'm': 50,
        'k': 5.0,
        'mu_true': 10.0,
        'prior_mean': 0.0,
        'prior_std': 20.0,
        'num_iterations_T': 10000, 
        'proposal_std_mu': 0.5,
        'proposal_std_z': 0.05
    }

    # Add z_domain based on k
    z_domain_half_width = 1 / (2 * np.sqrt(params['k']))
    params['z_domain'] = (-z_domain_half_width, z_domain_half_width)
    
    print(f"Parameters: k={params['k']}, m={params['m']}")

    # --- 2. GENERATE DATASETS (Clean vs. Outlier) ---
    x1_clean, x2_with_outlier, outlier_threshold_L = oef.generate_datasets(params)
    
    # --- 3. CALCULATE MLEs for both datasets ---
    print("\n--- Calculating MLEs for both datasets ---")
    mu_star_1 = oef.get_mle(x1_clean, params)
    mu_star_2 = oef.get_mle(x2_with_outlier, params)
    print(f"MLE of clean data, μ*(x1): {mu_star_1:.4f}")
    print(f"MLE of outlier data, μ*(x2): {mu_star_2:.4f}")

    # --- 4. GENERATE THE FOUR POSTERIORS FOR MU ---
    # This is the most computationally intensive part.
    print("\n--- Generating the four posterior distributions for μ ---")
    
    # Case 1: p(μ | x1) - Posterior from clean, full data
    mu_chain_x1 = oef.get_full_data_posterior_samples(x1_clean, params)
    
    # Case 2: p(μ | x2) - Posterior from outlier, full data
    mu_chain_x2 = oef.get_full_data_posterior_samples(x2_with_outlier, params)
    
    # Case 3: p(μ | μ*(x1)) - Posterior from clean data's MLE
    mu_chain_mle1 = oef.get_mle_conditional_posterior_samples(mu_star_1, params)
    
    # Case 4: p(μ | μ*(x2)) - Posterior from outlier data's MLE
    mu_chain_mle2 = oef.get_mle_conditional_posterior_samples(mu_star_2, params)

    # --- 5. CALCULATE FINAL OUTLIER PROBABILITIES ---
    print("\n--- Calculating final posterior predictive outlier probabilities ---")
    
    prob_x1 = oef.calculate_outlier_probability(mu_chain_x1, outlier_threshold_L, params)
    prob_x2 = oef.calculate_outlier_probability(mu_chain_x2, outlier_threshold_L, params)
    prob_mle1 = oef.calculate_outlier_probability(mu_chain_mle1, outlier_threshold_L, params)
    prob_mle2 = oef.calculate_outlier_probability(mu_chain_mle2, outlier_threshold_L, params)

    # --- 6. REPORT FINAL RESULTS ---
    results_data = {
        'p(x̃ > L | Full Data)': [f"{prob_x1:.3%}", f"{prob_x2:.3%}"],
        'p(x̃ > L | MLE Only)': [f"{prob_mle1:.3%}", f"{prob_mle2:.3%}"]
    }
    index_labels = ["No Outlier Case (x1)", "With Outlier Case (x2)"]
    results_df = pd.DataFrame(results_data, index=index_labels)
    
    # Use our styling function to display the table
    styles = [
        dict(selector="th", props=[("text-align", "center"), ("font-weight", "bold")]),
        dict(selector="td", props=[("text-align", "center")]),
        dict(selector="caption", props=[("caption-side", "top"), ("font-size", "1.2em"), ("font-weight", "bold")])
    ]
    styled_df = results_df.style.set_caption("Posterior Predictive Probability of Generating an Outlier").set_table_styles(styles)
    
    print("\n--- FINAL RESULTS ---")
    display(styled_df)


if __name__ == '__main__':
    main()