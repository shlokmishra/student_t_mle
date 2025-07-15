import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

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
        'num_iterations_T': 100,
        'prior_mean': 0.0,
        'prior_std': 20.0,
        'proposal_std_mu': 0.9,
        'proposal_std_z': 0.2
    }
    # Add z_domain to params, as it depends on k
    z_domain_half_width = 1 / (2 * np.sqrt(params['k']))
    params['z_domain'] = (-z_domain_half_width, z_domain_half_width)
    
    print("--- Experiment Parameters ---")
    for key, val in params.items():
        print(f"{key}: {val}")
    print(f"num_outliers: {args.num_outliers}")

    # --- 3. Create Custom Results Directory ---
    # Structure: results_outlier/{num_outliers}_outliers/k_{k}_m_{m}/
    results_path = os.path.join(
        "results_outlier",
        f"{args.num_outliers}_outliers",
        f"k_{args.k}_m_{args.m}"
    )
    os.makedirs(results_path, exist_ok=True)
    print(f"\nResults will be saved in: '{results_path}'")
    print("-" * 30)

    # --- 4. Generate Datasets ---
    x1_clean, x2_with_outlier, outlier_threshold_L = sf.generate_datasets(params, num_outliers=args.num_outliers)
    print("Datasets generated successfully.")

    # --- 5. Calculate MLE and Generate Plot ---
    mu_star_1 = sf.get_mle(x1_clean, params)
    mu_star_2 = sf.get_mle(x2_with_outlier, params)
    print(f"MLE for clean data, μ*(x1):     {mu_star_1:.4f}")
    print(f"MLE for outlier data, μ*(x2):  {mu_star_2:.4f}")

    plot_path = os.path.join(results_path, "mle_comparison_plot.png")
    an.plot_combined_analysis(
        x1=x1_clean, 
        x2=x2_with_outlier, 
        mu1=mu_star_1, 
        mu2=mu_star_2,
        threshold=outlier_threshold_L,
        save_path=plot_path
    )
    print("-" * 30)

    # --- 6. Generate All Four Posterior Chains for μ ---
    print("--- Generating the four posterior distributions for μ ---")
    mu_chain_x1 = oef.get_full_data_posterior_samples(x1_clean, params)
    mu_chain_x2 = oef.get_full_data_posterior_samples(x2_with_outlier, params)
    mu_chain_mle1 = oef.get_mle_conditional_posterior_samples(mu_star_1, params)
    mu_chain_mle2 = oef.get_mle_conditional_posterior_samples(mu_star_2, params)
    print("All four posterior chains for μ generated successfully.")
    print("-" * 30)

    # --- 7. Generate All Four Predictive Distributions for x ---
    print("--- Generating predictive distributions for x ---")
    scale = 1 # The scale parameter of the t-distribution
    
    # More efficient generation: create datasets once, then derive chains
    datasets_x1 = [stats.t.rvs(df=params['k'], loc=mu, scale=scale, size=params['m']) for mu in mu_chain_x1]
    datasets_x2 = [stats.t.rvs(df=params['k'], loc=mu, scale=scale, size=params['m']) for mu in mu_chain_x2]
    datasets_mle1 = [stats.t.rvs(df=params['k'], loc=mu, scale=scale, size=params['m']) for mu in mu_chain_mle1]
    datasets_mle2 = [stats.t.rvs(df=params['k'], loc=mu, scale=scale, size=params['m']) for mu in mu_chain_mle2]

    # Flattened chains for the main predictive plot
    x_pred_x1 = np.array(datasets_x1).flatten()
    x_pred_x2 = np.array(datasets_x2).flatten()
    x_pred_mle1 = np.array(datasets_mle1).flatten()
    x_pred_mle2 = np.array(datasets_mle2).flatten()
    
    # Chains of the maximum value from each generated dataset
    x_pred_x1_max = np.array([d.max() for d in datasets_x1])
    x_pred_x2_max = np.array([d.max() for d in datasets_x2])
    x_pred_mle1_max = np.array([d.max() for d in datasets_mle1])
    x_pred_mle2_max = np.array([d.max() for d in datasets_mle2])
    
    print("Predictive distributions and max value chains generated successfully.")
    print("-" * 30)

    ## ------------------------------------------------------------------
    ## --- Detailed Analysis and Plotting ---
    ## ------------------------------------------------------------------
    print("Generating detailed analysis plots...")

    # --- Plot 1: Posterior Distributions of μ ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    sns.kdeplot(mu_chain_mle1, label="Posterior $p(μ|μ^*(x_1))$", color='red', lw=2, ax=axes[0])
    sns.kdeplot(mu_chain_x1, label="Posterior $p(μ|x_1)$", color='green', lw=2, ax=axes[0])
    axes[0].axvline(mu_star_1, color='black', linestyle='--', lw=2, label=f"MLE $\\mu^*(x_1) = {mu_star_1:.2f}$")
    axes[0].set_title("Posterior Distributions for μ (from Clean Data $x_1$)", fontsize=15)
    axes[0].set_xlabel("μ")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    sns.kdeplot(mu_chain_mle2, label="Posterior $p(μ|μ^*(x_2))$", color='red', lw=2, ax=axes[1])
    sns.kdeplot(mu_chain_x2, label="Posterior $p(μ|x_2)$", color='green', lw=2, ax=axes[1])
    axes[1].axvline(mu_star_2, color='black', linestyle='--', lw=2, label=f"MLE $\\mu^*(x_2) = {mu_star_2:.2f}$")
    axes[1].set_title("Posterior Distributions for μ (from Outlier Data $x_2$)", fontsize=15)
    axes[1].set_xlabel("μ")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    posterior_plot_path = os.path.join(results_path, "posterior_mu_comparison.png")
    plt.savefig(posterior_plot_path)
    print(f"Posterior μ comparison plot saved to: {posterior_plot_path}")
    plt.close()

    # --- Plot 2: Posterior Predictive Distributions of x ---
    # Calculate tail probabilities P(x > L)
    prob_x_mle1 = np.mean(x_pred_mle1 > outlier_threshold_L)
    prob_x_x1 = np.mean(x_pred_x1 > outlier_threshold_L)
    prob_x_mle2 = np.mean(x_pred_mle2 > outlier_threshold_L)
    prob_x_x2 = np.mean(x_pred_x2 > outlier_threshold_L)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    combined_preds = np.concatenate([x_pred_mle1, x_pred_x1, x_pred_mle2, x_pred_x2])
    x_min = np.percentile(combined_preds, 1)
    x_max = np.percentile(combined_preds, 99)
    padding = (x_max - x_min) * 0.1
    xlim = (x_min - padding, x_max + padding)
    bw = 0.25
    axes[0].set_xlim(xlim)
    sns.kdeplot(x_pred_mle1, label=rf"Predictive $p(x|\mu^*(x_1))$ ($P(x>L)={prob_x_mle1:.4f}$)", color='red', lw=2, fill=True, alpha=0.4, bw_adjust=bw, ax=axes[0])
    sns.kdeplot(x_pred_x1, label=rf"Predictive $p(x|x_1)$ ($P(x>L)={prob_x_x1:.4f}$)", color='green', lw=2, fill=True, alpha=0.4, bw_adjust=bw, ax=axes[0])
    axes[0].axvline(outlier_threshold_L, color='blue', linestyle=':', lw=2, label=f'Outlier Threshold L={outlier_threshold_L:.2f}')
    axes[0].set_title("Posterior Predictive Distributions (from Clean Data $x_1$)", fontsize=15)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].set_xlim(xlim)
    sns.kdeplot(x_pred_mle2, label=rf"Predictive $p(x|\mu^*(x_2))$ ($P(x>L)={prob_x_mle2:.4f}$)", color='red', lw=2, fill=True, alpha=0.4, bw_adjust=bw, ax=axes[1])
    sns.kdeplot(x_pred_x2, label=rf"Predictive $p(x|x_2)$ ($P(x>L)={prob_x_x2:.4f}$)", color='green', lw=2, fill=True, alpha=0.4, bw_adjust=bw, ax=axes[1])
    axes[1].axvline(outlier_threshold_L, color='blue', linestyle=':', lw=2, label=f'Outlier Threshold L={outlier_threshold_L:.2f}')
    axes[1].set_title("Posterior Predictive Distributions (from Outlier Data $x_2$)", fontsize=15)
    axes[1].set_xlabel("x")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    predictive_plot_path = os.path.join(results_path, "posterior_predictive_comparison.png")
    plt.savefig(predictive_plot_path)
    print(f"Posterior predictive comparison plot saved to: {predictive_plot_path}")
    plt.close()

    # --- Plot 3: Posterior Predictive Distributions of max(x) ---
    # Calculate tail probabilities P(max(x) > L)
    prob_max_mle1 = np.mean(x_pred_mle1_max > outlier_threshold_L)
    prob_max_x1 = np.mean(x_pred_x1_max > outlier_threshold_L)
    prob_max_mle2 = np.mean(x_pred_mle2_max > outlier_threshold_L)
    prob_max_x2 = np.mean(x_pred_x2_max > outlier_threshold_L)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    combined_max_preds = np.concatenate([x_pred_mle1_max, x_pred_x1_max, x_pred_mle2_max, x_pred_x2_max])
    x_min_max = np.percentile(combined_max_preds, 1)
    x_max_max = np.percentile(combined_max_preds, 99)
    padding_max = (x_max_max - x_min_max) * 0.1
    xlim_max = (x_min_max - padding_max, x_max_max + padding_max)
    
    axes[0].set_xlim(xlim_max)
    sns.kdeplot(x_pred_mle1_max, label=rf"Predictive $p(\max(x)|\mu^*(x_1))$ ($P(\max>L)={prob_max_mle1:.4f}$)", color='red', lw=2, fill=True, alpha=0.4, bw_adjust=bw, ax=axes[0])
    sns.kdeplot(x_pred_x1_max, label=rf"Predictive $p(\max(x)|x_1)$ ($P(\max>L)={prob_max_x1:.4f}$)", color='green', lw=2, fill=True, alpha=0.4, bw_adjust=bw, ax=axes[0])
    axes[0].axvline(outlier_threshold_L, color='blue', linestyle=':', lw=2, label=f'Outlier Threshold L={outlier_threshold_L:.2f}')
    axes[0].set_title("Posterior Predictive of max(x) (from Clean Data $x_1$)", fontsize=15)
    axes[0].set_xlabel("max(x)")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_xlim(xlim_max)
    sns.kdeplot(x_pred_mle2_max, label=rf"Predictive $p(\max(x)|\mu^*(x_2))$ ($P(\max>L)={prob_max_mle2:.4f}$)", color='red', lw=2, fill=True, alpha=0.4, bw_adjust=bw, ax=axes[1])
    sns.kdeplot(x_pred_x2_max, label=rf"Predictive $p(\max(x)|x_2)$ ($P(\max>L)={prob_max_x2:.4f}$)", color='green', lw=2, fill=True, alpha=0.4, bw_adjust=bw, ax=axes[1])
    axes[1].axvline(outlier_threshold_L, color='blue', linestyle=':', lw=2, label=f'Outlier Threshold L={outlier_threshold_L:.2f}')
    axes[1].set_title("Posterior Predictive of max(x) (from Outlier Data $x_2$)", fontsize=15)
    axes[1].set_xlabel("max(x)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    predictive_max_plot_path = os.path.join(results_path, "posterior_predictive_max_comparison.png")
    plt.savefig(predictive_max_plot_path)
    print(f"Posterior predictive max(x) comparison plot saved to: {predictive_max_plot_path}")
    plt.close()
    
    print("\nScript finished successfully for num_outliers =", args.num_outliers, "and k =", args.k, "with m =", args.m)


if __name__ == "__main__":
    # This block handles command-line argument parsing
    parser = argparse.ArgumentParser(description="Run an outlier sensitivity analysis experiment.")
    
    parser.add_argument('-k', type=float, default=2.0, 
                        help='Degrees of freedom for the t-distribution.')
    parser.add_argument('-m', type=int, default=20, 
                        help='Number of samples in each dataset.')
    parser.add_argument('--num_outliers', type=int, default=1, 
                        help='Number of outliers to plant in the second dataset.')
    
    args = parser.parse_args()
    
    # Run the main function with the parsed arguments
    main(args)



