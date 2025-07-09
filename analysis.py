import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from IPython.display import display
import sampler_functions as sf

# ==============================================================================
# --- Main Sampler Diagnostics
# ==============================================================================

def plot_diagnostics_with_benchmark(results, params, kde_0, filename=None):
    """
    Creates a comprehensive diagnostic plot for a single Gibbs run, including
    a validation comparison against a benchmark KDE posterior.
    This version includes the histogram of the Gibbs samples.

    Args:
        results (dict): The dictionary returned by the main sampler function.
        params (dict): The parameters dictionary for context.
        kde_0 (scipy.stats.gaussian_kde): The pre-computed KDE object for p(μ̂|μ=0).
        filename (str, optional): Path to save the plot. If None, displays the plot.
    """
    print("\n--- Generating Consolidated Diagnostic and Validation Plot ---")
    
    # --- Extract data from results dictionary ---
    mu_chain = results['mu_chain']
    T = len(mu_chain)
    burn_in = int(T * 0.2)
    posterior_mu_samples = mu_chain[burn_in:]

    # --- Create a multi-panel figure ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Sampler Diagnostics and Validation", fontsize=20)

    # --- Plot 1: Mu Trace Plot (Unchanged) ---
    axes[0].plot(mu_chain, color='royalblue', lw=0.8)
    axes[0].axvline(burn_in, color='red', linestyle='--', label=f'Burn-in ({burn_in} steps)')
    axes[0].set_title("Trace Plot of μ")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Value of μ")
    axes[0].legend()
    axes[0].grid(True, alpha=0.5)

    # --- Plot 2: Sampler Posterior vs. Benchmark Posterior ---
    
    # Create a grid for the smooth density curves
    mu_grid = np.linspace(posterior_mu_samples.min() - 1, posterior_mu_samples.max() + 1, 500)
    
    # --- MODIFICATION STARTS HERE ---
    
    # NEW: Plot the histogram of the Gibbs sampler output in the background
    axes[1].hist(posterior_mu_samples, bins=50, density=True, color='red', alpha=0.3, label="Sampler Histogram")
    
    # Plot the KDE of our Gibbs sampler's output over the histogram
    kde_sampler = stats.gaussian_kde(posterior_mu_samples)
    axes[1].plot(mu_grid, kde_sampler(mu_grid), color='red', lw=2.5, label="p(μ | μ*) - Sampler KDE")
    
    # --- MODIFICATION ENDS HERE ---
    
    # Now, calculate and plot the benchmark posterior on the same axes
    likelihood_kde = kde_0.pdf(params['mu_star'] - mu_grid)
    prior_on_grid = stats.norm.pdf(mu_grid, loc=params['prior_mean'], scale=params['prior_std'])
    unnormalized_posterior_kde = likelihood_kde * prior_on_grid
    # area = np.trapezoid(unnormalized_posterior_kde, mu_grid)
    area = np.trapz(unnormalized_posterior_kde, mu_grid)
    benchmark_posterior = unnormalized_posterior_kde / area
    axes[1].plot(mu_grid, benchmark_posterior, color='green', linestyle='--', lw=2.5, label="Benchmark Posterior")

    # Add reference lines as requested
    axes[1].axvline(params['mu_true'], color='black', linestyle='-.', lw=2, label=f"True μ")
    axes[1].axvline(params['mu_star'], color='darkorange', linestyle=':', lw=2.5, label=f"Observed MLE (μ*)")
    
    axes[1].set_title("Posterior Comparison: Sampler vs. Benchmark")
    axes[1].set_xlabel("Value of μ")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Diagnostic plot saved to {filename}")
        plt.close(fig)
    else:
        plt.show()


# ==============================================================================
# --- Experiment 1: Mu Posterior Comparison
# ==============================================================================

def plot_posterior_comparison(mle_chain, x_original, params, filename=None):
    """
    Compares the sampler's posterior p(μ|μ*) against the true analytical
    posterior from the full data, p(μ|x).

    Args:
        mle_chain (np.array): The MCMC chain for μ from the Gibbs sampler.
        x_original (np.array): The original, full dataset.
        params (dict): The parameters dictionary.
        filename (str, optional): Path to save the plot.
    """
    print("\n--- Generating Final Posterior Comparison Plot (Sampler vs. True Analytical) ---")
    
    # --- 1. Get the samples from the Gibbs sampler (for p(μ|μ*)) ---
    burn_in = int(len(mle_chain) * 0.2)
    posterior_mle_samples = mle_chain[burn_in:]

    # --- 2. Calculate the True Analytical Posterior p(μ|x) ---
    mu_grid = np.linspace(posterior_mle_samples.min() - 1, posterior_mle_samples.max() + 1, 500)
    
    # Calculate the unnormalized posterior using our log_posterior_mu function
    log_posterior_vals = [sf.log_posterior_mu(mu, x_original, params['k'], params['prior_mean'], params['prior_std']) for mu in mu_grid]
    unnormalized_posterior_full = np.exp(log_posterior_vals - np.max(log_posterior_vals)) # Subtract max for stability


    # Normalize using numerical integration
    # integral_area = np.trapezoid(unnormalized_posterior_full, mu_grid)
    integral_area = np.trapz(unnormalized_posterior_full, mu_grid)
    true_posterior_full_data = unnormalized_posterior_full / integral_area

    # --- 3. Quantitative Comparison ---
    mean_mle = np.mean(posterior_mle_samples)
    std_mle = np.std(posterior_mle_samples)
    # mean_full = np.trapezoid(mu_grid * true_posterior_full_data, mu_grid)
    # std_full = np.sqrt(np.trapezoid((mu_grid - mean_full)**2 * true_posterior_full_data, mu_grid))
    mean_full = np.trapz(mu_grid * true_posterior_full_data, mu_grid)
    std_full = np.sqrt(np.trapz((mu_grid - mean_full)**2 * true_posterior_full_data, mu_grid))


    print(f"Posterior from MLE only (Sampler): Mean = {mean_mle:.4f}, Std Dev = {std_mle:.4f}")
    print(f"Posterior from Full Data (True):   Mean = {mean_full:.4f}, Std Dev = {std_full:.4f}")

    # --- 4. Visual Comparison ---
    plt.figure(figsize=(12, 8))
    
    # Plot the KDE of our Gibbs sampler's output
    kde_mle = stats.gaussian_kde(posterior_mle_samples)
    plt.plot(mu_grid, kde_mle(mu_grid), color='red', lw=2.5, label=f"p(μ | μ*) - Sampler Output [Std Dev = {std_mle:.3f}]")
    plt.fill_between(mu_grid, kde_mle(mu_grid), color='red', alpha=0.2)

    # Plot the true analytical posterior curve
    plt.plot(mu_grid, true_posterior_full_data, color='blue', lw=2.5, label=f"p(μ | x) - True Posterior [Std Dev = {std_full:.3f}]")
    plt.fill_between(mu_grid, true_posterior_full_data, color='blue', alpha=0.2)

    # Reference lines
    plt.axvline(params['mu_true'], color='black', linestyle='--', linewidth=2, label=f"True μ")
    plt.axvline(params['mu_star'], color='darkorange', linestyle=':', linewidth=2.5, label=f"Observed MLE (μ*)")
    plt.title("Comparison of Posteriors: Full Data vs. MLE Only", fontsize=16)
    plt.xlabel("Value of μ")
    plt.ylabel("Posterior Density")
    plt.legend(fontsize=12)
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")
        plt.close()
    else:
        plt.show()

# ==============================================================================
# --- Experiment 2: Posterior Predictive Check for X
# ==============================================================================

def create_x_comparison_table_markdown(x_original, x_pred_full_data, x_pred_mle, filename=None):
    """
    Creates and saves/displays a Markdown table comparing data statistics,
    including tail-focused percentiles.
    """
    print("\n--- Generating Statistics Comparison Table (Markdown) ---")
    
    stats_to_compare = {
        "Mean": [np.mean(x_original), np.mean(x_pred_full_data), np.mean(x_pred_mle)],
        "Std. Dev.": [np.std(x_original), np.std(x_pred_full_data), np.std(x_pred_mle)],
        "0.5th Percentile": [np.percentile(x_original, 0.5), np.percentile(x_pred_full_data, 0.5), np.percentile(x_pred_mle, 0.5)],
        "5th Percentile": [np.percentile(x_original, 5), np.percentile(x_pred_full_data, 5), np.percentile(x_pred_mle, 5)],
        "25th Percentile (Q1)": [np.percentile(x_original, 25), np.percentile(x_pred_full_data, 25), np.percentile(x_pred_mle, 25)],
        "Median (50th)": [np.percentile(x_original, 50), np.percentile(x_pred_full_data, 50), np.percentile(x_pred_mle, 50)],
        "75th Percentile (Q3)": [np.percentile(x_original, 75), np.percentile(x_pred_full_data, 75), np.percentile(x_pred_mle, 75)],
        "95th Percentile": [np.percentile(x_original, 95), np.percentile(x_pred_full_data, 95), np.percentile(x_pred_mle, 95)],
        "99.5th Percentile": [np.percentile(x_original, 99.5), np.percentile(x_pred_full_data, 99.5), np.percentile(x_pred_mle, 99.5)],
    }
    index_labels = ["Ground Truth (x_original)", "Pred. from Full Data (x̃|x)", "Pred. from MLE (x̃|μ*)"]
    df = pd.DataFrame(stats_to_compare, index=index_labels)

    # Build the Markdown table
    header = "| Statistic | " + " | ".join(df.columns) + " |"
    separator = "|" + "---|" * (len(df.columns) + 1)
    rows = []
    for idx, row in df.iterrows():
        formatted = [f"{v:.4f}" for v in row]
        rows.append(f"| {idx} | " + " | ".join(formatted) + " |")
    markdown = "\n".join([header, separator] + rows)

    # Optionally save to file
    if filename:
        with open(filename, "w") as f:
            f.write("# Final Comparison of Data Statistics (with Tail Analysis)\n\n")
            f.write(markdown)
        print(f"Markdown table saved to {filename}")
    else:
        print("\n# Final Comparison of Data Statistics (with Tail Analysis)\n")
        print(markdown)
    
    return markdown

def plot_x_distribution_comparison(x_pred_full_data, x_pred_mle, filename=None, bw_adjust=1.0):
    """
    Plots the KDEs of the two posterior predictive distributions.
    This version omits the noisy KDE of the original data for clarity.

    Args:
        x_pred_full_data (np.array): Posterior predictive samples given full data.
        x_pred_mle (np.array): Posterior predictive samples given MLE.
        filename (str, optional): If provided, saves the plot to this file.
        bw_adjust (float, optional): Bandwidth adjustment for KDE smoothing.
    """
    print("\n--- Generating Final Data Distribution Comparison Plot ---")
    plt.figure(figsize=(12, 8))

    # Input validation and conversion
    x_pred_full_data = np.asarray(x_pred_full_data).flatten()
    x_pred_mle = np.asarray(x_pred_mle).flatten()

    # Remove NaNs or infs if present
    x_pred_full_data = x_pred_full_data[np.isfinite(x_pred_full_data)]
    x_pred_mle = x_pred_mle[np.isfinite(x_pred_mle)]

    # Set x-axis limits based on percentiles to avoid outlier distortion
    combined_data = np.concatenate([x_pred_full_data, x_pred_mle])
    x_min, x_max = np.percentile(combined_data, [1, 99])
    plt.xlim(x_min, x_max)

    # Plot KDEs with optional bandwidth adjustment
    sns.kdeplot(
        x_pred_full_data, color='green', lw=3, 
        label='Posterior Predictive given Full Data (x̃|x)', 
        fill=True, alpha=0.2, bw_adjust=bw_adjust
    )
    sns.kdeplot(
        x_pred_mle, color='red', lw=3, linestyle='--', 
        label='Posterior Predictive given MLE (x̃|μ*)', 
        fill=True, alpha=0.2, bw_adjust=bw_adjust
    )

    plt.title("Comparison of Posterior Predictive Distributions", fontsize=16)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.4)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")
        plt.close()
    else:
        plt.show()

def plot_predictive_density_difference(x_pred_full_data, x_pred_mle, filename=None):
    """
    Plots the difference between the two posterior predictive density curves
    to visualize the subtle increase in variance.

    Difference = (Density from MLE) - (Density from Full Data)

    Args:
        x_pred_full_data (np.array): Flattened samples from p(x̃|x).
        x_pred_mle (np.array): Flattened samples from p(x̃|μ*).
        filename (str, optional): Path to save the plot.
    """
    print("\n--- Generating Predictive Density Difference Plot ---")

    # 1. Create a Kernel Density Estimate (KDE) for each distribution
    kde_full = stats.gaussian_kde(x_pred_full_data)
    kde_mle = stats.gaussian_kde(x_pred_mle)

    # 2. Create a grid to evaluate the densities on
    #    We use a wide range to see the behavior in the tails
    combined_data = np.concatenate([x_pred_full_data, x_pred_mle])
    grid_min = np.percentile(combined_data, 0.5)
    grid_max = np.percentile(combined_data, 99.5)
    x_grid = np.linspace(grid_min, grid_max, 500)

    # 3. Calculate the difference between the two KDEs on the grid
    difference = kde_mle(x_grid) - kde_full(x_grid)

    # 4. Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(x_grid, difference, color='purple', lw=2)
    
    # Add a horizontal line at y=0 for reference
    plt.axhline(0, color='black', linestyle=':')
    
    # Fill the areas to make the difference clearer
    plt.fill_between(x_grid, difference, 0, where=difference > 0,
                     color='red', alpha=0.3, interpolate=True, label='p(x̃|μ*) is higher')
    plt.fill_between(x_grid, difference, 0, where=difference < 0,
                     color='green', alpha=0.3, interpolate=True, label='p(x̃|x) is higher')

    plt.title("Difference Between Posterior Predictive Densities", fontsize=16)
    plt.xlabel("Value")
    plt.ylabel("Density Difference\n(Pred. from MLE - Pred. from Full Data)")
    plt.legend()
    plt.grid(True, alpha=0.5)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Difference plot saved to {filename}")
        plt.close()
    else:
        plt.show()

def plot_outlier_predictive_distributions(x_pred_x1, x_pred_x2, x_pred_mle1, x_pred_mle2, L, filename_prefix=None):
    """
    Plots the four posterior predictive densities in two separate plots to compare their tail behavior.
    
    Args:
        x_pred_x1 (np.array): Samples from p(x̃|x1) - Bayesian predictive from clean data.
        x_pred_x2 (np.array): Samples from p(x̃|x2) - Bayesian predictive from outlier data.
        x_pred_mle1 (np.array): Samples from p(x̃|μ*(x1)) - MLE-based predictive from clean data.
        x_pred_mle2 (np.array): Samples from p(x̃|μ*(x2)) - MLE-based predictive from outlier data.
        L (float): The outlier threshold value.
        filename_prefix (str, optional): Prefix for saving the plots. If provided, two files
                                         will be saved: '{prefix}_bayesian.png' and '{prefix}_mle.png'.
    """
    print("\n--- Generating Posterior Predictive Density Comparison Plots ---")
    
    # Determine a suitable x-axis range based on all data and the outlier threshold
    # We want to show the bulk of the distribution and the relevant tail around L
    all_x_data = np.concatenate([x_pred_x1, x_pred_x2, x_pred_mle1, x_pred_mle2])
    
    # Calculate quantiles to get a robust range for the x-axis
    # Use a small quantile (e.g., 0.001) for the left limit to capture the left tail
    # and extend slightly beyond L for the right limit.
    x_min_val = np.percentile(all_x_data, 0.1) # Capture more of the left tail
    x_max_val = max(np.percentile(all_x_data, 99.9), L * 1.2) # Ensure L and some beyond are visible

    # Adjust x_min_val if it's too far negative compared to L
    # This helps to zoom in if the data is very spread out but we care about the positive tail
    if L > 0: # If L is positive, we might want to focus more on positive values
        x_min_val = min(x_min_val, -L * 2) # Ensure some negative range, but not excessively so
        x_min_val = max(x_min_val, np.percentile(all_x_data, 0.5)) # Don't cut off too much of the distribution

    # --- Plot 1: Bayesian Predictive Densities ---
    plt.figure(figsize=(10, 6)) # Adjusted figure size for single comparison
    
    sns.kdeplot(x_pred_x1, color='green', lw=2.5, linestyle='-', label='Bayesian Pred. (Clean Data)')
    sns.kdeplot(x_pred_x2, color='green', lw=2.5, linestyle='--', label='Bayesian Pred. (Outlier Data)')
    
    plt.axvline(L, color='black', linestyle=':', lw=2, label=f'Outlier Threshold (L={L:.2f})')
    
    plt.yscale('log')
    plt.xlim(x_min_val, x_max_val) # Set dynamic x-axis limits
    plt.ylim(bottom=1e-5) # Set a lower limit for the log scale

    plt.title("Bayesian Posterior Predictive Tails", fontsize=16)
    plt.xlabel("Value")
    plt.ylabel("Density (Log Scale)")
    plt.legend(fontsize=11)
    plt.grid(True, which="both", linestyle='--', alpha=0.5)

    if filename_prefix:
        bayesian_filename = f"{filename_prefix}_bayesian.png"
        plt.savefig(bayesian_filename, dpi=300, bbox_inches='tight')
        print(f"Bayesian plot saved to {bayesian_filename}")
        plt.close()
    else:
        plt.show()

    # --- Plot 2: MLE-based Predictive Densities ---
    plt.figure(figsize=(10, 6)) # Adjusted figure size
    
    sns.kdeplot(x_pred_mle1, color='red', lw=2.5, linestyle='-', label='MLE-based Pred. (Clean Data)')
    sns.kdeplot(x_pred_mle2, color='red', lw=2.5, linestyle='--', label='MLE-based Pred. (Outlier Data)')
    
    plt.axvline(L, color='black', linestyle=':', lw=2, label=f'Outlier Threshold (L={L:.2f})')
    
    plt.yscale('log')
    plt.xlim(x_min_val, x_max_val) # Set dynamic x-axis limits
    plt.ylim(bottom=1e-5) # Set a lower limit for the log scale

    plt.title("MLE-based Posterior Predictive Tails", fontsize=16)
    plt.xlabel("Value")
    plt.ylabel("Density (Log Scale)")
    plt.legend(fontsize=11)
    plt.grid(True, which="both", linestyle='--', alpha=0.5)

    if filename_prefix:
        mle_filename = f"{filename_prefix}_mle.png"
        plt.savefig(mle_filename, dpi=300, bbox_inches='tight')
        print(f"MLE-based plot saved to {mle_filename}")
        plt.close()
    else:
        plt.show()
