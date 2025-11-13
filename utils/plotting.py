import math

import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from scipy.stats import linregress


def fit_power_law(x, y):
    """
    Fits a power-law relationship of the form y ≈ a * x^b using log-log regression.

    Parameters:
        x (array-like): Independent variable (noise variance).
        y (array-like): Dependent variable (distance metric).

    Returns:
        a (float): Scale factor in y = a * x^b
        b (float): Exponent in y = a * x^b
    """
    log_x = np.log(x)
    log_y = np.log(y)
    slope, intercept, _, _, _ = linregress(log_x, log_y)
    return np.exp(intercept), slope  # Convert back from log-space


def fit_power_law_fixed_exponent(x, y, exponent=0.5):
    """
    Fits a power-law relationship y ≈ a * x^b for a fixed exponent b.

    The model is y = a * x^exponent. Taking logs, we get log(y) = log(a) + exponent * log(x).
    This means log(a) = log(y) - exponent * log(x). We find the best 'a' by
    taking the mean of this expression over all data points.

    Parameters:
        x (array-like): Independent variable.
        y (array-like): Dependent variable.
        exponent (float): The fixed exponent 'b' for the power law.

    Returns:
        a (float): The fitted scale factor.
        b (float): The fixed exponent.
    """
    log_x = np.log(x)
    log_y = np.log(y)
    # Calculate log(a) as the mean of (log(y) - exponent * log(x))
    log_a = np.mean(log_y - exponent * log_x)
    a = np.exp(log_a)
    return a, exponent


def fit_all_distances(df) -> dict:
    """
    Fit power-law relationships for all *_distance_avg columns present.
    - For Wk_distance_avg, also fit a fixed exponent = 1/k.
    Returns: {metric: (a, b), f"{metric}_fixed": (a, b_fixed) for Wk}.
    """
    x = df["noise_std"]
    # Fit only columns that actually exist
    metrics_to_fit = [m for m in [
        "W1_distance_avg", "W2_distance_avg", "L2_distance_avg",
    ] if m in df.columns]

    fits = {}
    for metric in metrics_to_fit:
        y = df[metric]
        fits[metric] = fit_power_law(x, y)

        # If it's a Wasserstein metric, extract p and fit with fixed exponent = 1/p
        p = re.match(r"^W(\d+)_", metric)
        if p:
            k = int(p.group(1))          # works for W1, W2, ..., W100, ...
            exp = 1.0 / k
            fits[f"{metric}_fixed"] = fit_power_law_fixed_exponent(x, y, exponent=exp)

    return fits


def plot_fitted_curves(results_df, fits, prefix, save):
    """
    Plots the original distance metrics along with their power-law fits.

    Parameters:
        results_df (DataFrame): Data containing noise variance and distances.
        fits (dict): Fitted parameters {metric: (a, b)}
        save (bool): Whether to save the plots.
        prefix (str): Prefix for the saved plot filenames.
    """
    plt.figure(figsize=(10, 4))

    colors = {"W1_distance_avg": "#b1182b", "W2_distance_avg": "#2065ab", "L2_distance_avg": "sandybrown",}
    markers = {"W1_distance_avg": "o", "W2_distance_avg": "s", "L2_distance_avg": "^",}

    metrics_to_plot = ["W1_distance_avg", "W2_distance_avg", "L2_distance_avg",]

    for metric in metrics_to_plot:
        if metric not in fits:
            print(f"No fitted parameters for {metric}.")
            continue

        x_original = results_df["noise_std"]
        res = results_df['resolution'].iloc[0]
        y_original = results_df[metric]

        # Plot original data
        plt.plot(x_original, y_original, markersize=4, color=colors[metric], marker=markers[metric], linestyle='None')
        if metric.startswith("W") and metric != "W1_distance_avg":
            # print(metric)
            a, b = fits[metric+'_fixed']
            # Plot the theoretical bound as a dashed line
            p = int(metric[1:2])
            if p == 1:
                p = 100
            const1 = (4 * res) / math.sqrt(math.pi)
            plt.plot(x_original, ((const1 * x_original) ** (1/p)),
                     color=colors[metric], linestyle='--', linewidth=2)
        elif metric.startswith("W") and metric == "W1_distance_avg":
            a, b = fits[metric+'_fixed']
            const = 2 * res * math.log2(res) + (res / (2 * math.sqrt(math.pi)))
            plt.plot(x_original, const * x_original,
                     color=colors[metric], linestyle='--', linewidth=2)

        else:
            a, b = fits[metric]
        if metric != "W2_distance_avg":
            label_fit = fr"{metric.replace('_distance_avg', '')} = $ {a:.3f}\;\sigma$"
        else:
            label_fit = f"{metric.replace('_distance_avg', '')} = $ {a:.3f}\;\\sqrt{{\\sigma}}$"
        plt.plot(x_original, a * x_original ** b, label=label_fit, color=colors[metric], linewidth=2)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("$\\sigma$", fontsize=12)
    plt.ylabel("$d\\,(\\mu+\\epsilon, \\mu)$", fontsize=12)
    plt.legend(fontsize=12, loc="best")
    plt.xlim(min(results_df["noise_std"]), max(results_df["noise_std"]))
    plt.grid()
    plt.tight_layout()

    if save:
        plt.savefig(f'{prefix}distances_vs_noise_avg_subset.pdf', format='pdf', dpi=1200)
        plt.savefig(f'{prefix}distances_vs_noise_avg_subset.png', format='png', dpi=1200)

    plt.show()


def plot_distance_ratios(df, file_prefix, save_fig=True):
    """
    Plots distance ratios and returns the Figure and Axes objects.
    The function signature is preserved.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    plot_configs = [
        {'col': 'W1_ratio', 'label': r'$W_1 : \frac{W_1(\mu_\sigma, \nu_\sigma)}{W_1(\mu, \nu)}$', 'color': '#b1182b'},
        {'col': 'W2_ratio', 'label': r'$W_2 : \frac{W_2(\mu_\sigma, \nu_\sigma)}{W_2(\mu, \nu)}$', 'color': '#2065ab'},
        {'col': 'L2_ratio', 'label': r'$L_2 : \frac{\|\mu_\sigma - \nu_\sigma\|_2}{\|\mu - \nu\|_2}$', 'color': 'sandybrown'},
    ]

    for config in plot_configs:
        ax.plot(df['noise_std'], df[config['col']], label=config['label'], color=config['color'])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\sigma$', fontsize=12)
    ax.set_ylabel(r'Distance Ratio', fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True)

    if save_fig:
        file_name = f'{file_prefix}_distance_ratios'
        plt.savefig(f'{file_name}.pdf', format='pdf', dpi=1200, bbox_inches='tight')

    plt.show()


def plot_distance_differences(df, file_prefix, save_fig=True):
    """
    Plots the differences between noisy and original distances for W1, W2, L2, and diffusion distances.
    The function signature is preserved.
    """
    plt.figure(figsize=(10, 4))
    resolution = df['resolution'].iloc[0]
    d = math.sqrt(2) / 2  # Diameter constant
    const1 = 4 * resolution * np.log2(resolution) / math.sqrt(math.pi)
    const2 = resolution * 2 / math.sqrt(math.pi)
    const_total = const1 + const2

    plt.plot(df['noise_std'], df['noisy_vs_noisy_W1'],
             label='$W_1(\\mu_\\sigma, \\nu_\\sigma)$',
             color='#b1182b')
    plt.plot(df['noise_std'],  df['original_W1'] + (math.sqrt(1/2) * const_total * df['noise_std']),
             color='#b1182b', linestyle='--')
    # W2
    plt.plot(df['noise_std'], df['noisy_vs_noisy_W2'], label='$W_2(\\mu_\\sigma, \\nu_\\sigma)$',
             color='#2065ab')
    plt.plot(df['noise_std'], (d ** (1-0.5)) * (df['original_W1'] ** 0.5) + d * ((const_total * df['noise_std']) ** 0.5),
             color='#2065ab', linestyle='--')
    # W3
    plt.plot(df['noise_std'], df['noisy_vs_noisy_W3'], label='$W_3(\\mu_\\sigma, \\nu_\\sigma)$',
             color='sandybrown')
    plt.plot(df['noise_std'], (d ** (1-1/3)) * (df['original_W1'] ** (1/3)) + d * ((const_total * df['noise_std']) ** (1/3)),
             color='sandybrown', linestyle='--')

    plt.xlabel('$\\sigma$')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel("$d\\,(\\mu+\\epsilon_\\mu, \\nu+\\epsilon_\\nu)$", fontsize=12)
    plt.legend()
    plt.xlim(min(df['noise_std']), max(df['noise_std']))
    plt.grid(True)
    if save_fig:
        file_name = f'{file_prefix}_distance_differences_W1_W2_W3_W4'
        plt.savefig(f'{file_name}.pdf', format='pdf', dpi=1200, bbox_inches='tight')
    plt.show()


def visualize_single_image_results(results_df, file_prefix, save_fig=True):
    """Visualize the results of single image experiments."""
    # Fitted curves
    fits = fit_all_distances(results_df)

    plot_fitted_curves(results_df, fits, save=save_fig, prefix=file_prefix)

    return fits


def plot_multiple_distance_ratios(
        results_dfs: list[pd.DataFrame],
        subplot_titles: list[str],
        file_prefix: str,
        save_fig: bool = True
):
    """
    Plots distance ratios from multiple DataFrames in adjacent subplots
    with a shared x-axis and a single, common x-label.
    """

    # 1. Create subplots with shared x and y axes
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(results_dfs),
        figsize=(16, 4),
        sharex=True,
        sharey=True
    )

    for ax, df, title in zip(axes, results_dfs, subplot_titles):

        ax.plot(df['noise_std'], df['W1_ratio'],
                label=r'$W_1$', color='#b1182b', alpha=0.7)
        ax.plot(df['noise_std'], df['W2_ratio'],
                label=r'$W_2$', color='#2065ab', alpha=0.7)
        ax.plot(df['noise_std'], df['L2_ratio'],
                label=r'$L_2$', color='sandybrown', alpha=0.7)
        ax.set_xlabel(r'$\sigma$', fontsize=16)
        ax.set_xlim(df['noise_std'].min(), df['noise_std'].max())

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(title, fontsize=16)
        ax.grid()

    # 4. Add shared elements and finalize the layout
    axes[0].set_ylabel('Distance Ratio', fontsize=14)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, -0.1), ncol=len(labels), fontsize=16)

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    if save_fig:
        file_name = f'{file_prefix}multiple_types_ratios_comparison.pdf'
        fig.savefig(file_name, format='pdf', dpi=1200, bbox_inches='tight')

    plt.show()
