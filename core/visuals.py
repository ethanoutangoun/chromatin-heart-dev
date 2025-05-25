import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors

def plot_clique_size_optimization(sizes, p_values):
    """
    Plot the empirical p-values against clique sizes to visualize the optimization results.
    
    Parameters:
        sizes (list): List of clique sizes.
        p_values (list): Corresponding list of empirical p-values for each size.
    """
    if len(sizes) != len(p_values):
        raise ValueError("Sizes and p_values must have the same length.")
    

    plt.style.use('seaborn-v0_8-whitegrid')  # Clean modern look
    # 1) Plot p-value vs size
    plt.figure()
    plt.plot(sizes, p_values, marker='o', linestyle='-')
    plt.axhline(0.05, color='red', linestyle='--', label='α=0.05')
    plt.xlabel('Clique size')
    plt.ylabel('Empirical p-value')
    plt.title('P-value vs Clique Size')
    plt.xticks(sizes)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2) Choose the size with minimum p-value
    opt_idx = int(np.argmin(p_values))
    opt_size = sizes[opt_idx]
    opt_pval = p_values[opt_idx]
    print(f'Optimal clique size = {opt_size}, p-value = {opt_pval:.4f}')




def plot_pval_heatmap_from_log(csv_path: str, alpha_precision: int = 3):
    df = pd.read_csv(csv_path)
    df["alpha_rounded"] = df["alpha"].round(alpha_precision)

    heatmap_data = df.pivot_table(
        index="alpha_rounded",
        columns="k",
        values="pval",
        aggfunc="min"
    )
    heatmap_data.sort_index(ascending=True, inplace=True)

    # Find min (alpha, k) pair
    min_val = df["pval"].min()
    best = df[df["pval"] == min_val].iloc[0]
    best_alpha = round(best["alpha"], alpha_precision)
    best_k = int(best["k"])

    plt.figure(figsize=(16, 6))
    ax = sns.heatmap(
        heatmap_data,
        cmap="viridis_r",
        norm=mcolors.LogNorm(vmin=heatmap_data.min().min(), vmax=heatmap_data.max().max()),
        annot=True,
        fmt=".1e",
        linewidths=0.4,
        cbar_kws={"label": "p-value"},
        annot_kws={"fontsize": 6}
    )
    plt.title("P-value Heatmap over (α, k) Grid", fontsize=14)
    plt.xlabel("Clique size (k)", fontsize=12)
    plt.ylabel("Restart probability (α)", fontsize=12)

    # Highlight best cell
    ax.scatter(
        x=[heatmap_data.columns.get_loc(best_k) + 0.5],
        y=[heatmap_data.index.get_loc(best_alpha) + 0.5],
        s=100, color='red', edgecolors='black', linewidth=1.5
    )
    plt.tight_layout()
    plt.show()


def plot_pval_heatmap_from_trials(trial_df, alpha_precision=2):
    """
    Plot a heatmap of p-values over (alpha, k) combinations from an Optuna trial dataframe.
    """
    # Round alphas for better heatmap grouping
    trial_df['alpha_rounded'] = trial_df['alpha'].round(alpha_precision)

    # Pivot the table to have alpha as rows and k as columns
    heatmap_data = trial_df.pivot_table(
        index='alpha_rounded',
        columns='k',
        values='pval',
        aggfunc='min'  # or 'mean' if you want average over duplicates
    )

    # Sort alpha axis from low to high
    heatmap_data.sort_index(ascending=True, inplace=True)

    # Plot
    plt.figure(figsize=(14, 6))
    sns.heatmap(
        heatmap_data,
        cmap="viridis_r",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "p-value"}
    )
    plt.title("P-value Heatmap by α and k")
    plt.xlabel("Clique size (k)")
    plt.ylabel("Restart probability (α)")
    plt.tight_layout()
    plt.show()



