import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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



def plot_pval_heatmap(csv_path):
    # Load and preprocess data
    df = pd.read_csv(csv_path)
    df.set_index('alpha', inplace=True)

    # Compute log10 of p-values (clip to avoid log(0))
    log_df = np.log10(df.clip(lower=1e-300))

    # Find minimum p-value and its coordinates
    min_val = df.min().min()
    min_alpha, min_size = None, None
    for alpha in df.index:
        for size in df.columns:
            if df.loc[alpha, size] == min_val:
                min_alpha = alpha
                min_size = int(size)
                break

    # Plot configuration
    plt.figure(figsize=(10, 6))
    im = plt.imshow(log_df.values, cmap="viridis_r", aspect='auto')

    # Axis ticks and labels
    plt.xticks(ticks=np.arange(len(df.columns)), labels=df.columns, fontsize=10)
    plt.yticks(ticks=np.arange(len(df.index)), labels=df.index, fontsize=10)
    plt.xlabel("Clique Size ($k$)", fontsize=12)
    plt.ylabel("Restart Probability ($\\alpha$)", fontsize=12)
    plt.title("Log$_{10}$ P-values Across $\mathbf{\\alpha}$ and Clique Size", fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("log$_{10}$(p-value)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Highlight minimum
    plt.scatter(df.columns.get_loc(str(min_size)), df.index.get_loc(min_alpha),
                color='red', edgecolor='black', zorder=10,
                label=f"Minimum p-value:\nα = {min_alpha}, k = {min_size}\np = {min_val:.1e}")
    plt.legend(loc="upper right", fontsize=10)

    # Optional: Add grid for visual guidance
    plt.grid(False)
    plt.tight_layout()
    plt.show()
