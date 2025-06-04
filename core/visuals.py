import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
import networkx as nx



def clique_to_graph(contact_matrix, clique, selected_bin=None):
    
    edges = []
    for i in range(len(clique)):
        for j in range(i+1, len(clique)):
            edges.append((clique[i], clique[j], contact_matrix[clique[i], clique[j]]))

    G = nx.Graph()
    G.add_nodes_from(clique)
    G.add_weighted_edges_from(edges)

    base_color = '#7fbfdf'    
    highlight_color = "#bd5931" 
    node_colors = [highlight_color if bin == selected_bin else base_color for bin in clique]

    # pos = nx.spring_layout(G, seed=42)
    pos = nx.circular_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=800, node_color=node_colors,
            font_size=10, font_weight='bold', edge_color='dimgray', linewidths=1.5, font_color="#232323")

    labels = nx.get_edge_attributes(G, 'weight')
    labels = {k: f"{v:.5f}" for k, v in labels.items()}  # Filter out very small weights
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=10)

    plt.tight_layout()
    plt.show()
    return G




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



# def plot_pval_contours_from_csv(csv_path: str, alpha_precision: int = 3):
#     """
#     Load a CSV containing alpha, k, and pval columns and produce a smooth contour plot
#     showing log10(p-value) over the (alpha, k) parameter grid.
#     """
#     df = pd.read_csv(csv_path)
#     df["alpha_rounded"] = df["alpha"].round(alpha_precision)

#     # Pivot for plotting
#     pval_mat = df.pivot_table(
#         index="alpha_rounded",
#         columns="k",
#         values="pval",
#         aggfunc="min"
#     ).sort_index()

#     # Prepare grid
#     alpha_vals = pval_mat.index.values
#     size_vals = pval_mat.columns.values
#     AA, SS = np.meshgrid(alpha_vals, size_vals, indexing='ij')
#     Z = np.log10(pval_mat.values)

#     # Find the most significant point
#     min_idx = np.unravel_index(np.nanargmin(Z), Z.shape)
#     optimal_alpha = alpha_vals[min_idx[0]]
#     optimal_k = size_vals[min_idx[1]]
#     min_log_pval = Z[min_idx]
#     actual_pval = 10 ** min_log_pval

#     print(f"Most significant combination:")
#     print(f"α = {optimal_alpha:.3f}, k = {optimal_k}, p-value = {actual_pval:.2e}")
#     print(f"log₁₀(p-value) = {min_log_pval:.3f}")

#     # Plot
#     plt.figure(figsize=(10, 8))
#     levels = np.linspace(np.nanmin(Z), np.nanmax(Z), 25)

#     cp = plt.contourf(SS, AA, Z, levels=levels, cmap='viridis_r')
#     plt.contour(SS, AA, Z, levels=10, colors='black', alpha=0.4, linewidths=0.5)

#     plt.colorbar(cp, label='log₁₀(p-value)', shrink=0.8)

#     plt.plot(optimal_k, optimal_alpha, 'o', color='red', markersize=8,
#              markeredgecolor='black', markeredgewidth=1.5,
#              label=f'Min: α={optimal_alpha:.3f}, k={optimal_k}')

#     plt.xlabel('Clique size (k)', fontsize=12, fontweight='bold')
#     plt.ylabel('Restart probability (α)', fontsize=12, fontweight='bold')
#     plt.title('log₁₀(p-value) Parameter Optimization',
#               fontsize=14, fontweight='bold', pad=20)

#     plt.grid(True, alpha=0.3, linestyle='--')
#     plt.legend(loc='upper right', fontsize=11, framealpha=0.9)

#     plt.xlim(size_vals.min() - 0.5, size_vals.max() + 0.5)
#     plt.ylim(alpha_vals.min() - 0.01, alpha_vals.max() + 0.01)

#     textstr = f'Min p-value: {actual_pval:.2e}\nα={optimal_alpha:.3f}, k={optimal_k}'
#     props = dict(boxstyle='round', facecolor='white', alpha=0.8)
#     plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
#              verticalalignment='top', bbox=props)

#     plt.tight_layout()
#     plt.show()

def plot_pval_contours_from_csv(csv_path: str, alpha_precision: int = 3, render_range: tuple = None):
    """
    Plot a contour map of log10(p-values) from a CSV with alpha, k, and pval columns.

    Parameters:
    - csv_path: Path to CSV file.
    - alpha_precision: Rounding precision for alpha values.
    - render_range: Optional tuple (k_min, k_max) to limit x-axis range.
    """
    df = pd.read_csv(csv_path)
    df["alpha_rounded"] = df["alpha"].round(alpha_precision)

    # Pivot to get matrix of p-values
    pval_mat = df.pivot_table(
        index="alpha_rounded",
        columns="k",
        values="pval",
        aggfunc="min"
    ).sort_index()

    # Apply optional render range filter
    if render_range is not None:
        k_min, k_max = render_range
        pval_mat = pval_mat.loc[:, (pval_mat.columns >= k_min) & (pval_mat.columns <= k_max)]

    # Build grid
    alpha_vals = pval_mat.index.values
    size_vals = pval_mat.columns.values
    AA, SS = np.meshgrid(alpha_vals, size_vals, indexing='ij')
    Z = np.log10(pval_mat.values)

    # Find min point
    min_idx = np.unravel_index(np.nanargmin(Z), Z.shape)
    optimal_alpha = alpha_vals[min_idx[0]]
    optimal_k = size_vals[min_idx[1]]
    min_log_pval = Z[min_idx]
    actual_pval = 10 ** min_log_pval

    print(f"Most significant combination:")
    print(f"α = {optimal_alpha:.3f}, k = {optimal_k}, p-value = {actual_pval:.2e}")
    print(f"log₁₀(p-value) = {min_log_pval:.3f}")

    # Plot
    plt.figure(figsize=(10, 8))
    levels = np.linspace(np.nanmin(Z), np.nanmax(Z), 25)
    cp = plt.contourf(SS, AA, Z, levels=levels, cmap='viridis_r')
    plt.contour(SS, AA, Z, levels=10, colors='black', alpha=0.4, linewidths=0.5)

    plt.colorbar(cp, label='log₁₀(p-value)', shrink=0.8)

    plt.plot(optimal_k, optimal_alpha, 'o', color='red', markersize=8,
             markeredgecolor='black', markeredgewidth=1.5,
             label=f'Min: α={optimal_alpha:.3f}, k={optimal_k}')

    plt.xlabel('Clique size (k)', fontsize=12, fontweight='bold')
    plt.ylabel('Restart probability (α)', fontsize=12, fontweight='bold')
    plt.title('log₁₀(p-value) Parameter Optimization',
              fontsize=14, fontweight='bold', pad=20)

    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='upper right', fontsize=11, framealpha=0.9)

    plt.xlim(size_vals.min() - 0.5, size_vals.max() + 0.5)
    plt.ylim(alpha_vals.min() - 0.01, alpha_vals.max() + 0.01)

    textstr = f'Min p-value: {actual_pval:.2e}\nα={optimal_alpha:.3f}, k={optimal_k}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()


def plot_pval_heatmap_from_log(csv_path: str, alpha_precision: int = 3, alpha_range=None, k_range=None):
    df = pd.read_csv(csv_path)

    # Optional filtering
    if alpha_range is not None:
        df = df[(df["alpha"] >= alpha_range[0]) & (df["alpha"] <= alpha_range[1])]
    if k_range is not None:
        df = df[(df["k"] >= k_range[0]) & (df["k"] <= k_range[1])]

    df["alpha_rounded"] = df["alpha"].round(alpha_precision)

    heatmap_data = df.pivot_table(
        index="alpha_rounded",
        columns="k",
        values="pval",
        aggfunc="min"
    )
    heatmap_data.sort_index(ascending=True, inplace=True)

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

    plt.tight_layout()
    plt.show()
    


def plot_pval_heatmaps_by_bg(
    csv_path: str,
    alpha_precision: int = 3,
    alpha_range=None,
    k_range=None,
    bg_to_plot=None
):
    """
    Reads a CSV with columns ["alpha", "k", "bg_label", "pval", "fold_change"],
    optionally filters by alpha_range and k_range, and then produces one heatmap
    per distinct bg_label—showing the minimum p-value over the (alpha, k) grid
    for that background model. If bg_to_plot is provided, only that background
    will be displayed; otherwise, all backgrounds are shown.

    Parameters
    ----------
    csv_path : str
        Path to the CSV log file.
    alpha_precision : int
        Number of decimal places to round alpha before pivoting (determines row bins).
    alpha_range : tuple or None
        If provided, a (min_alpha, max_alpha) tuple to filter rows by alpha.
    k_range : tuple or None
        If provided, a (min_k, max_k) tuple to filter rows by k.
    bg_to_plot : str or None
        If provided, only this bg_label will be plotted. Must exactly match one of
        the bg_label values in the CSV. If None, all bg_labels are plotted.
    """

    # Load the full log
    df = pd.read_csv(csv_path)

    # Optional filtering by alpha and k
    if alpha_range is not None:
        df = df[(df["alpha"] >= alpha_range[0]) & (df["alpha"] <= alpha_range[1])]
    if k_range is not None:
        df = df[(df["k"] >= k_range[0]) & (df["k"] <= k_range[1])]

    # Round alpha for grouping
    df["alpha_rounded"] = df["alpha"].round(alpha_precision)

    # Identify unique background labels
    all_bg_labels = sorted(df["bg_label"].unique())
    if not all_bg_labels:
        raise ValueError("No background labels found in the CSV.")

    # Determine which bg_labels to plot
    if bg_to_plot is None:
        bg_labels = all_bg_labels
    else:
        if bg_to_plot not in all_bg_labels:
            raise ValueError(f"bg_to_plot='{bg_to_plot}' not found in CSV. "
                             f"Available labels: {all_bg_labels}")
        bg_labels = [bg_to_plot]

    # Build a pivot table for each selected bg_label
    pivot_tables = {}
    for label in bg_labels:
        subset = df[df["bg_label"] == label]
        pivot = subset.pivot_table(
            index="alpha_rounded",
            columns="k",
            values="pval",
            aggfunc="min"
        )
        pivot.sort_index(ascending=True, inplace=True)
        pivot_tables[label] = pivot

    # Determine the global min/max p-value across all pivot tables for a shared color scale
    all_min = min(pivot.min().min() for pivot in pivot_tables.values())
    all_max = max(pivot.max().max() for pivot in pivot_tables.values())

    # Create a subplot grid: one row, len(bg_labels) columns
    n_labels = len(bg_labels)
    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_labels,
        figsize=(6 * n_labels, 6),
        sharey=True
    )

    # If there's only one label, axes is a single Axes; wrap it in a list
    if n_labels == 1:
        axes = [axes]

    for ax, label in zip(axes, bg_labels):
        pivot = pivot_tables[label]
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="viridis_r",
            norm=mcolors.LogNorm(vmin=all_min, vmax=all_max),
            annot=True,
            fmt=".1e",
            linewidths=0.4,
            cbar=(ax is axes[-1]),  # show colorbar on the last subplot only
            cbar_kws={"label": "p-value"},
            annot_kws={"fontsize": 6}
        )
        ax.set_title(f"Background = {label}", fontsize=14)
        ax.set_xlabel("Clique size (k)", fontsize=12)
        if ax is axes[0]:
            ax.set_ylabel("Restart probability (α)", fontsize=12)
        else:
            ax.set_ylabel("")

    plt.tight_layout()
    plt.show()








def plot_fold_change_heatmap_from_log(csv_path: str, alpha_precision: int = 3):
    df = pd.read_csv(csv_path)
    df["alpha_rounded"] = df["alpha"].round(alpha_precision)

    # Ensure fold_enrichment column exists
    if "fold_change" not in df.columns:
        raise ValueError("CSV must contain a 'fold_change' column.")

    heatmap_data = df.pivot_table(
        index="alpha_rounded",
        columns="k",
        values="fold_change",
        aggfunc="max"  # max enrichment is typically more interesting
    )
    heatmap_data.sort_index(ascending=True, inplace=True)

    # Find max (alpha, k) pair
    max_val = df["fold_change"].max()
    best = df[df["fold_change"] == max_val].iloc[0]
    best_alpha = round(best["alpha"], alpha_precision)
    best_k = int(best["k"])

    plt.figure(figsize=(16, 6))
    ax = sns.heatmap(
        heatmap_data,
        cmap="YlOrRd",
        norm=mcolors.LogNorm(vmin=max(1.0, heatmap_data.min().min()), vmax=heatmap_data.max().max()),
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        cbar_kws={"label": "Fold Change"},
        annot_kws={"fontsize": 6}
    )
    plt.title("Fold Change Heatmap over (α, k) Grid", fontsize=14)
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



import math

def plot_contact_matrix(
    matrix: np.ndarray,
    title: str,
    max_bins: int = 5000,
    vmin: float = 0,
    vmax: float = 99,
    genome_size_bp: int = 3_100_000_000,  
    n_ticks: int = 5                  
):

    # --- (1) optionally aggregate large matrices ---
    n_bins = matrix.shape[0]
    if n_bins > max_bins:
        factor = math.ceil(n_bins / max_bins)
        new_n = math.ceil(n_bins / factor)
        pad_amt = new_n * factor - n_bins
        if pad_amt > 0:
            matrix = np.pad(
                matrix,
                ((0, pad_amt), (0, pad_amt)),
                mode='constant',
                constant_values=np.nan
            )
        matrix = matrix.reshape(new_n, factor, new_n, factor).mean(axis=(1,3))
        n_bins = new_n

    # --- (2) color scale ceiling at the given percentile ---
    vmax_val = np.nanpercentile(matrix, vmax)

    # --- (3) compute resolution (bp per bin) and tick positions/labels ---
    bp_per_bin = genome_size_bp / n_bins
    # tick positions in bin indices
    tick_bins = np.linspace(0, n_bins, num=n_ticks, endpoint=True)
    # convert to Mb
    tick_mbs = (tick_bins * bp_per_bin) / 1e6
    # format labels as integers or one decimal
    tick_labels = [f"{mb:.1f}" if mb < 10 else f"{int(mb)}" for mb in tick_mbs]

    # --- (4) set global font modern ---
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': 'Arial',
        'font.size': 10
    })

    # --- (5) plot matrix ---
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.cm.Reds
    cmap.set_bad(color='white')
    im = ax.imshow(
        matrix,
        cmap=cmap,
        interpolation='nearest',
        vmin=vmin,
        vmax=vmax_val,
        aspect='equal'
    )

    # --- (6) title & axis labels ---
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Genomic position (Mb)", labelpad=8)
    ax.set_ylabel("Genomic position (Mb)", labelpad=8)

    # --- (7) set ticks ---
    ax.set_xticks(tick_bins)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_bins)
    ax.set_yticklabels(tick_labels)

    # --- (8) improved colorbar ---
    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.04)    
    cbar.set_label("Interaction frequency", fontsize=10, labelpad=6)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.show()


def show_matrix_only(matrix: np.ndarray, vmax: float = 99):
    vmax_val = np.nanpercentile(matrix, vmax)
    cmap = plt.cm.Reds
    cmap.set_bad(color='white')

    plt.figure(figsize=(6, 6))
    plt.imshow(matrix, cmap=cmap, interpolation='nearest', vmin=0, vmax=vmax_val)
    plt.axis('off')  # removes axes
    plt.tight_layout()
    plt.show()