
import numpy as np
import optuna
import core.stats
import core.clique_finding as cf
from tqdm import tqdm
import pandas as pd
import os

# Optimize diffusion parameters (alpha and k) for clique finding in a contact matrix using Optuna.
def optimize_diffusion_params(contact_matrix, seed_bin, k_range=(5, 50), alpha_bounds=(0.01, 0.95), n_trials=50, timeout_minutes=120):
    N = contact_matrix.shape[0]

    # Build row-stochastic transition matrix P once
    P = np.zeros((N, N), dtype=float)
    row_sums = contact_matrix.sum(axis=1)
    for i in range(N):
        if row_sums[i] > 0:
            P[i, :] = contact_matrix[i, :] / row_sums[i]
        else:
            P[i, i] = 1.0

    I = np.eye(N)

    def objective(trial):
        alpha = trial.suggest_float("alpha", alpha_bounds[0], alpha_bounds[1])
        k = trial.suggest_int("k", k_range[0], k_range[1])

        # Compute full diffusion matrix F
        A = I - (1 - alpha) * P
        F = np.linalg.inv(A)

        # TTN seed clique
        ttn_ranks = np.argsort(F[seed_bin])[::-1]
        clique = ttn_ranks[:k]
        ttn_score = core.stats.calculate_avg_interaction_strength(contact_matrix, clique)

        # Background scores for all other seeds
        bg_scores = []
        for i in range(N):
            if i == seed_bin:
                continue
            ranks = np.argsort(F[i])[::-1][:k]
            score = core.stats.calculate_avg_interaction_strength(contact_matrix, ranks)
            bg_scores.append(score)

        # Compute p-value
        bg_scores = np.array(bg_scores)
        pval = core.stats.empirical_p_value(ttn_score, bg_scores)
        fold_change = ttn_score / (np.median(bg_scores) + 1e-10)

        # Save fold change in trial's user attributes
        trial.set_user_attr("fold_change", fold_change)
        return pval

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout_minutes * 60)

    best_alpha = study.best_params['alpha']
    best_k = study.best_params['k']

    A = I - (1 - best_alpha) * P
    F = np.linalg.inv(A)
    best_clique = np.argsort(F[seed_bin])[::-1][:best_k]

    # Compute final p-value
    best_score = core.stats.calculate_avg_interaction_strength(contact_matrix, best_clique)
    final_bg_scores = []
    for i in range(N):
        if i == seed_bin:
            continue
        ranks = np.argsort(F[i])[::-1][:best_k]
        score = core.stats.calculate_avg_interaction_strength(contact_matrix, ranks)
        final_bg_scores.append(score)
    final_pval = (np.sum(np.array(final_bg_scores) >= best_score) + 1) / (len(final_bg_scores) + 1)


    # Retrieve top 5 trials by lowest p-value
    all_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else np.inf)
    trial_history = [
        {"alpha": t.params["alpha"], "k": t.params["k"], "pval": t.value, "fold_change": t.user_attrs.get("fold_change", None)}
        for t in all_trials
    ]

    # save as DataFrame
    trial_df = pd.DataFrame(trial_history)
    trial_df.to_csv("diffusion_trial_history.csv", index=False)
    return best_alpha, best_k, best_clique, final_pval, trial_history


def optimize_diffusion_params_stochastic(
    contact_matrix: np.ndarray,
    seed_bin: int,
    k_range=(5, 50),
    alpha_bounds=(0.01, 0.95),
    n_trials: int = 50,
    neighbors=None,
    cdfs=None,
    background_size = 1000,
    num_walkers=1000,
    timeout_minutes: int = 120,
    log_csv: str = "diffusion_alpha_k_log.csv",
    background_bins=None,
):
    N = contact_matrix.shape[0]

    if neighbors is None or cdfs is None:
        return ValueError("Must provide precomputed neighbors and CDFs for stochastic diffusion.")

    
    # If no background list is provided, default to all bins ≠ seed_bin
    if background_bins is None:
        bg_list = [i for i in range(N) if i != seed_bin]
    else:
        # ensure seed_bin isn't accidentally included
        bg_list = [i for i in background_bins if i != seed_bin]

    all_trials_log = []
    study = optuna.create_study(direction="minimize")

    # Ensure alpha bounds are evaluated
    alpha_lower, alpha_upper = alpha_bounds
    study.enqueue_trial({"alpha": alpha_lower})
    study.enqueue_trial({"alpha": alpha_upper})

    def objective(trial):
        alpha = trial.suggest_float("alpha", *alpha_bounds)


        bg_dist = []
        for _ in tqdm(range(background_size), desc=f"Generating background (α = {alpha:.3f})"):
            random_bin = np.random.choice(bg_list)
            # Generate a random walk for the background bin
            bg_dist.append(cf.random_walk(contact_matrix, random_bin, k_range[1],neighbors=neighbors, cdfs=cdfs, num_molecules=num_walkers, alpha=alpha))


        ttn_clique_all = cf.random_walk(
            contact_matrix,
            seed_bin,
            k_range[1],
            neighbors=neighbors,
            cdfs=cdfs,
            num_molecules=num_walkers,
            alpha=alpha
        )

        best_p = np.inf
        for k in tqdm(range(k_range[0], k_range[1] + 1), desc=f"α = {alpha:.3f}"):
            clique = ttn_clique_all[:k]
            ttn_score = core.stats.calculate_avg_interaction_strength(contact_matrix, clique)

            # Only use bg_list here
            bg_scores = [
                core.stats.calculate_avg_interaction_strength(contact_matrix, clique[:k])
                for clique in bg_dist
            ]


            bg_scores = np.array(bg_scores)

            pval = core.stats.empirical_p_value(ttn_score, bg_scores)
            fold = ttn_score / (np.median(bg_scores) + 1e-10)

            all_trials_log.append({
                "alpha": alpha,
                "k": k,
                "pval": pval,
                "fold_change": fold
            })
            best_p = min(best_p, pval)

        return best_p

    study.optimize(objective, n_trials=n_trials, timeout=timeout_minutes * 60)

    # Extract best params
    best_alpha = study.best_params["alpha"]



    # Find the best (alpha, k) record for final clique
    best_entry = min(
        (r for r in all_trials_log if np.isclose(r["alpha"], best_alpha, atol=1e-6)),
        key=lambda r: r["pval"]
    )

    best_k = best_entry["k"]
    best_clique = cf.random_walk(
        contact_matrix,
        seed_bin,
        best_k,
        neighbors=neighbors,
        cdfs=cdfs,
        num_molecules=num_walkers,
        alpha=best_alpha
    )
    final_pval = best_entry["pval"]
    final_fold = best_entry["fold_change"]

    # Save full grid log
    pd.DataFrame(all_trials_log).to_csv(log_csv, index=False)

    return {
        "best_alpha": best_alpha,
        "best_k": best_k,
        "best_clique": best_clique,
        "final_pval": final_pval,
        "final_fold": final_fold,
        "full_log": all_trials_log
    }
    


def optimize_diffusion_params_smart(
    contact_matrix: np.ndarray,
    seed_bin: int,
    k_range=(5, 50),
    alpha_bounds=(0.01, 0.95),
    n_trials: int = 50,
    timeout_minutes: int = 120,
    log_csv: str = "full_alpha_k_log.csv",
    background_bins=None,
):
    """
    If background_bins is provided, only those bins (excluding seed_bin) will be
    used to compute the empirical p-value. Otherwise all other bins are used.

    This version appends the results of each trial to `log_csv` immediately after
    that trial finishes, so that a crash midway won't lose already‐completed trials.
    """
    N = contact_matrix.shape[0]

    # Build row-stochastic transition matrix P and identity I
    P = np.zeros((N, N), dtype=float)
    row_sums = contact_matrix.sum(axis=1)
    for i in range(N):
        if row_sums[i] > 0:
            P[i, :] = contact_matrix[i, :] / row_sums[i]
        else:
            P[i, i] = 1.0
    I = np.eye(N)

    # Determine background list
    if background_bins is None:
        bg_list = [i for i in range(N) if i != seed_bin]
    else:
        bg_list = [i for i in background_bins if i != seed_bin]

    all_trials_log = []
    study = optuna.create_study(direction="minimize")

    # Ensure alpha bounds are evaluated first
    alpha_lower, alpha_upper = alpha_bounds
    study.enqueue_trial({"alpha": alpha_lower})
    study.enqueue_trial({"alpha": alpha_upper})

    # If the CSV does not exist, write the header row first
    if not os.path.isfile(log_csv):
        header_df = pd.DataFrame(columns=["alpha", "k", "pval", "fold_change"])
        header_df.to_csv(log_csv, index=False)

    def objective(trial):
        alpha = trial.suggest_float("alpha", *alpha_bounds)

        # Invert diffusion matrix for this alpha
        A = I - (1 - alpha) * P
        F = np.linalg.inv(A)
        sorted_ranks = [np.argsort(F[i])[::-1] for i in range(N)]

        best_p = np.inf
        # Collect rows just for this trial:
        trial_log = []

        for k in tqdm(range(k_range[0], k_range[1] + 1), desc=f"α = {alpha:.3f}"):
            clique = sorted_ranks[seed_bin][:k]
            ttn_score = core.stats.calculate_avg_interaction_strength(contact_matrix, clique)

            # Compute background scores
            bg_scores = [
                core.stats.calculate_avg_interaction_strength(contact_matrix, sorted_ranks[i][:k])
                for i in bg_list
            ]
            bg_scores = np.array(bg_scores)

            pval = core.stats.empirical_p_value(ttn_score, bg_scores)
            fold = ttn_score / (np.median(bg_scores) + 1e-10)

            row = {
                "alpha": alpha,
                "k": k,
                "pval": pval,
                "fold_change": fold
            }
            trial_log.append(row)
            all_trials_log.append(row)

            best_p = min(best_p, pval)

        # After finishing all k values for this trial, append to CSV:
        df_trial = pd.DataFrame(trial_log)
        df_trial.to_csv(log_csv, mode="a", header=False, index=False)

        return best_p

    # Run optimization (Optuna will call objective() repeatedly)
    study.optimize(objective, n_trials=n_trials, timeout=timeout_minutes * 60)

    # Once optimization is done, extract the best parameters & recompute final clique
    best_alpha = study.best_params["alpha"]
    A = I - (1 - best_alpha) * P
    F = np.linalg.inv(A)
    sorted_ranks = [np.argsort(F[i])[::-1] for i in range(N)]

    # Find the best (alpha, k) record among all_trials_log for final clique
    best_entry = min(
        (r for r in all_trials_log if np.isclose(r["alpha"], best_alpha, atol=1e-6)),
        key=lambda r: r["pval"]
    )

    best_k = best_entry["k"]
    best_clique = sorted_ranks[seed_bin][:best_k]
    final_pval = best_entry["pval"]
    final_fold = best_entry["fold_change"]

    # As a final sanity check, overwrite the CSV with the full history 
    # (optional—your individual trial‐by‐trial appends have already been saved)
    pd.DataFrame(all_trials_log).to_csv(log_csv, index=False)

    return {
        "best_alpha": best_alpha,
        "best_k": best_k,
        "best_clique": best_clique,
        "final_pval": final_pval,
        "final_fold": final_fold,
        "full_log": all_trials_log
    }



# Given a fixed alpha, # optimize the clique size k for a given seed bin
def optimize_clique_size_diffusion(
    contact_matrix: np.ndarray,
    max_clique_size: int,
    seed_bin: int,
    alpha: float,
):
    """
    For each possible seed bin i:
      – use F[i, :] to pick the top‐k visited bins as its clique
      – score those cliques against the TTN seed_bin clique
    """

    N = contact_matrix.shape[0]
    sizes = list(range(1, max_clique_size + 1))

    print("🔄 Computing full analytical diffusion matrix F…")
    F = cf.all_analytical_diffusions(contact_matrix=contact_matrix, alpha=alpha)
    # F.shape == (N, N)

    # 1) Build the TTN clique once
    ttn_full = np.argsort(F[seed_bin])[-max_clique_size:][::-1]
    print(f"✔︎ TTN seed clique (size {max_clique_size}) ready")

    # 2) Build background cliques for every possible seed i
    bg_full = [
        np.argsort(F[i])[-max_clique_size:][::-1]
        for i in range(N)
    ]
    print(f"✔︎ Built background cliques for all {N} seeds")

    # 3) Now for each size, compute scores & empirical p‐values
    ttn_scores, p_values, fold_changes = [], [], []
    bg_dists = {}

    for size in tqdm(sizes, desc="Processing sizes"):
        # TTN subclique & score
        ttn_sub = ttn_full[:size]
        ttn_score = core.stats.calculate_avg_interaction_strength(contact_matrix, ttn_sub)

        # Background scores for this same size
        bg_scores = [
            core.stats.calculate_avg_interaction_strength(contact_matrix, clique[:size])
            for clique in bg_full
        ]
        bg_dists[size] = bg_scores

        median_bg = np.median(bg_scores)
        # empirical p‐value, +1 correction
        pval = core.stats.empirical_p_value(ttn_score, bg_scores)
        # pval = (np.sum(np.array(bg_scores) >= ttn_score) + 1) / (N + 1)
        fold  = ttn_score / median_bg if median_bg != 0 else float("nan")

        ttn_scores.append(ttn_score)
        p_values.append(pval)
        fold_changes.append(fold)

    print("✅ Done optimize_clique_size_analytical")
    return sizes, ttn_scores, p_values, fold_changes, bg_dists


# general optimization function for greedy and rw
def optimize_clique_size(
    contact_matrix,
    max_clique_size,
    seed_bin,
    num_samples=1000,
    clique_alg=cf.find_greedy_clique,
    background_bins = None,
    label=None,
    **alg_kwargs
):
    """
    Runs a single full-size clique search with `clique_alg`, then trims down to all sizes.

    Parameters:
    - contact_matrix: Hi-C contact matrix
    - max_clique_size: maximum clique size to search
    - seed_bin: start bin for your TTN clique
    - num_samples: number of random background samples
    - clique_alg: function(contact_matrix, size, seed_bin, **alg_kwargs)
    - alg_kwargs: extra keyword arguments for `clique_alg` (e.g. num_neighbors)

    Returns:
    sizes, ttn_scores, p_values, fold_changes, bg_dists
    """
    print(f"Starting optimize_clique_size: max_clique_size={max_clique_size}, "
          f"seed_bin={seed_bin}, num_samples={num_samples}, alg={clique_alg.__name__}")

    # 1) Full-size TTN clique
    ttn_full = clique_alg(
        contact_matrix,
        max_clique_size,
        seed_bin,
        **alg_kwargs
    )
    print(f"Computed TTN full clique of size {len(ttn_full)} using {clique_alg.__name__}")

    bg_set = []
    if background_bins is None:
        bg_set = [i for i in range(contact_matrix.shape[0]) if i != seed_bin]
    else:
        # ensure seed_bin isn't accidentally included
        bg_set = [i for i in background_bins if i != seed_bin]


    # 2) Background samples (full size)
    bg_full = []
    for _ in tqdm(range(num_samples), desc="Sampling background cliques"):
        rand_bin = np.random.choice(bg_set)
        bg = clique_alg(
            contact_matrix,
            max_clique_size,
            rand_bin,
            **alg_kwargs
        )
        bg_full.append(bg)
    print("Background sampling complete.")

    sizes = list(range(1, max_clique_size + 1))
    ttn_scores, p_values, fold_changes = [], [], []
    bg_dists = {}

    # 3) Trim & score for each size
    for size in tqdm(sizes, desc="Processing sizes"):

        # TTN subclique
        ttn_sub = ttn_full[:size]
        ttn_score = core.stats.calculate_avg_interaction_strength(
            contact_matrix,
            ttn_sub
        )
 

        # Background scores
        bg_scores = []
        for clique in bg_full:
            sub = clique[:size]
            score = core.stats.calculate_avg_interaction_strength(
                contact_matrix,
                sub
            )
            bg_scores.append(score)
        bg_dists[size] = bg_scores

        # Stats
        median_bg = np.median(bg_scores)
        pval = (np.sum(np.array(bg_scores) >= ttn_score) + 1) / (num_samples + 1)
        fold = ttn_score / median_bg if median_bg != 0 else float('nan')


        ttn_scores.append(ttn_score)
        p_values.append(pval)
        fold_changes.append(fold)

    # save as df to csv
    results_df = pd.DataFrame({
        "size": sizes,
        "ttn_score": ttn_scores,
        "p_value": p_values,
        "fold_change": fold_changes
    })


    if label is None:
        results_df.to_csv(f"greedy_clique_size_optimization_results_seed_{seed_bin}.csv", index=False)
    else:
        results_df.to_csv(f"greedy_clique_size_optimization_results_seed_{seed_bin}_{label}.csv", index=False)



    print("Completed optimize_clique_size")
    return sizes, ttn_scores, p_values, fold_changes, bg_dists




def grid_search_diffusion_params(
    contact_matrix: np.ndarray,
    seed_bin: int,
    alphas: list,
    max_clique_size: int,
    save_suffix = None,
):
    

    # --- Parameters ---
    alphas          = [0.01, 0.05, 0.1, 0.25, 0.5]
    max_clique_size = 50

    print("🔄 Starting full sweep over alphas and sizes...")

    # --- Collect all size-wise results ---
    records = []
    for alpha in alphas:
        print(f"→ Running optimize_clique_size for α={alpha}")
        sizes, scores, pvals, folds, _ = optimize_clique_size_diffusion(
            contact_matrix=contact_matrix,
            max_clique_size=max_clique_size,
            seed_bin=seed_bin,
            alpha=alpha,
        )
        print(f"   • Completed α={alpha} (collected sizes 1–{max_clique_size})")
        for size, pval, fold in zip(sizes, pvals, folds):
            records.append({
                'alpha': alpha,
                'size':  size,
                'pval':  pval,
                'fold':  fold
            })

    df = pd.DataFrame(records)
    print(f"✅ DataFrame assembled: {df.shape[0]} rows")

    # --- Pivot into matrices ---
    pval_mat = df.pivot(index='alpha', columns='size', values='pval')
    fold_mat = df.pivot(index='alpha', columns='size', values='fold')

    print("📊 Plotting heatmaps...")
    # Enhanced contour plot of log10 p-value over the full grid
    import numpy as np
    import matplotlib.pyplot as plt

    # Get data
    alpha_vals = pval_mat.index.values
    size_vals = pval_mat.columns.values
    AA, SS = np.meshgrid(alpha_vals, size_vals, indexing='ij')
    Z = np.log10(pval_mat.values)

    # Find the most significant point (minimum log10 p-value)
    min_idx = np.unravel_index(np.nanargmin(Z), Z.shape)
    optimal_alpha = alpha_vals[min_idx[0]]
    optimal_size = size_vals[min_idx[1]]
    min_log_pval = Z[min_idx]
    actual_pval = 10**min_log_pval

    print(f"Most significant combination:")
    print(f"α = {optimal_alpha:.3f}, k = {optimal_size}, p-value = {actual_pval:.2e}")
    print(f"log₁₀(p-value) = {min_log_pval:.3f}")

    # Create enhanced figure
    plt.figure(figsize=(10, 8))

    # Create contour plot with more levels for smoother appearance
    levels = np.linspace(np.nanmin(Z), np.nanmax(Z), 25)
    cp = plt.contourf(SS, AA, Z, levels=levels, cmap='viridis_r')

    # Add contour lines for better readability
    contour_lines = plt.contour(SS, AA, Z, levels=10, colors='black', alpha=0.4, linewidths=0.5)

    # Enhanced colorbar
    cbar = plt.colorbar(cp, label='log₁₀(p-value)', shrink=0.8)
    cbar.ax.tick_params(labelsize=11)

    # Mark the optimal point with professional styling
    plt.plot(optimal_size, optimal_alpha, 'o', color='red', markersize=8, 
            markeredgecolor='black', markeredgewidth=1.5, 
            label=f'Minimum: α={optimal_alpha:.3f}, k={optimal_size}')

    # Enhanced labels and title
    plt.xlabel('Clique size (k)', fontsize=12, fontweight='bold')
    plt.ylabel('Restart probability (α)', fontsize=12, fontweight='bold')
    plt.title('log₁₀(p-value) Parameter Optimization', 
            fontsize=14, fontweight='bold', pad=20)

    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--')

    # Legend
    plt.legend(loc='upper right', fontsize=11, framealpha=0.9)

    # Set axis limits with some padding
    plt.xlim(size_vals.min() - 0.5, size_vals.max() + 0.5)
    plt.ylim(alpha_vals.min() - 0.01, alpha_vals.max() + 0.01)

    # Add text box with statistics
    textstr = f'Min p-value: {actual_pval:.2e}\nAt α={optimal_alpha:.3f}, k={optimal_size}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()




    print("🎉 All done!")

    if save_suffix is None:
        suffix = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    else:
        suffix = save_suffix
        
    

    # Save results to CSV
    df.to_csv(f"sweep_results_{suffix}.csv", index=False)

    # Save pivoted matrices
    pval_mat.to_csv(f"pval_matrix_{suffix}.csv")
    fold_mat.to_csv(f"fold_matrix_{suffix}.csv")

    print("💾 CSV files saved")











# DEPRECATED: Use optimize_diffusion_params instead
# This function uses a different approach to optimize alpha and k
# but is kept for reference. It uses a p-value minimization strategy.

# import numpy as np
# from scipy.sparse.linalg import bicgstab
# import optuna


# def compute_clique_score(contact_matrix, clique):
#     submatrix = contact_matrix[np.ix_(clique, clique)]
#     upper_triangle = submatrix[np.triu_indices_from(submatrix, k=1)]
#     return np.mean(upper_triangle)


# def calculate_p_value(observed, null_distribution):
#     null_distribution = np.array(null_distribution)
#     return np.mean(null_distribution >= observed)


# def find_best_alpha_k_with_pval(contact_matrix, seed_bin, background_bins, k_range=(5, 50), alpha_bounds=(0.01, 0.95), n_trials=50):
#     N = contact_matrix.shape[0]

#     # Row-stochastic transition matrix P
#     P = np.zeros((N, N), dtype=float)
#     row_sums = contact_matrix.sum(axis=1)
#     for i in range(N):
#         if row_sums[i] > 0:
#             P[i, :] = contact_matrix[i, :] / row_sums[i]
#         else:
#             P[i, i] = 1.0

#     I = np.eye(N)

#     def objective(trial):
#         alpha = trial.suggest_float("alpha", alpha_bounds[0], alpha_bounds[1])
#         k = trial.suggest_int("k", k_range[0], k_range[1])

#         A = I - (1 - alpha) * P
#         b = np.zeros(N)
#         b[seed_bin] = 1.0
#         visits, _ = bicgstab(A, b)
#         ranked_nodes = np.argsort(visits)[::-1]
#         clique = ranked_nodes[:k]
#         observed_score = core.stats.calculate_avg_interaction_strength(contact_matrix, clique)

#         null_scores = []
#         for bg_bin in background_bins:
#             b_bg = np.zeros(N)
#             b_bg[bg_bin] = 1.0
#             visits_bg, _ = bicgstab(A, b_bg)
#             ranked_bg = np.argsort(visits_bg)[::-1]
#             bg_clique = ranked_bg[:k]
#             null_score = core.stats.calculate_avg_interaction_strength(contact_matrix, bg_clique)
#             null_scores.append(null_score)

#         p_val = calculate_p_value(observed_score, null_scores)
#         return p_val  # Minimize p-value

#     study = optuna.create_study(direction="minimize")
#     study.optimize(objective, n_trials=n_trials)

#     best_alpha = study.best_params['alpha']
#     best_k = study.best_params['k']

#     # Compute best clique and score
#     A = I - (1 - best_alpha) * P
#     b = np.zeros(N)
#     b[seed_bin] = 1.0
#     visits, _ = bicgstab(A, b)
#     ranked_nodes = np.argsort(visits)[::-1]
#     best_clique = ranked_nodes[:best_k]

#     # Final p-value
#     observed_score = core.stats.calculate_avg_interaction_strength(contact_matrix, best_clique)
#     null_scores = []
#     for bg_bin in background_bins:
#         b_bg = np.zeros(N)
#         b_bg[bg_bin] = 1.0
#         visits_bg, _ = bicgstab(A, b_bg)
#         ranked_bg = np.argsort(visits_bg)[::-1]
#         bg_clique = ranked_bg[:best_k]
#         null_score = core.stats.calculate_avg_interaction_strength(contact_matrix, bg_clique)
#         null_scores.append(null_score)
#     best_pval = calculate_p_value(observed_score, null_scores)

#     return best_alpha, best_k, best_clique, best_pval