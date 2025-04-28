# background_model.py
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from .clique_finding import find_greedy_clique, random_walk
from .stats import calculate_avg_interaction_strength


def create_greedy(
    contact_matrix,
    clique_size,
    num_iterations=10000,
    bins=None,
    label=None,
    display=True,
    write=True,
):
    
    if label is None:
        label = 'all'
    
    if bins is None:
        bins = np.arange(contact_matrix.shape[0])

    output_dir = os.path.join(os.getcwd(), 'background_models', 'greedy')
    os.makedirs(output_dir, exist_ok=True)

    scores = []
    for _ in tqdm(range(num_iterations), desc="Sampling cliques", unit="iter"):
        seed_bin = np.random.choice(bins)
        clique = find_greedy_clique(
            contact_matrix,
            clique_size,
            target_bin=seed_bin
        )
        scores.append(calculate_avg_interaction_strength(contact_matrix, clique))

    if display:
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=50, edgecolor='black')
        plt.xlabel('Average Interaction Score')
        plt.ylabel('Frequency')
        plt.title(
            f'Distribution of AIS ({label}) – '
            f'{num_iterations} random cliques of size {clique_size}'
        )
        plt.tight_layout()
        plt.show()

    if write:   
        fname = f'greedy_scores_{clique_size}_iters_{num_iterations}_{label}.txt'
        outpath = os.path.join(output_dir, fname)
        with open(outpath, 'w') as fh:
            for s in scores:
                fh.write(f"{s}\n")
    
    return scores




def create_rw(
    contact_matrix,
    clique_size,
    num_iterations=1000,
    bins=None,
    neighbors=None,
    cdfs=None,
    num_molecules=1000,
    alpha=0.05,
    label=None,
    plot=True,
    write=True,
):
    """
    Generate a background distribution of average interaction scores
    using random walks.
    """

    if bins is None:
        bins = np.arange(contact_matrix.shape[0])
    if label is None:
        label = 'all'

    output_dir = 'background_models/random_walk'
    os.makedirs(output_dir, exist_ok=True)

    scores = np.empty(num_iterations)

    for i in tqdm(range(num_iterations), desc="Random walks", unit="iter"):
        seed = np.random.choice(bins)
        clique = random_walk(
            contact_matrix,
            seed,
            clique_size,
            neighbors=neighbors,
            cdfs=cdfs,
            num_molecules=num_molecules,
            alpha=alpha
        )
        scores[i] = calculate_avg_interaction_strength(contact_matrix, clique)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=50, edgecolor='black')
        plt.xlabel('Average Interaction Score')
        plt.ylabel('Frequency')
        plt.title(
            f'Distribution of AIS ({label}) — '
            f'{num_molecules} walks of length {clique_size}'
        )
        plt.tight_layout()
        plt.show()

    if write:
        fname = f'rw_scores_{clique_size}_molecules_{num_molecules}_iters_{num_iterations}_alpha_{alpha}_{label}.txt'
        path = os.path.join(output_dir, fname)
        np.savetxt(path, scores, fmt='%.6f')  

    return scores



import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def create_rw_parallel(
    contact_matrix,
    clique_size,
    bins=None,
    neighbors=None,
    cdfs=None,
    num_molecules=100,
    num_iterations=1000,
    alpha=0.05,
    label=None,
    plot=True,
    write=True,
    n_jobs=-1,  # <-- new: number of parallel workers, -1 = use all CPUs
):
    """
    Generate a background distribution of average interaction scores
    using random walks.
    """

    if bins is None:
        bins = np.arange(contact_matrix.shape[0])
    if label is None:
        label = 'all'

    output_dir = 'background_models/random_walk'
    os.makedirs(output_dir, exist_ok=True)

    def single_random_walk():
        seed = np.random.choice(bins)
        clique = random_walk(
            contact_matrix,
            seed,
            clique_size,
            neighbors=neighbors,
            cdfs=cdfs,
            num_molecules=num_molecules,
            alpha=alpha
        )
        return calculate_avg_interaction_strength(contact_matrix, clique)

    # Run random walks in parallel
    scores = Parallel(n_jobs=n_jobs)(
        delayed(single_random_walk)() for _ in tqdm(range(num_iterations), desc="Random walks", unit="iter")
    )
    scores = np.array(scores)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=50, edgecolor='black')
        plt.xlabel('Average Interaction Score')
        plt.ylabel('Frequency')
        plt.title(
            f'Distribution of AIS ({label}) — '
            f'{num_molecules} walks of length {clique_size}'
        )
        plt.tight_layout()
        plt.show()

    if write:
        fname = f'rw_scores_{clique_size}_molecules_{num_molecules}_iters_{num_iterations}_alpha_{alpha}_{label}.txt'
        path = os.path.join(output_dir, fname)
        np.savetxt(path, scores, fmt='%.6f')

    return scores