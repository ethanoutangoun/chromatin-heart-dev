# Helper functions for the main script
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import os

# Clique finding algorithms
def find_clique_greedy(contact_matrix, n, target_bin=None, bin_map=None):
    if target_bin is not None:
        clique = [target_bin]
        chromosomes = {bin_map.get(target_bin, None)}
    else:
        initial_node = np.random.randint(0, contact_matrix.shape[0])
        clique = [initial_node]
        chromosomes = {bin_map.get(initial_node, None)}

    excluded_bins = set()
    edges = []

    while len(clique) < n:
        max_mean_edge = -1
        max_node = None
        best_connection = None

        for node in range(contact_matrix.shape[0]):
            if node not in clique and node not in excluded_bins:
                # Calculate mean interaction with all nodes in the clique
                sum_connections = 0
                count = 0

                for c in clique:
                    if bin_map.get(node, None) == bin_map.get(c, None):
                        sum_connections += 0
                    else:
                        sum_connections += contact_matrix[node, c]
                    count += 1

                mean_connection = sum_connections / count if count > 0 else 0


                if mean_connection > max_mean_edge:
                    max_mean_edge = mean_connection
                    max_node = node
                    best_connection = (node, max(clique, key=lambda c: contact_matrix[node, c]), mean_connection)

        if max_node is None:
            break

        clique.append(max_node)
        
        # exclude nodes +-10 bins from the selected node
        excluded_bins.add(max_node)
        for i in range(max_node - 10, max_node + 11):
            excluded_bins.add(i)

        chromosomes.add(bin_map.get(max_node, None))
        edges.append(best_connection)
        
    return clique

def find_clique_greedy_fast(contact_matrix, n, target_bin=None):
    """
    Greedy clique of size n, optimized via NumPy vector operations.
    Assumes contact_matrix[i, j] == 0 whenever i and j are on the same chromosome.
    """
    N = contact_matrix.shape[0]
    # 1) pick a starting node
    if target_bin is None:
        current = np.random.randint(N)
    else:
        current = target_bin

    clique = [current]

    # 2) boolean array of bins we cannot pick
    excluded = np.zeros(N, dtype=bool)
    # exclude ±10 around the start
    lo, hi = max(0, current - 10), min(N, current + 11)
    excluded[lo:hi] = True

    # 3) greedy growing
    while len(clique) < n:
        # a) compute total interaction of every node with the current clique
        #    this gives an (N,) array of sums
        sums = contact_matrix[:, clique].sum(axis=1)

        # b) mean = sums / |clique|
        means = sums / len(clique)

        # c) mask out already excluded or already in clique
        means[excluded] = -np.inf
        means[clique]   = -np.inf

        # d) pick the best new node
        max_node = int(np.argmax(means))
        if means[max_node] == -np.inf:
            # no more valid candidates
            break

        # e) add to clique and update exclusion window
        clique.append(max_node)
        lo, hi = max(0, max_node - 10), min(N, max_node + 11)
        excluded[lo:hi] = True

    return clique

def random_walk(contact_matrix, start_node, n, num_molecules=100, alpha=0.1, verbose=False):
    num_nodes = contact_matrix.shape[0]
    visit_count = np.zeros(num_nodes, dtype=int)
    
    iterator = tqdm(range(num_molecules)) if verbose else range(num_molecules)
    
    for _ in iterator: 
        current_node = start_node
        
        while True:
            visit_count[current_node] += 1  # Track visits per molecule
            
            if np.random.rand() < alpha:
                break
            
            neighbors = np.where(contact_matrix[current_node] > 0)[0]
            if len(neighbors) == 0:
                break
            
            weights = contact_matrix[current_node, neighbors]
            probabilities = weights / np.sum(weights)
            
            next_node = np.random.choice(neighbors, p=probabilities)
            current_node = next_node

    clique = np.argsort(visit_count)[-n:][::-1]


    
    return clique

def random_walk_fast(contact_matrix, start_node, n,
                     neighbors, cdfs,
                     num_molecules=100, alpha=0.1):
    """
    A much faster random‐walk using prebuilt neighbor/CDF lists.
    """
    N = contact_matrix.shape[0]
    visit_count = np.zeros(N, dtype=int)

    for _ in range(num_molecules):
        cur = start_node
        while True:
            visit_count[cur] += 1
            if np.random.rand() < alpha or neighbors[cur].size == 0:
                break
            r = np.random.rand()
            # find next index in CDF
            j = np.searchsorted(cdfs[cur], r, side='right')
            cur = neighbors[cur][j]

    # top-n visited
    return np.argsort(visit_count)[-n:][::-1]

def create_background_model_greedy(
    contact_matrix,
    clique_size,
    bin_map,
    bins,
    label=None,
    num_iterations=1000,
    display=True,
    write=True,
):
    """
    Generate a background distribution of average interaction scores by
    repeatedly finding random greedy cliques.

    Parameters
    ----------
    contact_matrix : array-like
        Hi-C contact matrix.
    clique_size : int
        Number of bins in each clique.
    bin_map : dict
        Mapping from bin index to matrix coordinates.
    bins : sequence of int
        List of bin indices to sample from (e.g. gene_bins or non_gene_bins).
    label : str
        Identifier to append to the output filename (e.g. 'strong' or 'weak').
    num_iterations : int, optional
        How many random cliques to sample (default 1000).
    display : bool, optional
        If True, show a histogram of the scores (default True).
    output_dir : str or None, optional
        Directory to write the score file. If None, uses cwd/background_models.
    """
    if label is None:
        label = 'all'

    output_dir = os.path.join(os.getcwd(), 'background_models', 'random_walk')
    os.makedirs(output_dir, exist_ok=True)

    scores = []
    for _ in tqdm(range(num_iterations), desc="Sampling cliques", unit="iter"):
        seed_bin = np.random.choice(bins)
        clique = find_clique_greedy(
            contact_matrix,
            clique_size,
            bin_map=bin_map,
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



def create_background_model_rw(
    contact_matrix,
    n,
    bins=None,
    label=None,
    neighbors=None,
    cdfs=None,
    num_molecules=100,
    num_iterations=1000,
    alpha=0.05,
    plot=True,
    write=True,
):
    """
    Generate a background distribution of average interaction scores
    using random walks.

    Parameters
    ----------
    contact_matrix : array-like
        Hi-C contact matrix.
    n : int
        Length of each random walk.
    bins : sequence of int, optional
        Which bin-indices to seed from (e.g. gene_bins or non_gene_bins).
        If None, samples from all bins (0..matrix.shape[0]-1).
    label : str, optional
        Text to include in the output filename/title (e.g. 'strong' or 'weak').
        If None, defaults to 'all'.
    num_molecules : int, optional
        Number of walkers per random walk (default 100).
    num_iterations : int, optional
        Number of random walks to draw (default 1000).
    alpha : float, optional
        Restart probability for the random walk (default 0.05).
    plot : bool, optional
        Whether to display a histogram (default True).
    output_dir : str, optional
        Directory to write the score file. If None, uses
        cwd/background_models.
    """
    

    # prepare bins and label
    if bins is None:
        bins = np.arange(contact_matrix.shape[0])
    if label is None:
        label = 'all'

    # prepare output directory
    output_dir = os.path.join(os.getcwd(), 'background_models', 'random_walk')
    os.makedirs(output_dir, exist_ok=True)

    # sample
    scores = []
    for _ in tqdm(range(num_iterations), desc="Random walks", unit="iter"):
        seed = np.random.choice(bins)
        clique = random_walk_fast(
            contact_matrix,
            seed,
            n,
            neighbors= neighbors,
            cdfs=cdfs,
            num_molecules=num_molecules,
            alpha=alpha
        )
        scores.append(calculate_avg_interaction_strength(contact_matrix, clique))

    # plot
    if plot:
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=50, edgecolor='black')
        plt.xlabel('Average Interaction Score')
        plt.ylabel('Frequency')
        plt.title(
            f'Distribution of AIS ({label}) — '
            f'{num_molecules} walks of length {n}'
        )
        plt.tight_layout()
        plt.show()

    if write:
        fname = f'rw_scores_{n}_molecules_{num_molecules}_iters_{num_iterations}_alpha_{alpha}_{label}.txt'
        path = os.path.join(output_dir, fname)
        with open(path, 'w') as fh:
            for s in scores:
                fh.write(f"{s}\n")

    return scores



# Scoring and testing functions
def calculate_avg_interaction_strength(contact_matrix, clique):
    total_interaction_strength = 0
    num_edges = 0

    # Loop through each unique pair of bins in the clique to get score of each edge
    for i in range(len(clique)):
        for j in range(i + 1, len(clique)):
            bin1 = clique[i]
            bin2 = clique[j]
            total_interaction_strength += contact_matrix[bin1, bin2]
            num_edges += 1

    # Calculate average interaction strength
    avg_interaction_strength = total_interaction_strength / num_edges if num_edges > 0 else 0
    return avg_interaction_strength

def simple_p_test(observed_score, random_scores):
    return np.mean(random_scores >= observed_score)

def permutation_test(contact_matrix, start_node, n, num_molecules=100, alpha=0.1, num_permutations=1000, verbose=False):
    real_top_nodes = random_walk(contact_matrix, start_node, n, num_molecules, alpha, verbose)

    permuted_top_nodes = []

    # shuffle only contact weights
    for _ in tqdm(range(num_permutations), desc="Permutation Tests" if verbose else None):
        permuted_matrix = contact_matrix.copy()
        for i in range(permuted_matrix.shape[0]):
            np.random.shuffle(permuted_matrix[i])  

        permuted_top_nodes.append(random_walk(permuted_matrix, start_node, n, num_molecules, alpha, verbose))
    
    p_values = np.zeros(n)
    
    for i, node in enumerate(real_top_nodes):
        null_distribution = [top_nodes[i] for top_nodes in permuted_top_nodes]
        p_value = np.mean(np.array(null_distribution) == node)
        p_values[i] = p_value
    
    return real_top_nodes, p_values, permuted_top_nodes



# Function for displaying found clique as a fully connected graph
def clique_to_graph(contact_matrix, clique, selected_bin = None):
    edges = []

    for i in range(len(clique)):
        for j in range(i+1, len(clique)):
            edges.append((clique[i], clique[j], contact_matrix[clique[i], clique[j]]))

    # create a graph
    G = nx.Graph()
    G.add_nodes_from(clique)
    G.add_weighted_edges_from(edges)
    node_colors = ['red' if bin == selected_bin else 'skyblue' for bin in clique]

    # visualize the graph
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color=node_colors,
            font_size=15, font_weight='bold', edge_color='gray')

    labels = nx.get_edge_attributes(G, 'weight')
    labels = {k: f"{v:.5f}" for k, v in labels.items()}

    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=12, font_weight='bold')
    plt.show()

    return G




# Finds all genes overlapping with a given bin
def find_gene_from_bin(bin_id, node_bed_path, gtf_file_path):
    def parse_gene_name(attribute):
        match = re.search(r'gene_name "([^"]+)"', attribute)
        return match.group(1) if match else None

    nodes_df = pd.read_csv(node_bed_path, sep="\t", header=None,
                           names=["chrom", "start", "end", "bin"])
    node_intervals = nodes_df[nodes_df["bin"] == bin_id]
    if node_intervals.empty:
        return []

    gtf_cols = ["chrom", "source", "feature", "start", "end",
                "score", "strand", "frame", "attribute"]
    gtf_df = pd.read_csv(gtf_file_path, sep="\t", comment="#",
                         header=None, names=gtf_cols)
    
    genes_df = gtf_df.loc[gtf_df["feature"] == "gene"].copy()
    genes_df["gene_name"] = genes_df["attribute"].apply(parse_gene_name)

    # collect overlaps
    gene_abbrevs = set()
    for _, node in node_intervals.iterrows():
        overlaps = genes_df[
            (genes_df["start"] < node["end"]) &
            (genes_df["end"] > node["start"])
        ]
        gene_abbrevs.update(overlaps["gene_name"].dropna())

    return list(gene_abbrevs)



def find_ttn_bin(gtf_file_path, node_bed_path):
    gtf_cols = ["chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]
    gtf_df = pd.read_csv(gtf_file_path, sep="\t", comment="#", header=None, names=gtf_cols)
    genes_df = gtf_df[gtf_df["feature"] == "gene"]
    ttn_gene = genes_df[genes_df["attribute"].str.contains('gene_name "TTN"')]
    
    if ttn_gene.empty:
        print("TTN gene not found in the GTF file.")
        return None
    
    ttn_chrom = ttn_gene["chrom"].iloc[0]
    ttn_start = int(ttn_gene["start"].iloc[0])
    ttn_end = int(ttn_gene["end"].iloc[0])
    
    nodes_df = pd.read_csv(node_bed_path, sep="\t", header=None, names=["chrom", "start", "end", "bin"])
    overlapping_bins = nodes_df[(nodes_df["chrom"] == ttn_chrom) &
                                (nodes_df["start"] < ttn_end) &
                                (nodes_df["end"] > ttn_start)]
    
    if overlapping_bins.empty:
        print("No bins overlap with the TTN gene.")
        return None
    
    return overlapping_bins["bin"].tolist()


def load_bin_map(file_path):
    bin_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 4:
                chromosome, _, _, bin_id = parts
                bin_dict[int(bin_id)] = chromosome
    return bin_dict


def get_chromosome_span(bins, bin_map):
    chromosomes = set()
    for bin_id in bins:
        chromosome = bin_map[bin_id]
        chromosomes.add(chromosome)

    return chromosomes


def load_bin_map_loc(file_path):
    bin_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 4:
                chromosome, start, end, bin_id = parts
                start, end, bin_id = int(start), int(end), int(bin_id)

                # Store bin_id for the entire range
                if chromosome not in bin_dict:
                    bin_dict[chromosome] = []
                bin_dict[chromosome].append((start, end, bin_id))

    # Sort ranges for faster querying
    for chrom in bin_dict:
        bin_dict[chrom].sort()

    return bin_dict

import bisect

def find_bin(chromosome, position, bin_dict):
    if chromosome not in bin_dict:
        return None  # Chromosome not found
    
    bins = bin_dict[chromosome]
    idx = bisect.bisect_left(bins, (position,))  # Find closest start position

    if idx > 0 and bins[idx - 1][0] <= position <= bins[idx - 1][1]:
        return bins[idx - 1][2]  # Return bin_id

    return None  # Position not found in any range




# Helpers for generating and visualizing graphs
def generate_sample_matrix_bins(n_bins):
    np.random.seed(42)
    
    dense_regions = [
        (0, n_bins // 10),    # Community 1
        (n_bins // 10, n_bins // 5),  # Community 2
        (n_bins // 5, n_bins // 2)  # Community 3
    ]

    contact_matrix = np.random.rand(n_bins, n_bins) * 2

    for start, end in dense_regions:
        for i in range(start, end + 1):
            for j in range(start, end + 1):
                if i != j:
                    contact_matrix[i, j] = np.random.rand() * 8 + 2  

    # Make matrix symmetric
    contact_matrix = (contact_matrix + contact_matrix.T) / 2
    np.fill_diagonal(contact_matrix, 0)

    return contact_matrix

