# Helper functions for the main script
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import os






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


def build_walk_index(contact_matrix):
    """
    Precompute for each node:
      - neighbors[i]: 1D int array of neighbors
      - cdfs[i]:      1D float array of cumulative probabilities
    """
    N = contact_matrix.shape[0]
    neighbors = [None]*N
    cdfs      = [None]*N

    for i in tqdm(range(N)):
        w = contact_matrix[i]
        idx = np.nonzero(w)[0]
        if idx.size == 0:
            neighbors[i] = np.empty(0, dtype=int)
            cdfs[i]      = np.empty(0, dtype=float)
        else:
            probs = w[idx] / w[idx].sum()
            neighbors[i] = idx
            cdfs[i]      = np.cumsum(probs)
    return neighbors, cdfs


import bisect

def find_bin(chromosome, position, bin_dict):
    if chromosome not in bin_dict:
        return None  # Chromosome not found
    
    bins = bin_dict[chromosome]
    idx = bisect.bisect_left(bins, (position,))  # Find closest start position

    if idx > 0 and bins[idx - 1][0] <= position <= bins[idx - 1][1]:
        return bins[idx - 1][2]  # Return bin_id

    return None  # Position not found in any range


def get_bins_on_gene(bin_map, gtf_file_path):   
    gtf_cols = ["chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]
    gtf_df = pd.read_csv(gtf_file_path, sep="\t", comment="#", header=None, names=gtf_cols)
    genes_df = gtf_df[gtf_df["feature"] == "gene"]

    bins_on_genes = set()

    for _, row in genes_df.iterrows():
        gene_start = row["start"]
        gene_chrom = row["chrom"]

        bin = find_bin(gene_chrom, gene_start, bin_map)    
        if bin is not None:
            bins_on_genes.add(bin)

    return list(bins_on_genes)

def get_bins_not_on_gene(gene_bins, num_bins):   
    gene_bins = [int(x) for x in gene_bins]

    # All bins [0 to 30894]
    all_bins = set(range(0, num_bins))
    bins_not_on_genes = all_bins - set(gene_bins)


    return list(bins_not_on_genes)

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
