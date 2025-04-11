# Helper functions for the main script
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from tqdm import tqdm

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




# Background Model Generation
def create_background_model_greedy(contact_matrix, clique_size, bin_map, num_iterations=1000, display=True):
    interaction_scores = []

    # Use tqdm to display a progress bar
    for _ in tqdm(range(num_iterations), desc="Processing", unit="iteration"):
        random_clique = find_clique_greedy(contact_matrix, clique_size, bin_map=bin_map)
        score = calculate_avg_interaction_strength(contact_matrix, random_clique)
        interaction_scores.append(score)

    if display:
        plt.figure(figsize=(10, 6))
        plt.hist(interaction_scores, bins=50, color='skyblue', edgecolor='black')
        plt.xlabel('Average Interaction Score')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of AIS using Greedy for {num_iterations} Random Cliques of Size {clique_size}')
        plt.show()

    filename = f'/Users/ethan/Desktop/chromatin-heart-dev/background_models/greedy_scores_{clique_size}_iterations_{num_iterations}.txt'
    with open(filename, 'w') as f: 
        for item in interaction_scores:
            f.write("%s\n" % item)

    return interaction_scores

def create_background_model_greedy_strong(contact_matrix, clique_size, bin_map, gene_bins, num_iterations=1000, display=True):
    interaction_scores = []

    # Use tqdm to display a progress bar
    for _ in tqdm(range(num_iterations), desc="Processing", unit="iteration"):
        random_idx = np.random.choice(gene_bins) 
        random_clique = find_clique_greedy(contact_matrix, clique_size, bin_map=bin_map, target_bin=random_idx)
        score = calculate_avg_interaction_strength(contact_matrix, random_clique)
        interaction_scores.append(score)

    if display:
        plt.figure(figsize=(10, 6))
        plt.hist(interaction_scores, bins=50, color='skyblue', edgecolor='black')
        plt.xlabel('Average Interaction Score')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of AIS using Greedy for {num_iterations} Random Cliques of Size {clique_size}')
        plt.show()

    filename = f'/Users/ethan/Desktop/chromatin-heart-dev/background_models/greedy_scores_{clique_size}_iterations_{num_iterations}_strong.txt'
    with open(filename, 'w') as f: 
        for item in interaction_scores:
            f.write("%s\n" % item)

    return interaction_scores

def create_background_model_greedy_weak(contact_matrix, clique_size, bin_map, non_gene_bins, num_iterations=1000, display=True):
    interaction_scores = []

    # Use tqdm to display a progress bar
    for _ in tqdm(range(num_iterations), desc="Processing", unit="iteration"):
        random_idx = np.random.choice(non_gene_bins) 
        random_clique = find_clique_greedy(contact_matrix, clique_size, bin_map=bin_map, target_bin=random_idx)
        score = calculate_avg_interaction_strength(contact_matrix, random_clique)
        interaction_scores.append(score)

    if display:
        plt.figure(figsize=(10, 6))
        plt.hist(interaction_scores, bins=50, color='skyblue', edgecolor='black')
        plt.xlabel('Average Interaction Score')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of AIS using Greedy for {num_iterations} Random Cliques of Size {clique_size}')
        plt.show()

    filename = f'/Users/ethan/Desktop/chromatin-heart-dev/background_models/greedy_scores_{clique_size}_iterations_{num_iterations}_weak.txt'
    with open(filename, 'w') as f: 
        for item in interaction_scores:
            f.write("%s\n" % item)

    return interaction_scores


def create_background_model_rw(contact_matrix, n, num_molecules=100, num_iterations=1000, alpha=0.05):   
    interaction_scores = []

    for _ in tqdm(range(num_iterations), desc="Processing", unit="iteration"):
        random_idx = np.random.randint(0, contact_matrix.shape[0])  
        random_clique = random_walk(contact_matrix, random_idx, n, num_molecules=num_molecules, alpha=alpha)
        interaction_score = calculate_avg_interaction_strength(contact_matrix, random_clique)
        interaction_scores.append(interaction_score)

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.hist(interaction_scores, bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('Average Interaction Score')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Average Interaction Scores for {num_molecules} Random Walks of Size {n}')
    plt.show()

    filename = f'/Users/ethan/Desktop/chromatin-heart-dev/background_models/rw_scores_{n}_molecules_{num_molecules}_iterations_{num_iterations}.txt'
    with open(filename, 'w') as f: 
        for item in interaction_scores:
            f.write("%s\n" % item)

    return interaction_scores


def create_background_model_rw_strong(contact_matrix, n, gene_bins, num_molecules=100, num_iterations=1000, alpha=0.05, plot=False):    

    interaction_scores = []

    for _ in tqdm(range(num_iterations), desc="Processing", unit="iteration"):
        random_idx = np.random.choice(gene_bins) 
        random_clique = random_walk(contact_matrix, random_idx, n, num_molecules=num_molecules, alpha=alpha)
        interaction_score = calculate_avg_interaction_strength(contact_matrix, random_clique)
        interaction_scores.append(interaction_score)

    # Plot the distribution
    if plot:
        plt.figure(figsize=(10, 6))
        plt.hist(interaction_scores, bins=50, color='skyblue', edgecolor='black')
        plt.xlabel('Average Interaction Score')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Average Interaction Scores for {num_molecules} Random Walks of Size {n}')
        plt.show()

    filename = f'/Users/ethan/Desktop/chromatin-heart-dev/background_models/rw_scores_{n}_molecules_{num_molecules}_iterations_{num_iterations}_strong.txt'
    with open(filename, 'w') as f: 
        for item in interaction_scores:
            f.write("%s\n" % item)

    return interaction_scores


def create_background_model_rw_weak(contact_matrix, n, non_gene_bins, num_molecules=100, num_iterations=1000, alpha=0.05, plot=False):  
    interaction_scores = []

    for _ in tqdm(range(num_iterations), desc="Processing", unit="iteration"):
        random_idx = np.random.choice(non_gene_bins) 
        random_clique = random_walk(contact_matrix, random_idx, n, num_molecules=num_molecules, alpha=alpha)
        interaction_score = calculate_avg_interaction_strength(contact_matrix, random_clique)
        interaction_scores.append(interaction_score)

    # Plot the distribution
    if plot:
        plt.figure(figsize=(10, 6))
        plt.hist(interaction_scores, bins=50, color='skyblue', edgecolor='black')
        plt.xlabel('Average Interaction Score')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Average Interaction Scores for {num_molecules} Random Walks of Size {n}')
        plt.show()

    filename = f'/Users/ethan/Desktop/chromatin-heart-dev/background_models/rw_scores_{n}_molecules_{num_molecules}_iterations_{num_iterations}_weak.txt'
    with open(filename, 'w') as f: 
        for item in interaction_scores:
            f.write("%s\n" % item)

    return interaction_scores


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

def get_bins_not_on_gene(bin_map, gtf_file_path):   
    gene_bins = []
    with open('/Users/ethan/Desktop/chromatin-heart-dev/data/gene_bins.txt', 'r') as file:
        for line in file:
            gene_bins.append(line.strip())
    gene_bins = [int(x) for x in gene_bins]

    # All bins [0 to 30894]
    all_bins = set(range(0, 30894))
    bins_not_on_genes = all_bins - set(gene_bins)


    return list(bins_not_on_genes)





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





# Helpers for post clustering analysis
def parse_gene_name(attribute):
    match = re.search(r'gene_name "([^"]+)"', attribute)
    return match.group(1) if match else None

def get_gene_abbreviations_in_node(chromosome, bin_id, node_bed_path, gtf_file_path):
    nodes_df = pd.read_csv(node_bed_path, sep="\t", header=None, names=["chrom", "start", "end", "bin"])
    node_intervals = nodes_df[(nodes_df["chrom"] == chromosome) & (nodes_df["bin"] == bin_id)]
    print('node_intervals:', node_intervals)    

    if node_intervals.empty:
        print(f"No intervals found for chromosome {chromosome} and bin {bin_id}.")
        return []
    
    gtf_cols = ["chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]
    gtf_df = pd.read_csv(gtf_file_path, sep="\t", comment="#", header=None, names=gtf_cols)
    genes_df = gtf_df[gtf_df["feature"] == "gene"]
    genes_df = genes_df[genes_df["chrom"] == str(chromosome)]
    genes_df["gene_name"] = genes_df["attribute"].apply(parse_gene_name)
    
    gene_abbrevs = set()
    for _, node in node_intervals.iterrows():
        overlaps = genes_df[(genes_df["start"] < node["end"]) & (genes_df["end"] > node["start"])]
        gene_abbrevs.update(overlaps["gene_name"].dropna().tolist())
    
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

def construct_graph_from_contact_matrix(contact_matrix, threshold=0):
    if isinstance(contact_matrix, np.ndarray):
        contact_matrix = pd.DataFrame(contact_matrix)
    
    if contact_matrix.shape[0] != contact_matrix.shape[1]:
        raise ValueError("Contact matrix must be square.")
    
    graph = nx.Graph()
    num_bins = contact_matrix.shape[0]

    # Add nodes to the graph 
    graph.add_nodes_from(range(num_bins))
    
    for i in range(num_bins):
        for j in range(i + 1, num_bins): 
            weight = contact_matrix.iloc[i, j]
            if weight > threshold:
                graph.add_edge(i, j, weight=weight)

    # Define positions using spring layout
    pos = nx.spring_layout(graph, seed=42, k=1.5)
    plt.figure(figsize=(10, 10))
    
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_size=500, node_color='lightblue', edgecolors='black')
    
    # Draw edges with width proportional to weight
    edges = graph.edges(data=True)
    nx.draw_networkx_edges(graph, pos, edgelist=edges, width=[d['weight'] * 0.2 for (_, _, d) in edges], alpha=0.7)
    
    nx.draw_networkx_labels(graph, pos, font_size=12, font_color='black', font_weight='bold')
    
    # Draw edge labels (weights)
    edge_labels = {(i, j): f"{d['weight']:.1f}" for i, j, d in edges}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10, font_color='red')
    
    plt.title("Graph Representation of Contact Matrix", fontsize=14)
    plt.show()
    
    return graph

