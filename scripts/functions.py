# Helper functions for the main script

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# GRAPH FUNCTIONS
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

def generate_sample_matrix():
    np.random.seed(42)
    n_nodes = 15
    dense_regions = [
        (0, 4),   # Community 1
        (5, 9),   # Community 2
        (10, 14)  # Community 3
    ]

    contact_matrix = np.random.rand(n_nodes, n_nodes) * 2
    for start, end in dense_regions:
        for i in range(start, end + 1):
            for j in range(start, end + 1):
                if i != j:
                    contact_matrix[i, j] = np.random.rand() * 8 + 2 

    # Make matrix symmetric
    contact_matrix = (contact_matrix + contact_matrix.T) / 2
    np.fill_diagonal(contact_matrix, 0)

    return contact_matrix

def generate_random_matrix():
    np.random.seed(42)
    n_nodes = 10
    contact_matrix = np.random.rand(n_nodes, n_nodes)
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
    plt.savefig("graph_verbose.png")
    plt.show()
    
    return graph

def clique_to_graph(graph_data, spacing=1.0, selected_bin=None, verbose=False):
    nodes = graph_data['nodes']
    edges = graph_data['edges']

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(edges)  
    

    if not verbose:
      return G


    pos = nx.spring_layout(G, k=spacing, seed=42)  # Adjust spacing

    plt.figure(figsize=(8, 6))  # Increase figure size for better visibility

    # Set node colors: red for selected_bin, skyblue for others
    node_colors = ['red' if node == selected_bin else 'skyblue' for node in nodes]

    nx.draw(G, pos, with_labels=True, node_size=700, node_color=node_colors, 
            font_size=15, font_weight='bold', edge_color='gray')

    labels = nx.get_edge_attributes(G, 'weight')
    labels = {k: f"{v:.5f}" for k, v in labels.items()}  # Format weights to 5 decimals
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=12, font_weight='bold')

    plt.show()
    return G



# HELPER FUNCTIONS
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


import numpy as np

def find_clique_greedy(contact_matrix, n, target_bin=None, bin_map=None):
    if target_bin is not None:
        clique = [target_bin]
        chromosomes = {bin_map.get(target_bin, None)}
    else:
        initial_node = np.random.randint(0, contact_matrix.shape[0])
        clique = [initial_node]
        chromosomes = {bin_map.get(initial_node, None)}

    edges = []

    while len(clique) < n:
        max_mean_edge = -1
        max_node = None
        best_connection = None

        for node in range(contact_matrix.shape[0]):

            if node not in clique:
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
        chromosomes.add(bin_map.get(max_node, None))
        edges.append(best_connection)
        
        
        total_weight = sum(data[2] for data in edges)

        avg_interaction_score = total_weight / len(edges) if len(edges) > 0 else 0

    return {"nodes": clique, "edges": edges, "score": avg_interaction_score}








import matplotlib.pyplot as plt
from tqdm import tqdm

def create_background_model_greedy(contact_matrix, clique_size, bin_map, num_iterations=1000):
    interaction_scores = []

    # Use tqdm to display a progress bar
    for _ in tqdm(range(num_iterations), desc="Processing", unit="iteration"):
        random_clique = find_clique_greedy(contact_matrix, clique_size, bin_map=bin_map)
        interaction_scores.append(random_clique['score'])

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.hist(interaction_scores, bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('Average Interaction Score')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Average Interaction Scores for {num_iterations} Random Cliques of Size {clique_size}')
    plt.show()

    return interaction_scores

def create_background_model_rw(contact_matrix, n, num_steps=1000, num_molecules=100, num_iterations=1000):
    interaction_scores = []

    for _ in tqdm(range(num_iterations), desc="Processing", unit="iteration"):
        random_idx = np.random.randint(0, contact_matrix.shape[0])  
        print('random_idx:', random_idx)
        random_clique, interaction_score = random_walk(contact_matrix, random_idx, n, num_steps=num_steps)
        interaction_scores.append(interaction_score)

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.hist(interaction_scores, bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('Average Interaction Score')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Average Interaction Scores for {num_molecules} Random Walks of Size {n}')
    plt.show()

    return interaction_scores




import numpy as np

def calculate_avg_interaction_strength(node_indices, contact_matrix):
    total_interaction_strength = 0
    num_edges = 0

    # Loop through each unique pair of nodes in the array
    for i in range(len(node_indices)):
        for j in range(i + 1, len(node_indices)):
            node1 = node_indices[i]
            node2 = node_indices[j]
            total_interaction_strength += contact_matrix[node1, node2]
            num_edges += 1

    # Calculate average interaction strength
    avg_interaction_strength = total_interaction_strength / num_edges if num_edges > 0 else 0
    return avg_interaction_strength

def random_walk(contact_matrix, start_node, n, num_steps=1000, num_molecules=100):
    num_nodes = contact_matrix.shape[0]
    visit_count = np.zeros(num_nodes, dtype=int)
    
    for _ in tqdm(range(num_molecules)):  # Inject multiple fluid molecules
        current_node = start_node
        
        for _ in range(num_steps):
            visit_count[current_node] += 1  # Track visits per molecule
            
            neighbors = np.where(contact_matrix[current_node] > 0)[0]
            if len(neighbors) == 0:
                break
            
            weights = contact_matrix[current_node, neighbors]
            probabilities = weights / np.sum(weights)
            
            next_node = np.random.choice(neighbors, p=probabilities)
            current_node = next_node

    # Get top 5 nodes with highest visit count
    top_nodes = np.argsort(visit_count)[-n:][::-1]

    interaction_scores = calculate_avg_interaction_strength(top_nodes, contact_matrix)
    
    return top_nodes, interaction_scores


    




import numpy as np
from tqdm import tqdm

def stable_random_walk(contact_matrix, start_nodes, num_steps=1000, num_molecules=100, top_n=5, num_runs=10):
    num_nodes = contact_matrix.shape[0]
    total_visit_count = np.zeros(num_nodes, dtype=float)

    for _ in tqdm(range(num_runs)):  # Run multiple times for stability
        visit_count = np.zeros(num_nodes, dtype=int)
        
        for _ in range(num_molecules):  
            current_node = np.random.choice(start_nodes)  # Choose a random starting node
            
            for _ in range(num_steps):
                visit_count[current_node] += 1  
                
                neighbors = np.where(contact_matrix[current_node] > 0)[0]
                if len(neighbors) == 0:
                    break
                
                weights = contact_matrix[current_node, neighbors]
                probabilities = weights / np.sum(weights)
                
                next_node = np.random.choice(neighbors, p=probabilities)
                current_node = next_node

        total_visit_count += visit_count / num_molecules  # Normalize by molecules per run

    mean_visit_count = total_visit_count / num_runs  # Average across runs

    # Get top 5 nodes with highest average visit count excluding the starting nodes

    filter_count = top_n + len(start_nodes)

    top_nodes = np.argsort(mean_visit_count)[-filter_count:][::-1]  # Get most visited nodes
    
    return top_nodes, mean_visit_count




import numpy as np
from tqdm import tqdm
from scipy.stats import ttest_ind

