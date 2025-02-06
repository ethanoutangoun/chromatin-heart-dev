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
