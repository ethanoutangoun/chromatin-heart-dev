# This file constructs a graph from a contact matrix for downstream analysis.

import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def construct_graph_from_contact_matrix(contact_matrix, threshold=0):
    """
    Constructs and visualizes a graph from a contact matrix with detailed annotations.

    Args:
        contact_matrix (pd.DataFrame or np.ndarray): The contact matrix where rows and columns represent individuals.
        threshold (float): The minimum contact strength to consider an edge.

    Returns:
        nx.Graph: The constructed graph.
    """
    if isinstance(contact_matrix, np.ndarray):
        contact_matrix = pd.DataFrame(contact_matrix)
    
    if contact_matrix.shape[0] != contact_matrix.shape[1]:
        raise ValueError("Contact matrix must be square.")
    
    graph = nx.Graph()
    num_bins = contact_matrix.shape[0]

    # Add nodes to the graph 
    graph.add_nodes_from(range(num_bins))
    
    # Add edges based on contact strength and threshold
    for i in range(num_bins):
        for j in range(i + 1, num_bins): 
            weight = contact_matrix.iloc[i, j]
            if weight > threshold:
                graph.add_edge(i, j, weight=weight)

    # Define positions using spring layout
    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(10, 10))
    
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_size=500, node_color='lightblue', edgecolors='black')
    
    # Draw edges with width proportional to weight
    edges = graph.edges(data=True)
    nx.draw_networkx_edges(graph, pos, edgelist=edges, width=[d['weight'] * 0.2 for (_, _, d) in edges], alpha=0.7)
    
    # Draw labels
    nx.draw_networkx_labels(graph, pos, font_size=12, font_color='black', font_weight='bold')
    
    # Draw edge labels (weights)
    edge_labels = {(i, j): f"{d['weight']:.1f}" for i, j, d in edges}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10, font_color='red')
    
    plt.title("Graph Representation of Contact Matrix", fontsize=14)
    plt.savefig("graph_verbose.png")
    plt.show()
    
    return graph


# Load the contact matrix
matrix = np.load("samples/contact_matrix_100kb_balanced.npy")

# Create a graph from the contact matrix
G = nx.Graph()
n = matrix.shape[0]
percent_done = 0
for i in range(n):
    for j in range(i, n):
        if matrix[i, j] > 0:
            # print('add_edge', i, j, matrix[i, j])   
            G.add_edge(i, j, weight=matrix[i, j])
        percent_done = (i * n + j) / (n * n) * 100  
        print(f"Progress: {percent_done:.2f}%", end="\r")

# Visualize the graph
pos = nx.spring_layout(G)
plt.figure(figsize=(10, 10))
nx.draw(G, pos, node_size=20, width=0.5)

# Save the graph as an image
plt.savefig("graph.png")
plt.show()
# The graph is saved as an image for visualization purposes.


            