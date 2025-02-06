import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from collections import defaultdict

class ContactNetworkAnalyzer:
    """
    A class for analyzing and visualizing contact networks with multiple metrics
    and visualization options.
    """
    
    def __init__(self, contact_matrix, threshold=0):
        """
        Initialize the analyzer with a contact matrix.
        
        Args:
            contact_matrix (pd.DataFrame or np.ndarray): The contact matrix
            threshold (float): Minimum contact strength to consider
        """
        self.threshold = threshold
        self.contact_matrix = pd.DataFrame(contact_matrix) if isinstance(
            contact_matrix, np.ndarray) else contact_matrix
        self.graph = self._construct_graph()
        
    def _construct_graph(self):
        """Constructs the network graph from the contact matrix."""
        if self.contact_matrix.shape[0] != self.contact_matrix.shape[1]:
            raise ValueError("Contact matrix must be square.")
            
        graph = nx.Graph()
        num_nodes = self.contact_matrix.shape[0]
        graph.add_nodes_from(range(num_nodes))
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                weight = self.contact_matrix.iloc[i, j]
                if weight > self.threshold:
                    graph.add_edge(i, j, weight=weight)
                    
        return graph
    
    def calculate_network_metrics(self):
        """Calculate various network metrics."""
        metrics = {
            'degree_centrality': nx.degree_centrality(self.graph),
            'betweenness_centrality': nx.betweenness_centrality(self.graph),
            'closeness_centrality': nx.closeness_centrality(self.graph),
            'eigenvector_centrality': nx.eigenvector_centrality(self.graph, max_iter=1000),
            'clustering_coefficient': nx.clustering(self.graph),
            'average_shortest_path': nx.average_shortest_path_length(self.graph) if nx.is_connected(self.graph) else np.nan
        }
        
        # Add community detection
        communities = list(nx.community.greedy_modularity_communities(self.graph))
        metrics['communities'] = {node: i for i, comm in enumerate(communities) 
                                for node in comm}
        
        return metrics
    
    def visualize_network(self, metrics=None, layout='spring', figsize=(15, 15)):
        """
        Visualize the network with optional metric overlays.
        
        Args:
            metrics (dict): Dictionary of node metrics to visualize
            layout (str): Layout algorithm ('spring', 'circular', 'random')
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize)
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, k=1, iterations=50, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph)
        else:
            pos = nx.random_layout(self.graph, seed=42)
            
        # Node colors based on community if available
        if metrics and 'communities' in metrics:
            node_colors = [metrics['communities'][node] for node in self.graph.nodes()]
        else:
            node_colors = 'lightblue'
            
        # Node sizes based on degree centrality if available
        if metrics and 'degree_centrality' in metrics:
            node_sizes = [metrics['degree_centrality'][node] * 3000 for node in self.graph.nodes()]
        else:
            node_sizes = 500
            
        # Draw the network
        nx.draw_networkx_nodes(self.graph, pos, 
                             node_color=node_colors, 
                             node_size=node_sizes,
                             cmap=plt.cm.tab20,
                             edgecolors='black')
        
        # Draw edges with width proportional to weight
        edges = self.graph.edges(data=True)
        nx.draw_networkx_edges(self.graph, pos, 
                             edgelist=edges,
                             width=[d['weight'] * 0.5 for (_, _, d) in edges],
                             alpha=0.6)
        
        # Add labels
        nx.draw_networkx_labels(self.graph, pos, font_size=10)
        
        plt.title("Contact Network Analysis", fontsize=16, pad=20)
        plt.axis('off')
        return plt.gcf()
    
    def generate_report(self):
        """Generate a comprehensive network analysis report."""
        metrics = self.calculate_network_metrics()
        
        report = {
            'basic_stats': {
                'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges(),
                'density': nx.density(self.graph),
                'average_clustering': nx.average_clustering(self.graph),
                'number_of_communities': len(set(metrics['communities'].values()))
            },
            'centrality_stats': {
                'degree': metrics['degree_centrality'],
                'betweenness': metrics['betweenness_centrality'],
                'closeness': metrics['closeness_centrality'],
                'eigenvector': metrics['eigenvector_centrality']
            },
            'community_stats': defaultdict(list)
        }
        
        # Analyze communities
        for node, community in metrics['communities'].items():
            report['community_stats'][community].append(node)
            
        return report

def generate_sample_matrix():
    # Generate a larger sample dataset
    np.random.seed(42)
    n_nodes = 15
    dense_regions = [
        (0, 4),   # Community 1
        (5, 9),   # Community 2
        (10, 14)  # Community 3
    ]

    # Create base matrix with some noise
    contact_matrix = np.random.rand(n_nodes, n_nodes) * 2

    # Add stronger connections within communities
    for start, end in dense_regions:
        for i in range(start, end + 1):
            for j in range(start, end + 1):
                if i != j:
                    contact_matrix[i, j] = np.random.rand() * 8 + 2  # Strong connections

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


contact_matrix = generate_random_matrix()   

# Create analyzer instance
analyzer = ContactNetworkAnalyzer(contact_matrix, threshold=2)

# Generate and display results
metrics = analyzer.calculate_network_metrics()
report = analyzer.generate_report()

# Visualize with different layouts
layouts = ['spring', 'circular']
for layout in layouts:
    analyzer.visualize_network(metrics=metrics, layout=layout)
    plt.savefig(f"network_{layout}.png")
    plt.close()

# Print summary statistics
print("\nNetwork Summary:")
print("-" * 50)
print(f"Number of nodes: {report['basic_stats']['nodes']}")
print(f"Number of edges: {report['basic_stats']['edges']}")
print(f"Network density: {report['basic_stats']['density']:.3f}")
print(f"Average clustering coefficient: {report['basic_stats']['average_clustering']:.3f}")
print(f"Number of communities: {report['basic_stats']['number_of_communities']}")

# Analyze community structure
print("\nCommunity Structure:")
print("-" * 50)
for community, nodes in report['community_stats'].items():
    print(f"Community {community}: {len(nodes)} nodes - {nodes}")