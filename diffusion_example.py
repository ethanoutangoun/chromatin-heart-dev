import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Example: Simulated contact matrix with bins (using a simple random graph for illustration)
n_bins = 100
contact_matrix = np.random.rand(n_bins, n_bins)  


G = nx.Graph()
for i in range(n_bins):
    for j in range(i + 1, n_bins):
        weight = contact_matrix[i, j] 
        if weight > 0.5:  
            G.add_edge(i, j, weight=weight)


# ttn_bins = range(30, 41)
ttn_bin = 30

# diffusing from TTN bin
# personalization = {bin_idx: 1 for bin_idx in ttn_bins} 
personalization = {ttn_bin: 1}  
pagerank = nx.pagerank(G, alpha=0.85, personalization=personalization, weight='weight')

# Plot the nodes' influence (pagerank) distribution
plt.hist(list(pagerank.values()), bins=50, edgecolor='black')
plt.title('Distribution of Diffusion (Personalized PageRank) from TTN Bin')
plt.xlabel('Personalized PageRank')
plt.ylabel('Frequency')
plt.show()

# Optionally, find the most affected bins (nodes with the highest PageRank scores)
top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top 10 most influenced nodes by TTN bins:", top_nodes)