import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



# GLOBALS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from tqdm import tqdm
from time import time

# TODO: Move functions to separate modules
import functions as f

import core.background_model as background_model
import core.clique_finding as cf
import core.stats 


TTN_BIN = 4275
BIN_MAP_PATH = 'mappings/bin_map_human_100000.bed'
GTF_PATH = 'mappings/gencode.v38.annotation.gtf'
GENE_BIN_PATH = 'mappings/gene_bins.txt'
NON_GENE_BIN_PATH = 'mappings/non_gene_bins.txt'


gene_bins = []
with open('mappings/gene_bins.txt', 'r') as file:
    for line in file:
        gene_bins.append(line.strip())
gene_bins = [int(x) for x in gene_bins]


non_gene_bins = []
with open('mappings/non_gene_bins.txt', 'r') as file:
    for line in file:
        non_gene_bins.append(line.strip())
non_gene_bins = [int(x) for x in non_gene_bins]


tf_bins = []
with open('mappings/tf_bins.txt', 'r') as file:
    for line in file:
        tf_bins.append(line.strip())
tf_bins = [int(x) for x in tf_bins]


contact_matrix_zero = np.load('data/hic/wildtype_100kb_zeroed.npy')
# contact_matrix_zero = np.load('data/hic/wt_100kb_balanced_zeroed.npy')



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

neighbors, cdfs = build_walk_index(contact_matrix_zero)

clique_sizes = [5, 10]
num_samples = 5000
NUM_MOLECULES = 10000 
ALPHA = 0.5
all_bins = [i for i in range(contact_matrix_zero.shape[0])]

records = []


for i in tqdm(range(num_samples)):
    tf_clique = cf.random_walk(contact_matrix_zero, np.random.choice(tf_bins), max(clique_sizes), num_molecules=NUM_MOLECULES, alpha=ALPHA, cdfs=cdfs, neighbors=neighbors)
    non_gene_clique = cf.random_walk(contact_matrix_zero, np.random.choice(non_gene_bins), max(clique_sizes), num_molecules=NUM_MOLECULES, alpha=ALPHA, cdfs=cdfs, neighbors=neighbors)
    gene_clique = cf.random_walk(contact_matrix_zero, np.random.choice(gene_bins), max(clique_sizes), num_molecules=NUM_MOLECULES, alpha=ALPHA, cdfs=cdfs, neighbors=neighbors)
    generic_clique = cf.random_walk(contact_matrix_zero, np.random.choice(all_bins), max(clique_sizes), num_molecules=NUM_MOLECULES, alpha=ALPHA, cdfs=cdfs, neighbors=neighbors)

    for k in clique_sizes:
        tf_strength = core.stats.calculate_avg_interaction_strength(contact_matrix_zero, tf_clique[:k])
        non_gene_strength = core.stats.calculate_avg_interaction_strength(contact_matrix_zero, non_gene_clique[:k])
        gene_strength = core.stats.calculate_avg_interaction_strength(contact_matrix_zero, gene_clique[:k])
        generic_strength = core.stats.calculate_avg_interaction_strength(contact_matrix_zero, generic_clique[:k])

        records.append({"clique_size": k, "model_type": "tf", "strength": tf_strength})
        records.append({"clique_size": k, "model_type": "non_gene", "strength": non_gene_strength})
        records.append({"clique_size": k, "model_type": "gene", "strength": gene_strength})
        records.append({"clique_size": k, "model_type": "all", "strength": generic_strength})

df = pd.DataFrame(records)

# Save to CSV for future reuse
df.to_csv("background_strength_distributions_4.csv", index=False)




# clique_sizes = [5, 10]
# num_samples = 5000
# all_bins = [i for i in range(contact_matrix_zero.shape[0])]

# records = []

# for i in tqdm(range(num_samples)):
#     tf_clique = cf.find_greedy_clique(contact_matrix_zero, max(clique_sizes), np.random.choice(tf_bins))
#     non_gene_clique = cf.find_greedy_clique(contact_matrix_zero, max(clique_sizes), np.random.choice(non_gene_bins))
#     gene_clique = cf.find_greedy_clique(contact_matrix_zero, max(clique_sizes), np.random.choice(gene_bins))
#     generic_clique = cf.find_greedy_clique(contact_matrix_zero, max(clique_sizes), np.random.choice(all_bins))  

#     for k in clique_sizes:
#         tf_strength = core.stats.calculate_avg_interaction_strength(contact_matrix_zero, tf_clique[:k])
#         non_gene_strength = core.stats.calculate_avg_interaction_strength(contact_matrix_zero, non_gene_clique[:k])
#         gene_strength = core.stats.calculate_avg_interaction_strength(contact_matrix_zero, gene_clique[:k])
#         generic_strength = core.stats.calculate_avg_interaction_strength(contact_matrix_zero, generic_clique[:k])

#         records.append({"clique_size": k, "model_type": "tf", "strength": tf_strength})
#         records.append({"clique_size": k, "model_type": "non_gene", "strength": non_gene_strength})
#         records.append({"clique_size": k, "model_type": "gene", "strength": gene_strength})
#         records.append({"clique_size": k, "model_type": "all", "strength": generic_strength})

# df = pd.DataFrame(records)

# # Save to CSV for future reuse
# df.to_csv("background_strength_distributions_3.csv", index=False)
