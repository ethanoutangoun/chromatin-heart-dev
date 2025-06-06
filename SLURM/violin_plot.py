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






clique_sizes = [5, 10]
num_samples = 10
all_bins = [i for i in range(contact_matrix_zero.shape[0])]

records = []

for i in tqdm(range(num_samples)):
    tf_clique = cf.find_greedy_clique(contact_matrix_zero, max(clique_sizes), np.random.choice(tf_bins))
    non_gene_clique = cf.find_greedy_clique(contact_matrix_zero, max(clique_sizes), np.random.choice(non_gene_bins))
    gene_clique = cf.find_greedy_clique(contact_matrix_zero, max(clique_sizes), np.random.choice(gene_bins))
    generic_clique = cf.find_greedy_clique(contact_matrix_zero, max(clique_sizes), np.random.choice(all_bins))  

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
df.to_csv("background_strength_distributions_3.csv", index=False)



plt.figure(figsize=(10, 6)) 
# Optional: Plot

import seaborn as sns
sns.violinplot(x="clique_size", y="strength", hue="model_type", data=df, inner="quartile", scale="width")
plt.xlabel("Clique Size")
plt.ylabel("Avg. Interaction Strength")
plt.title("Strength Distribution by Background Type")
plt.tight_layout()
plt.show()