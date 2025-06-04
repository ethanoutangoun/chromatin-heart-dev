import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np

import core.optimizer
import functions as f
from tqdm import tqdm
# GLOBALS


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

TTN_BIN = 4275
# TTN_BIN = 100

# contact_matrix_zero = np.load('data/hic/wildtype_100kb_zeroed.npy') #SLURM
contact_matrix_zero = np.load('data/hic/wt_100kb_balanced_zeroed.npy')
# contact_matrix_zero = np.load('data/hic/wt_100kb_balanced_zeroed_no_chrY.npy')
# contact_matrix_zero = f.generate_sample_matrix_bins(2000)

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

# res = core.optimizer.optimize_diffusion_params_smart(contact_matrix_zero, TTN_BIN, (3,50), (0.05, 0.80), n_trials=10, timeout_minutes=300)
res = core.optimizer.optimize_diffusion_params_stochastic(contact_matrix_zero, TTN_BIN, (3, 50), (0.05, 0.80), n_trials=10, neighbors=neighbors, cdfs=cdfs, background_size=10000, num_walkers=5000, timeout_minutes=300)