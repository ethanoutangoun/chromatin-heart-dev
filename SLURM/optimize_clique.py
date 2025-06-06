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


tf_bins = []
with open('mappings/tf_bins.txt', 'r') as file:
    for line in file:
        tf_bins.append(line.strip())
tf_bins = [int(x) for x in tf_bins]

TTN_BIN = 4275


# contact_matrix_zero = np.load('data/hic/wildtype_100kb_zeroed.npy') #SLURM
# contact_matrix_zero = np.load('data/hic/wt_100kb_balanced_zeroed.npy')
contact_matrix_zero = np.load('data/hic/wt_100kb_balanced_zeroed_no_chrY.npy')
# contact_matrix_zero = f.generate_sample_matrix_bins(1000)

# contact_matrix_zero = np.load('data/hic/wt_1mb_zeroed.npy')



all_bins = [i for i in range(contact_matrix_zero.shape[0])]

NUM_SAMPLES = 10
MAX_SIZE = 5



core.optimizer.optimize_clique_size(contact_matrix_zero, MAX_SIZE, TTN_BIN, NUM_SAMPLES, background_bins=all_bins, label='no_y_greedy_optimize_all')
core.optimizer.optimize_clique_size(contact_matrix_zero, MAX_SIZE, TTN_BIN, NUM_SAMPLES, background_bins=tf_bins, label='no_y_greedy_optimize_tf')
core.optimizer.optimize_clique_size(contact_matrix_zero, MAX_SIZE, TTN_BIN, NUM_SAMPLES, background_bins=non_gene_bins, label='no_y_greedy_optimize_weak')
core.optimizer.optimize_clique_size(contact_matrix_zero, MAX_SIZE, TTN_BIN, NUM_SAMPLES, background_bins=gene_bins, label='no_y_greedy_optimize_strong')

# res = core.optimizer.optimize_diffusion_params_smart(contact_matrix_zero, TTN_BIN, (3,50), (0.05, 0.80), n_trials=10, timeout_minutes=300, log_csv='test_optimize.csv', background_bins={'gene': gene_bins, 'non_gene': non_gene_bins, 'all': all_bins})
# res = core.optimizer.optimize_diffusion_params_stochastic(contact_matrix_zero, TTN_BIN, (3, 50), (0.05, 0.80), n_trials=10, neighbors=neighbors, cdfs=cdfs, background_size=10000, num_walkers=5000, timeout_minutes=300)