import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import core.clique_finding as cf
import numpy as np

import functions as f
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

contact_matrix_zero = np.load('data/hic/wildtype_100kb_zeroed_no_chrY.npy')
# contact_matrix_zero = np.load('data/hic/wt_100kb_balanced_zeroed_no_chrY.npy')
# contact_matrix_zero = f.generate_sample_matrix_bins(100)
print('Contact matrix loaded, computing clique...')


clique, visits = cf.analytical_diffusion_clique(contact_matrix_zero, TTN_BIN, 100, 0.5)

# write the results to a file
with open('wt_ttn_diffusion_noY_alpha_50_top_100_bins.txt', 'w') as file:
    for bin in clique:
        file.write(f"{bin}\n")

print(f'Clique: {clique}')