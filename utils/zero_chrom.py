# file to zero out to take in a matrix, and return a matrix with the chrom column zeroed out

import numpy as np
from tqdm import tqdm

def load_bin_map(file_path):
    bin_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 4:
                chromosome, _, _, bin_id = parts
                bin_dict[int(bin_id)] = chromosome
    return bin_dict



contact_matrix = np.load('/Users/ethan/Desktop/chromatin-heart-dev/samples/contact_matrix_100kb_balanced.npy') 

bin_map = load_bin_map('/Users/ethan/Desktop/chromatin-heart-dev/data/bin_map_human_100000.bed')


# iterate through the matrix and zero out bins that are on the same chromosome
for i in tqdm(range(len(contact_matrix))):
    for j in range(len(contact_matrix)):
        chr_j = bin_map.get(j, None)
        chr_i = bin_map.get(i, None)
        if chr_j == chr_i:
            contact_matrix[i][j] = 0

np.save('/Users/ethan/Desktop/chromatin-heart-dev/samples/contact_matrix_100kb_balanced_zeroed.npy', contact_matrix)

       
