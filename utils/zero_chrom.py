#!/usr/bin/env python3
# To be ran from the command line on local machine

import os
import argparse
import numpy as np
from tqdm import tqdm

def load_bin_map(file_path):
    bin_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 4:
                chromosome, _, _, bin_id = parts
                bin_dict[int(bin_id)] = chromosome
    return bin_dict

def zero_same_chrom(contact_matrix, bin_map):
    n = contact_matrix.shape[0]
    for i in tqdm(range(n), desc="Zeroing intra-chromosomal contacts"):
        for j in range(n):
            if bin_map.get(i) == bin_map.get(j):
                contact_matrix[i, j] = 0
    return contact_matrix

def main():
    parser = argparse.ArgumentParser(
        description="Zero out all contacts within the same chromosome in a contact matrix."
    )
    parser.add_argument(
        "input_matrix",
        help="Path to the input .npy contact matrix"
    )
    parser.add_argument(
        "--bin-map",
        default="mappings/bin_map_human_100000.bed",
        help="Path to the bin map BED file (default: data/bin_map_human_100000.bed)"
    )
    parser.add_argument(
        "--output-dir",
        default="data/hic",
        help="Directory to save the zeroed matrix (default: data/hic)"
    )
    args = parser.parse_args()

    # load inputs
    print(f"Loading contact matrix from {args.input_matrix}...")
    matrix = np.load(args.input_matrix)
    print(f"Loading bin map from {args.bin_map}...")
    bin_map = load_bin_map(args.bin_map)

    # zero out same-chromosome entries
    matrix = zero_same_chrom(matrix, bin_map)

    # prepare output path
    base = os.path.basename(args.input_matrix)
    name, ext = os.path.splitext(base)
    output_name = f"{name}_zeroed{ext}"
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, output_name)

    # save result
    print(f"Saving zeroed matrix to {out_path}...")
    np.save(out_path, matrix)
    print("Done.")

if __name__ == "__main__":
    main()