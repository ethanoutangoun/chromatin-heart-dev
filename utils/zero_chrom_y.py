#!/usr/bin/env python3
"""
Zero out all contacts involving chrY in a Hi-C contact matrix.

Usage:
    zero_chrY_contacts.py <input_matrix.npy> --bin-map <bin_map.tsv> [--output-dir <out_dir>]

Arguments:
    input_matrix.npy    Path to the numpy .npy contact matrix.
    --bin-map           Path to the tab-delimited bin map file with columns:
                        chrom, start, end, bin_id
    --output-dir        Directory to write the output matrix (default: data/hic)
"""

import os
import argparse
import numpy as np
from tqdm import tqdm

def load_bin_map(file_path):
    """
    Load a bin map file mapping integer bin IDs to chromosome names.
    Expects four tab-separated columns per line: chrom, start, end, bin_id.
    """
    bin_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            # skip empty/comment lines
            if not line.strip() or line.startswith('#'):
                continue
            chrom, start, end, bin_id = line.strip().split('\t')
            bin_dict[int(bin_id)] = chrom
    return bin_dict

def zero_chrY_interactions(mat: np.ndarray, bin_map: dict) -> np.ndarray:
    """
    Zero out any interaction where either bin is on chrY.
    """
    # find all bin indices that map to 'chrY'
    chrY_bins = [i for i, chrom in bin_map.items() if chrom.lower() == 'chry']
    n = mat.shape[0]
    # sanity check
    if any(i >= n or i < 0 for i in chrY_bins):
        raise IndexError("Found bin_id in bin_map outside matrix dimensions")
    # zero rows and columns for each chrY bin
    for i in tqdm(chrY_bins, desc="Zeroing chrY rows/cols"):
        mat[i, :] = 0
        mat[:, i] = 0
    return mat

def main():
    parser = argparse.ArgumentParser(description="Zero out all chrY interactions in a Hi-C contact matrix")
    parser.add_argument("input_matrix", help="Path to input .npy contact matrix")
    parser.add_argument("--bin-map", required=True, help="Path to bin map file (chrom\tstart\tend\tbin_id)")
    parser.add_argument("--output-dir", default="data/hic", help="Output directory (default: data/hic)")
    args = parser.parse_args()

    # load contact matrix
    print(f"Loading contact matrix from {args.input_matrix}...")
    matrix = np.load(args.input_matrix)

    # load bin map
    print(f"Loading bin map from {args.bin_map}...")
    bin_map = load_bin_map(args.bin_map)

    # zero out chrY interactions
    print("Zeroing all interactions involving chrY...")
    matrix = zero_chrY_interactions(matrix, bin_map)

    # prepare output path
    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.basename(args.input_matrix)
    name, ext = os.path.splitext(base)
    out_name = f"{name}_no_chrY{ext}"
    out_path = os.path.join(args.output_dir, out_name)

    # save zeroed matrix
    print(f"Saving modified matrix to {out_path}...")
    np.save(out_path, matrix)
    print("Done.")

if __name__ == "__main__":
    main()
