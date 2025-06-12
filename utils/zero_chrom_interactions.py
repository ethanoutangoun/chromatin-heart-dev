#!/usr/bin/env python3
"""
zero_chrom_interactions.py

Zeros out all interactions in a Hi-C contact matrix (.npy) that involve user-specified chromosomes,
using a bin map file to determine which bins belong to which chromosomes.

Usage:
    python zero_chrom_interactions.py path/to/matrix.npy \
        --bin-map path/to/bin_map.tsv \
        --exclude chrY chrM \
        --output-dir output_directory

Inputs:
    - input_matrix (.npy): square Hi-C contact matrix
    - bin_map (.tsv): tab-delimited file with columns [chrom, start, end, bin_id]
    - exclude: list of chromosome names to remove (e.g., chrY, chrM)
    - output_dir (optional): where to save the modified matrix (default: data/hic)

Output:
    - A new matrix with all interactions involving the excluded chromosomes set to 0

"""

import os
import argparse
import numpy as np
from tqdm import tqdm

def load_bin_map(file_path):
    """Load a bin map with columns: chrom, start, end, bin_id"""
    bin_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            chrom, start, end, bin_id = line.strip().split('\t')
            bin_dict[int(bin_id)] = chrom
    return bin_dict

def zero_chrom_interactions(mat: np.ndarray, bin_map: dict, exclude_chroms: list) -> np.ndarray:
    """Zero out rows and columns for bins that belong to excluded chromosomes."""
    exclude_chroms = set(chrom.lower() for chrom in exclude_chroms)
    excluded_bins = [i for i, chrom in bin_map.items() if chrom.lower() in exclude_chroms]
    
    n = mat.shape[0]
    if any(i >= n or i < 0 for i in excluded_bins):
        raise IndexError("Found bin_id in bin_map outside matrix dimensions")

    for i in tqdm(excluded_bins, desc="Zeroing excluded chrom rows/cols"):
        mat[i, :] = 0
        mat[:, i] = 0
    return mat

def main():
    parser = argparse.ArgumentParser(description="Zero out interactions involving specified chromosomes")
    parser.add_argument("input_matrix", help="Path to input .npy contact matrix")
    parser.add_argument("--bin-map", required=True, help="Path to bin map file (chrom\\tstart\\tend\\tbin_id)")
    parser.add_argument("--exclude", nargs="+", required=True, help="Chromosomes to exclude (e.g. chrY chrM)")
    parser.add_argument("--output-dir", default="data/hic", help="Output directory (default: data/hic)")
    args = parser.parse_args()

    print(f"Loading contact matrix from {args.input_matrix}...")
    matrix = np.load(args.input_matrix)

    print(f"Loading bin map from {args.bin_map}...")
    bin_map = load_bin_map(args.bin_map)

    print(f"Zeroing interactions involving: {', '.join(args.exclude)}...")
    matrix = zero_chrom_interactions(matrix, bin_map, args.exclude)

    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.basename(args.input_matrix)
    name, ext = os.path.splitext(base)
    suffix = "_no_" + "_".join(ch.lower() for ch in args.exclude)
    out_path = os.path.join(args.output_dir, f"{name}{suffix}{ext}")

    print(f"Saving modified matrix to {out_path}...")
    np.save(out_path, matrix)
    print("Done.")

if __name__ == "__main__":
    main()