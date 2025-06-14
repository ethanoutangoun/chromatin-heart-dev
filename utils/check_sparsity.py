#!/usr/bin/env python3
"""
check_sparsity.py

Calculates and prints the proportion of non-zero entries in a Hi-C contact matrix,
providing a quick estimate of matrix sparsity.

Usage:
    python check_sparsity.py -m path/to/matrix.npy --label MyMatrix

Inputs:
    - matrix (.npy): path to a Hi-C contact matrix
    - label (optional): label to prefix the output (default: "Matrix")

Output:
    - Printed percentage of non-zero values (e.g., "MyMatrix density: 2.34%")

Dependencies:
    - numpy
"""

import numpy as np
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Print the percentage of non-zero entries (sparsity) in a Hi-C matrix."
    )
    parser.add_argument(
        "-m", "--matrix", required=True, help="Path to the .npy matrix file"
    )
    parser.add_argument(
        "--label", default="Matrix", help="Label for the matrix (optional)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    logging.info(f"Loading matrix from {args.matrix}...")
    matrix = np.load(args.matrix)

    density = np.count_nonzero(matrix) / matrix.size    
    print(f"{args.label} density: {density:.4%}")

if __name__ == "__main__":
    main()
