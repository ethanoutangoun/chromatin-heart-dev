#!/usr/bin/env python3

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

    sparsity = np.count_nonzero(matrix) / matrix.size
    print(f"{args.label}: {sparsity:.2%} sparsity")

if __name__ == "__main__":
    main()
