#!/usr/bin/env python3
import numpy as np
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# CLI
parser = argparse.ArgumentParser(description="Print number of non-zero entries in Hi-C matrices.")
parser.add_argument("--matrix1", required=True, help="Path to the first .npy matrix")
parser.add_argument("--matrix2", required=True, help="Path to the second .npy matrix")
parser.add_argument("--label1", default="Matrix 1", help="Label for matrix1")
parser.add_argument("--label2", default="Matrix 2", help="Label for matrix2")
args = parser.parse_args()

# Load matrices
logging.info(f"Loading {args.matrix1}")
mat1 = np.load(args.matrix1)
logging.info(f"Loading {args.matrix2}")
mat2 = np.load(args.matrix2)

# Count non-zero interactions
count1 = np.count_nonzero(mat1)
count2 = np.count_nonzero(mat2)

# Print results
print(f"{args.label1}: {count1:,} non-zero entries")
print(f"{args.label2}: {count2:,} non-zero entries")
