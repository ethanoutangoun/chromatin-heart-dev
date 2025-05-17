#!/usr/bin/env python3
import numpy as np
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# CLI
parser = argparse.ArgumentParser(description="Print number of non-zero entries in Hi-C matrices.")
parser.add_argument("-m", required=True, help="Path to the .npy matrix")
parser.add_argument("--label1", default="Matrix 1", help="Label for matrix1")
args = parser.parse_args()

# Load matrices
logging.info(f"Loading {args.m}...")
# Load matrix
mat1 = np.load(args.m)


# Count non-zero interactions
count1 = np.count_nonzero(mat1)

# Print results
print(f"{args.label1}: {count1:,} non-zero entries")
