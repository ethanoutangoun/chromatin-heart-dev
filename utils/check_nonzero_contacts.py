#!/usr/bin/env python3
"""
check_nonzero_contacts.py

Counts and prints the total number of non-zero entries in a Hi-C contact matrix.
Useful for quickly assessing matrix sparsity or data density.

Usage:
    python check_nonzero_contacts.py -m path/to/matrix.npy

Inputs:
    - matrix (.npy): path to a Hi-C contact matrix

Output:
    - Printed number of non-zero entries (e.g., "153,492 non-zero entries")

Dependencies:
    - numpy
"""

import numpy as np
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# CLI
parser = argparse.ArgumentParser(description="Print number of non-zero entries in Hi-C matrices.")
parser.add_argument("-m", required=True, help="Path to the .npy matrix")
args = parser.parse_args()

# Load matrices
logging.info(f"Loading {args.m}...")
# Load matrix
mat1 = np.load(args.m)


# Count non-zero interactions
count1 = np.count_nonzero(mat1)

# Print results
print(f"{count1:,} non-zero entries")
