#!/usr/bin/env python3
"""
cooler_to_numpy.py

Extracts a contact matrix from a `.cool` file and saves it as a `.npy` file.
Supports both balanced and raw matrices depending on the `--no-balance` flag.

Usage:
    python cooler_to_numpy.py path/to/matrix.cool \
        --no-balance \
        --output path/to/output.npy

Inputs:
    - cool_file (.cool): input Cooler file
    - --no-balance (flag): disables matrix balancing (default is balanced)
    - --output (optional): output path for .npy file (default: data/hic/<basename>_balanced.npy)

Output:
    - A 2D numpy array saved as a `.npy` file

Dependencies:
    - cooler
    - numpy
"""

import argparse
import os
import cooler
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description="Extract a contact matrix from a .cool file and save as .npy"
    )
    parser.add_argument(
        "cool_file",
        help="Path to input .cool file (e.g. matrix_100kb.cool)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output .npy file path (default: data/hic/<basename>_balanced.npy or _raw.npy)"
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Disable balancing (use raw counts)"
    )
    args = parser.parse_args()

    clr_path = args.cool_file
    balance_flag = not args.no_balance

    # Derive output path
    if args.output:
        out_path = args.output
    else:
        os.makedirs("data/hic", exist_ok=True)
        base = os.path.splitext(os.path.basename(clr_path))[0]
        suffix = "raw" if args.no_balance else "balanced"
        out_path = os.path.join("data/hic", f"{base}_{suffix}.npy")

    print(f"Loading {clr_path}...")
    clr = cooler.Cooler(clr_path)

    print(f"Extracting {'raw' if args.no_balance else 'balanced'} matrix...")
    matrix = clr.matrix(balance=balance_flag)[:]
    matrix = np.nan_to_num(matrix)

    print(f"Saving to {out_path}...")
    np.save(out_path, matrix)
    print("Done.")

if __name__ == "__main__":
    main()
