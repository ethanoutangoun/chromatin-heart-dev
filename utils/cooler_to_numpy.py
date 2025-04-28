#!/usr/bin/env python3
import argparse
import os

import cooler
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description="Extract a balanced contact matrix from a .mcool and save as .npy"
    )
    parser.add_argument(
        "mcool_file",
        help="Path to input .mcool file (e.g. 4DNFISWHXA16.mcool)"
    )
    parser.add_argument(
        "-r", "--resolution",
        type=int,
        default=100000,
        help="Resolution in bp (default: 100000)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output .npy file path (defaults to <basename>_<res>kb_balanced.npy)"
    )
    args = parser.parse_args()

    mcool_path = args.mcool_file
    res = args.resolution

    # derive default output name if not given
    if args.output:
        out_path = args.output
    else:
        base = os.path.splitext(os.path.basename(mcool_path))[0]
        out_path = f"{base}_{res//1000}kb_balanced.npy"

    print(f"Loading {mcool_path} at resolution {res}…")
    clr = cooler.Cooler(f"{mcool_path}::resolutions/{res}")

    print("Balancing matrix…")
    matrix = clr.matrix(balance=True)[:]
    matrix = np.nan_to_num(matrix)

    print(f"Saving to {out_path}…")
    np.save(out_path, matrix)

    print("Done.")

if __name__ == "__main__":
    main()