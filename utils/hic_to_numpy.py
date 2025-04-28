# WORK IN PROGRESS, UNTESTED SCRIPT

#!/usr/bin/env python3
import os
import argparse
import subprocess

import cooler
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description="Convert a .hic file to .mcool and extract a balanced contact matrix as .npy"
    )
    parser.add_argument(
        "hic_file",
        help="Path to input .hic file (e.g. samples/inter_30.hic)"
    )
    parser.add_argument(
        "-m", "--mcool",
        help="Path for output .mcool file (defaults to same basename as hic_file)",
    )
    parser.add_argument(
        "-r", "--resolution",
        type=int,
        default=100000,
        help="Resolution in bp for matrix extraction (default: 100000)"
    )
    parser.add_argument(
        "-o", "--output-npy",
        help="Path for output .npy file (defaults to <mcool_basename>_<res>kb.npy)"
    )
    args = parser.parse_args()

    hic_path = args.hic_file
    base = os.path.splitext(os.path.basename(hic_path))[0]
    mcool_path = args.mcool or f"{base}.mcool"
    res = args.resolution

    if args.output_npy:
        npy_path = args.output_npy
    else:
        npy_path = f"{base}_{res//1000}kb.npy"

    print(f"Converting {hic_path} → {mcool_path} using hic2cool…")
    subprocess.run(["hic2cool", "convert", hic_path, mcool_path], check=True)
    print("MCOOL file generated successfully.")

    print(f"Loading balanced matrix at {res} bp resolution from {mcool_path}…")
    clr = cooler.Cooler(f"{mcool_path}::resolutions/{res}")
    matrix = clr.matrix(balance=True)[:]
    matrix = np.nan_to_num(matrix)

    print(f"Saving contact matrix to {npy_path}…")
    os.makedirs(os.path.dirname(npy_path) or ".", exist_ok=True)
    np.save(npy_path, matrix)
    print("Matrix saved successfully.")

if __name__ == "__main__":
    main()