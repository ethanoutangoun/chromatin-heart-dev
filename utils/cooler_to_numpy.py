#!/usr/bin/env python3
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
