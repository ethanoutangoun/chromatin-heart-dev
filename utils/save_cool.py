#!/usr/bin/env python3
import argparse
import os
import cooler
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description="Balance a .cool or .mcool::resolutions/<res> file and save as .npy"
    )
    parser.add_argument(
        "cool_path",
        help="Path to input .cool file or .mcool::/resolutions/<res>"
    )
    args = parser.parse_args()

    # Parse file path and output naming
    cool_path = args.cool_path
    base_name = os.path.basename(cool_path.split("::")[0])
    
    if "::" in cool_path:
        res_str = cool_path.split("::")[1].split("/")[-1]
        res_kb = int(res_str) // 1000
    else:
        clr = cooler.Cooler(cool_path)
        res_kb = clr.info['bin-size'] // 1000

    out_file = f"{base_name.replace('.mcool', '').replace('.cool', '')}_{res_kb}kb_balanced.npy"

    print(f"Loading {cool_path}...")
    clr = cooler.Cooler(cool_path)

    print("Extracting balanced matrix...")
    matrix = clr.matrix(balance=True)[:]
    matrix = np.nan_to_num(matrix)

    print(f"Saving to {out_file}...")
    np.save(out_file, matrix)

    print("Done.")

if __name__ == "__main__":
    main()