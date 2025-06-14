#!/usr/bin/env python3
"""
extract_binmap.py

Extracts a bin map from a `.cool` file and saves it as a BED-like tab-delimited file
with columns: chrom, start, end, bin_id. Useful for mapping matrix indices to genomic coordinates.

Usage:
    python extract_binmap.py path/to/input.cool --output bin_map.bed

Inputs:
    - input_cool (.cool): path to a Cooler contact matrix
    - output (optional): path to output BED-like bin map (default: <basename>_bin_map.bed)

Output:
    - A .bed file with 4 columns: chrom, start, end, bin_id

Dependencies:
    - cooler
"""

import os
import argparse
import cooler

def main():
    parser = argparse.ArgumentParser(
        description="Extract bin map (chromosome, start, end, bin_id) from a Cooler file."
    )
    parser.add_argument(
        "input_cool",
        help="Path to the input .cool file"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output bin map file (default: input filename + _bin_map.bed)"
    )
    args = parser.parse_args()

    # Load cooler file
    print(f"Loading Cooler file from {args.input_cool}...")
    c = cooler.Cooler(args.input_cool)

    # Extract bins DataFrame by fetching all rows
    print("Extracting bins table...")
    bins_df = c.bins()[:]  # RangeSelector1D supports slicing to return DataFrame
    # Reset index to bring bin_id into a column
    bins_df = bins_df.reset_index().rename(columns={"index": "bin_id"})

    # Determine output path
    if args.output:
        out_path = args.output
    else:
        base = os.path.basename(args.input_cool)
        name, _ = os.path.splitext(base)
        out_path = f"{name}_bin_map.bed"
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    # Write out as BED-like file: chrom, start, end, bin_id
    print(f"Writing bin map to {out_path}...")
    bins_df.to_csv(
        out_path,
        sep="\t",
        header=False,
        index=False,
        columns=["chrom", "start", "end", "bin_id"]
    )

    print("Done.")

if __name__ == "__main__":
    main()
