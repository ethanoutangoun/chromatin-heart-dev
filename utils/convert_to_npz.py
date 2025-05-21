#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
import sys

def main():
    p = argparse.ArgumentParser(description="Convert .npy → .npz and filter zero-sum rows")
    p.add_argument("input", type=Path,
                   help="Path to input .npy file")
    p.add_argument("--tol", type=float, default=1e-6,
                   help="Minimum row-sum to consider a row “nonzero” (default: 1e-6)")
    args = p.parse_args()

    inp = args.input
    if not inp.exists():
        sys.exit(f"Error: input file {inp!r} not found.")
    if inp.suffix.lower() != ".npy":
        sys.exit("Error: input file must have a .npy extension.")

    out = inp.with_suffix(".npz")

    # Load
    mat = np.load(str(inp))
    if mat.ndim != 2:
        sys.exit(f"Error: expected a 2D matrix; got array with shape {mat.shape}.")

    # Compute variable rows
    row_sums = mat.sum(axis=1)
    var = np.where(row_sums > args.tol)[0]

    # Save compressed
    np.savez_compressed(str(out), mat=mat, var=var)
    print(f"Wrote {out.name!r} (mat shape={mat.shape}, var length={len(var)})")

if __name__ == "__main__":
    main()
