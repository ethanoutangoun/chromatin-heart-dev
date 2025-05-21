#!/usr/bin/env python3
import cooler
from cooler import create_cooler
import numpy as np
import argparse
# ─── Parse Command-Line Arguments ───────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Filter out chrM from .cool or .mcool::/resolutions/RES file")
parser.add_argument("src", help="Path to .cool or .mcool::/resolutions/RES")
parser.add_argument("out_prefix", help="Prefix for output .cool file")
args = parser.parse_args()
SRC = args.src
OUT_COOL = f"{args.out_prefix}.cool"
SHARED = [str(i) for i in range(1, 23)] + ["X", "Y"]
# ─── Load Cooler File ───────────────────────────────────────────────────────────
c = cooler.Cooler(SRC)
# ─── Clean metadata ─────────────────────────────────────────────────────────────
meta = {}
for k, v in c.info.items():
    if isinstance(v, (np.integer,)):
        meta[k] = int(v)
    elif isinstance(v, (np.floating,)):
        meta[k] = float(v)
    else:
        meta[k] = v
# ─── Filter Bins and Pixels ─────────────────────────────────────────────────────
bins = c.bins()[:]
mask = bins["chrom"].isin(SHARED)
bins_f = bins.loc[mask].reset_index(drop=True)
old_ids = bins.index[mask]
old2new = {old: new for new, old in enumerate(old_ids)}
px = c.pixels()[:]
px_f = px[px.bin1_id.isin(old2new) & px.bin2_id.isin(old2new)].copy()
px_f["bin1_id"] = px_f["bin1_id"].map(old2new)
px_f["bin2_id"] = px_f["bin2_id"].map(old2new)
# ─── Create Filtered .cool ──────────────────────────────────────────────────────
create_cooler(OUT_COOL, bins=bins_f, pixels=px_f, metadata=meta)
print(f"Wrote filtered .cool without chrM: {OUT_COOL}")