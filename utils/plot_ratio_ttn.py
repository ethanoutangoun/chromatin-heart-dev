#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging

def main():
    parser = argparse.ArgumentParser(description="Plot log2(KO / WT) for TTN region from two Hi-C matrices.")
    parser.add_argument("wt_matrix", help="Path to WT .npy matrix")
    parser.add_argument("ko_matrix", help="Path to KO .npy matrix")
    parser.add_argument("--start_bin", type=int, default=4225, help="Start bin for TTN region (default: 4225)")
    parser.add_argument("--end_bin", type=int, default=4325, help="End bin for TTN region (default: 4325)")
    parser.add_argument("--output", default="ttn_log2ratio.png", help="Output filename for plot")
    parser.add_argument("--log", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=getattr(logging, args.log.upper())
    )

    logging.info("Loading WT matrix from %s", args.wt_matrix)
    wt = np.load(args.wt_matrix)

    logging.info("Loading KO matrix from %s", args.ko_matrix)
    ko = np.load(args.ko_matrix)

    logging.info("Slicing region bins %d to %d", args.start_bin, args.end_bin)
    wt_sub = wt[args.start_bin:args.end_bin, args.start_bin:args.end_bin]
    ko_sub = ko[args.start_bin:args.end_bin, args.start_bin:args.end_bin]

    logging.debug("WT submatrix shape: %s", wt_sub.shape)
    logging.debug("KO submatrix shape: %s", ko_sub.shape)

    logging.info("Computing log2(KO / WT) ratio")
    epsilon = 1e-6
    mask = np.isnan(wt_sub) | np.isnan(ko_sub)
    log2ratio = np.full_like(wt_sub, np.nan)
    log2ratio[~mask] = np.log2((ko_sub[~mask] + epsilon) / (wt_sub[~mask] + epsilon))

    logging.info("Plotting heatmap")
    plt.figure(figsize=(8, 6))
    plt.imshow(log2ratio, cmap="coolwarm", vmin=-2, vmax=2)
    plt.colorbar(label="log2(KO / WT)")
    plt.title("TTN Region log2(KO / WT)")
    plt.xlabel("Bin")
    plt.ylabel("Bin")
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.close()

    logging.info("Saved plot to %s", args.output)

if __name__ == "__main__":
    main()
