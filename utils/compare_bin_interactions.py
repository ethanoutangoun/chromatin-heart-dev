
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Compare interaction strengths of a specific bin between two Hi-C matrices.")
    parser.add_argument('--wt', required=True, help='Path to wildtype .npy matrix')
    parser.add_argument('--ko', required=True, help='Path to knockout .npy matrix')
    parser.add_argument('--bin', type=int, required=True, help='Bin index to extract interaction profile for')
    parser.add_argument('--output', default='compare_bin_interactions.png', help='Output plot filename')
    args = parser.parse_args()

    logging.info(f"Loading WT matrix from {args.wt}")
    wt_matrix = np.load(args.wt)
    logging.info(f"Loading KO matrix from {args.ko}")
    ko_matrix = np.load(args.ko)

    wt_profile = wt_matrix[args.bin, :]
    ko_profile = ko_matrix[args.bin, :]

    max_y = max(wt_profile.max(), ko_profile.max()) * 1.1

    logging.info("Plotting comparison")
    plt.figure(figsize=(16, 6))
    plt.plot(wt_profile, label='Wildtype', alpha=0.7)
    plt.plot(ko_profile, label='Knockout', alpha=0.7)
    plt.title(f'Bin Interaction Profile Comparison (Bin {args.bin})')
    plt.xlabel('Bin Index')
    plt.ylabel('Interaction Strength')
    plt.ylim(0, max_y)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output)
    logging.info(f"Saved plot to {args.output}")
    plt.close()

if __name__ == '__main__':
    main()

