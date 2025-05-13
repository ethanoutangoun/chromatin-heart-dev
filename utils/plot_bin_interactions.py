#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    parser = argparse.ArgumentParser(description='Plot interaction profile of a single bin across the genome.')
    parser.add_argument('--matrix', type=str, required=True, help='Path to .npy contact matrix')
    parser.add_argument('--bin', type=int, required=True, help='Bin index to plot interactions for (e.g., 4275)')
    parser.add_argument('--out', type=str, default='bin_interactions.png', help='Output plot filename')
    parser.add_argument('--title', type=str, default='Bin Interaction Profile', help='Plot title')
    args = parser.parse_args()

    logging.info(f'Loading matrix from {args.matrix}')
    mat = np.load(args.matrix)

    if args.bin < 0 or args.bin >= mat.shape[0]:
        logging.error(f'Invalid bin index {args.bin}. Matrix has shape {mat.shape}.')
        return

    logging.info(f'Extracting interaction profile for bin {args.bin}')
    profile = mat[args.bin, :] + mat[:, args.bin]  # symmetrical matrix: row + col gives total interaction

    logging.info(f'Plotting interaction profile to {args.out}')
    plt.figure(figsize=(12, 4))
    plt.plot(profile, color='black', linewidth=0.8)
    plt.title(f'{args.title} (Bin {args.bin})')
    plt.xlabel('Bin Index')
    plt.ylabel('Interaction Strength')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    logging.info('Done.')

if __name__ == '__main__':
    main()
