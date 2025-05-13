#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    parser = argparse.ArgumentParser(description='Plot log2 ratio heatmap of KO vs WT Hi-C contact matrices.')
    parser.add_argument('--ko', type=str, required=True, help='Path to KO numpy matrix (.npy)')
    parser.add_argument('--wt', type=str, required=True, help='Path to WT numpy matrix (.npy)')
    parser.add_argument('--out', type=str, default='log2ratio_full_matrix.png', help='Output image filename')
    parser.add_argument('--vmin', type=float, default=-2.0, help='Minimum value for colormap')
    parser.add_argument('--vmax', type=float, default=2.0, help='Maximum value for colormap')
    args = parser.parse_args()

    logging.info(f'Loading KO matrix from {args.ko}')
    ko_mat = np.load(args.ko)
    logging.info(f'Loading WT matrix from {args.wt}')
    wt_mat = np.load(args.wt)

    logging.info('Computing log2 ratio...')
    mask = np.isnan(ko_mat) | np.isnan(wt_mat)
    log2ratio = np.full_like(ko_mat, np.nan)
    log2ratio[~mask] = np.log2((ko_mat[~mask] + 1) / (wt_mat[~mask] + 1))

    logging.info(f'Plotting heatmap to {args.out}')
    plt.figure(figsize=(10, 8))
    plt.imshow(log2ratio, cmap='coolwarm', vmin=args.vmin, vmax=args.vmax)
    plt.colorbar(label='log2(KO / WT)')
    plt.title('Log2 Ratio Heatmap (KO vs WT)')
    plt.xlabel('Bin Index')
    plt.ylabel('Bin Index')
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    logging.info('Done.')

if __name__ == '__main__':
    main()
