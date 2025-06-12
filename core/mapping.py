# mapping.py

# contains mapping functions for genomic data analysis

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import os
import bisect


#########################
# BIN MAPPING FUNCTIONS #
#########################

def load_bin_map(file_path):
    bin_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 4:
                chromosome, _, _, bin_id = parts
                bin_dict[int(bin_id)] = chromosome
    return bin_dict


def load_bin_map_loc(file_path):
    bin_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 4:
                chromosome, start, end, bin_id = parts
                start, end, bin_id = int(start), int(end), int(bin_id)

                # Store bin_id for the entire range
                if chromosome not in bin_dict:
                    bin_dict[chromosome] = []
                bin_dict[chromosome].append((start, end, bin_id))

    # Sort ranges for faster querying
    for chrom in bin_dict:
        bin_dict[chrom].sort()

    return bin_dict

def find_bin(chromosome, position, bin_dict):
    if chromosome not in bin_dict:
        return None  # Chromosome not found
    
    bins = bin_dict[chromosome]
    idx = bisect.bisect_left(bins, (position,))  # Find closest start position

    if idx > 0 and bins[idx - 1][0] <= position <= bins[idx - 1][1]:
        return bins[idx - 1][2]  # Return bin_id

    return None  # Position not found in any range


# Find bins that contain genes based on a GTF file
def find_genic_bins(bin_map, gtf_file_path):   
    gtf_cols = ["chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]
    gtf_df = pd.read_csv(gtf_file_path, sep="\t", comment="#", header=None, names=gtf_cols)
    genes_df = gtf_df[gtf_df["feature"] == "gene"]

    bins_on_genes = set()

    for _, row in genes_df.iterrows():
        gene_start = row["start"]
        gene_chrom = row["chrom"]

        bin = find_bin(gene_chrom, gene_start, bin_map)    
        if bin is not None:
            bins_on_genes.add(bin)

    return list(bins_on_genes)


def find_intergenic_bins(bin_map, gtf_file_path):
    gtf_cols = ["chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]
    gtf_df = pd.read_csv(gtf_file_path, sep="\t", comment="#", header=None, names=gtf_cols)
    genes_df = gtf_df[gtf_df["feature"] == "gene"]

    genic_bins = set()

    for _, row in genes_df.iterrows():
        gene_start = row["start"]
        gene_chrom = row["chrom"]
        bin = find_bin(gene_chrom, gene_start, bin_map)
        if bin is not None:
            genic_bins.add(bin)

    # Collect all bins from bin_map
    all_bins = set()
    for chrom_bins in bin_map.values():
        all_bins.update(chrom_bins.values())

    intergenic_bins = all_bins - genic_bins
    return list(intergenic_bins)




#################
# GENE MAPPINGS #
#################

def find_gene_from_bin(bin_id, node_bed_path, gtf_file_path):
    def parse_gene_name(attribute):
        match = re.search(r'gene_name "([^"]+)"', attribute)
        return match.group(1) if match else None

    nodes_df = pd.read_csv(node_bed_path, sep="\t", header=None,
                           names=["chrom", "start", "end", "bin"])
    node_intervals = nodes_df[nodes_df["bin"] == bin_id]
    if node_intervals.empty:
        return []

    gtf_cols = ["chrom", "source", "feature", "start", "end",
                "score", "strand", "frame", "attribute"]
    gtf_df = pd.read_csv(gtf_file_path, sep="\t", comment="#",
                         header=None, names=gtf_cols)
    
    genes_df = gtf_df.loc[gtf_df["feature"] == "gene"].copy()
    genes_df["gene_name"] = genes_df["attribute"].apply(parse_gene_name)

    # collect overlaps
    gene_abbrevs = set()
    for _, node in node_intervals.iterrows():
        overlaps = genes_df[
            (genes_df["start"] < node["end"]) &
            (genes_df["end"] > node["start"])
        ]
        gene_abbrevs.update(overlaps["gene_name"].dropna())

    return list(gene_abbrevs)


def get_genes_from_bins(bin_ids, bin_map_path, gtf_file_path):
    GENE_SET = set()
    for bin in tqdm(bin_ids, desc="Finding genes from bins"):
        genes = find_gene_from_bin(bin, bin_map_path, gtf_file_path)
        if genes:
            GENE_SET.update(genes)

    return list(GENE_SET)
        

def find_ttn_bin(gtf_file_path, node_bed_path):
    gtf_cols = ["chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]
    gtf_df = pd.read_csv(gtf_file_path, sep="\t", comment="#", header=None, names=gtf_cols)
    genes_df = gtf_df[gtf_df["feature"] == "gene"]
    ttn_gene = genes_df[genes_df["attribute"].str.contains('gene_name "TTN"')]
    
    if ttn_gene.empty:
        print("TTN gene not found in the GTF file.")
        return None
    
    ttn_chrom = ttn_gene["chrom"].iloc[0]
    ttn_start = int(ttn_gene["start"].iloc[0])
    ttn_end = int(ttn_gene["end"].iloc[0])
    
    nodes_df = pd.read_csv(node_bed_path, sep="\t", header=None, names=["chrom", "start", "end", "bin"])
    overlapping_bins = nodes_df[(nodes_df["chrom"] == ttn_chrom) &
                                (nodes_df["start"] < ttn_end) &
                                (nodes_df["end"] > ttn_start)]
    
    if overlapping_bins.empty:
        print("No bins overlap with the TTN gene.")
        return None
    
    return overlapping_bins["bin"].tolist()



def get_ttn_locus(gtf_file_path):
    gtf_cols = [
        "chrom", "source", "feature", "start", "end",
        "score", "strand", "frame", "attribute"
    ]
    gtf = pd.read_csv(
        gtf_file_path,
        sep="\t",
        comment="#",
        header=None,
        names=gtf_cols,
        usecols=["chrom", "feature", "start", "end", "attribute"]
    )

    # Keep only gene features
    genes = gtf[gtf["feature"] == "gene"]

    # Find TTN by gene_name
    is_ttn = genes["attribute"].str.contains(r'gene_name "TTN"')
    if not is_ttn.any():
        print("TTN gene not found in the GTF file.")
        return None

    # If multiple entries, take the first
    rec = genes[is_ttn].iloc[0]
    return {
        "chrom": rec["chrom"],
        "start": int(rec["start"]),
        "end":   int(rec["end"])
    }



