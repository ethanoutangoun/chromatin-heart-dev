import pandas as pd
import re

def parse_gene_name(attribute):
    match = re.search(r'gene_name "([^"]+)"', attribute)
    return match.group(1) if match else None

def get_gene_abbreviations_in_node(chromosome, bin_id, node_bed_path, gtf_file_path):
    nodes_df = pd.read_csv(node_bed_path, sep="\t", header=None, names=["chrom", "start", "end", "bin"])
    node_intervals = nodes_df[(nodes_df["chrom"] == chromosome) & (nodes_df["bin"] == bin_id)]
    print('node_intervals:', node_intervals)    

    if node_intervals.empty:
        print(f"No intervals found for chromosome {chromosome} and bin {bin_id}.")
        return []
    
    gtf_cols = ["chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]
    gtf_df = pd.read_csv(gtf_file_path, sep="\t", comment="#", header=None, names=gtf_cols)
    genes_df = gtf_df[gtf_df["feature"] == "gene"]
    genes_df = genes_df[genes_df["chrom"] == str(chromosome)]
    genes_df["gene_name"] = genes_df["attribute"].apply(parse_gene_name)
    
    gene_abbrevs = set()
    for _, node in node_intervals.iterrows():
        overlaps = genes_df[(genes_df["start"] < node["end"]) & (genes_df["end"] > node["start"])]
        gene_abbrevs.update(overlaps["gene_name"].dropna().tolist())
    
    return list(gene_abbrevs)

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

ttn_bins = find_ttn_bin("data/gencode.v38.annotation.gtf", "data/bin_map_human_100000.bed")
print("ttn bins", ttn_bins)

genes = get_gene_abbreviations_in_node("chr2", 4275, "data/bin_map_human_100000.bed", "data/gencode.v38.annotation.gtf")
print('genes in node:', genes)