def get_bins_on_gene(bin_map, gtf_file_path):   
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

def get_bins_not_on_gene(bin_map, gtf_file_path):   
    gene_bins = []
    with open('/Users/ethan/Desktop/chromatin-heart-dev/data/gene_bins.txt', 'r') as file:
        for line in file:
            gene_bins.append(line.strip())
    gene_bins = [int(x) for x in gene_bins]

    # All bins [0 to 30894]
    all_bins = set(range(0, 30894))
    bins_not_on_genes = all_bins - set(gene_bins)


    return list(bins_not_on_genes)

