## Hi-C Matrix Utility Scripts

This repository contains a set of command-line tools for manipulating and analyzing Hi-C contact matrices in `.npy`, `.npz`, and `.cool` formats. These scripts support format conversions, filtering operations (e.g., zeroing cis contacts or excluding specific chromosomes), matrix sparsity checks, and interaction visualizations. Most scripts are designed for use in a modular pipeline and assume 2D square matrices.

---

### `check_nonzero_contacts.py`

Counts and prints the raw number of non-zero entries in a `.npy` Hi-C contact matrix.

```bash
python check_nonzero_contacts.py -m path/to/matrix.npy --label1 WT
```

### `check_sparsity.py`

Prints the percentage of non-zero entries in a `.npy` Hi-C contact matrix to assess sparsity.

```bash
python check_sparsity.py -m path/to/matrix.npy --label MyMatrix
```

### `compare_bin_interactions.py`

Compares the interaction profile of a specific bin across wildtype and knockout Hi-C matrices and saves a line plot.

```bash
python compare_bin_interactions.py --wt wt_matrix.npy --ko ko_matrix.npy --bin 123 --output bin123_comparison.png
```

### `convert_to_npz.py`

Converts a `.npy` Hi-C matrix to a compressed `.npz` format, filtering out rows with near-zero sums. Useful for some external tooling.

```bash
python convert_to_npz.py path/to/matrix.npy --tol 1e-6
```

### `cooler_to_numpy.py`

Extracts a contact matrix from a `.cool` file and saves it as a `.npy` file, with optional balancing.

```bash
python cooler_to_numpy.py path/to/matrix.cool -o output.npy --no-balance
```

### `extract_binmap.py`

Extracts a bin map (chrom, start, end, bin ID) from a `.cool` file and saves it in BED-like format.

```bash
python extract_binmap.py path/to/matrix.cool --output bin_map.bed
```

### `plot_bin_interactions.py`

Essentially a virtual 4C plot. Plots the genome-wide interaction profile of a single bin from a `.npy` Hi-C matrix, summing both row and column to reflect total interactions. 

```bash
python plot_bin_interaction_profile.py --matrix path/to/matrix.npy --bin 4275 --out bin4275_plot.png --title "My Bin"
```

### `remove_chrM_from_cool.py`

Sometimes, processing a Hi-C matrix will generate reads on chromosome M. To maintain consistency with other downstream analysis, this step is highly recommended. Removes mitochondrial chromosome (`chrM`) from a `.cool` or `.mcool::/resolutions/RES` Hi-C file and saves a filtered version containing only autosomes and sex chromosomes (chr1–22, X, Y).

```bash
python remove_chrM_from_cool.py input.cool output_prefix
```

### `unzip_dir.py`

Unzips all `.gz` files in a specified directory and deletes the original compressed files. Useful for processing bulk zipped data from ENCODE like transcription factors.

```bash
python unzip_dir.py path/to/directory
```

### `zero_chrom_interactions.py`

Zeros out all contacts in a `.npy` Hi-C matrix that involve any user-specified chromosomes (e.g., `chrY`, `chrM`), using a bin map to identify associated bins.

```bash
python zero_chrom_interactions.py path/to/matrix.npy \\
    --bin-map path/to/bin_map.bed \\
    --exclude chrY chrM \\
    --output-dir output_directory
```

### `zero_cis_contacts.py`

Zeros out all **intra-chromosomal** (cis) contacts in a Hi-C `.npy` matrix using a bin map, leaving only **interchromosomal** (trans) interactions.

```bash
python zero_cis_contacts.py path/to/matrix.npy --bin-map path/to/bin_map.bed --output-dir output_dir
```