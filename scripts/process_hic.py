# This script processes Hi-C data to generate .mcool files and .npy files for downstream analysis.

import subprocess
import cooler
import numpy as np

# Define paths
hic_path = "samples/inter_30.hic"  # Specify the input .hic file
mcool_path = "samples/inter_30.mcool"  # Specify the output .mcool file

print("Processing Hi-C data...")
# Convert .hic to .mcool using hic2cool
subprocess.run(["hic2cool", "convert", hic_path, mcool_path], check=True)

print("MCOOL file generated successfully!")
print("Processing contact matrix...")
# Load the .mcool file with cooler
resolution = 100000  # Adjust resolution as needed
c = cooler.Cooler(f"{mcool_path}::resolutions/{resolution}")

# Extract the contact matrix
matrix = c.matrix(balance=True)[:]
matrix = np.nan_to_num(matrix)

# Save as .npy
np.save("samples/contact_matrix.npy", matrix)
print("Matrix saved successfully!")