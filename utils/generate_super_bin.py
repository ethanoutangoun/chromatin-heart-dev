import numpy as np
import pandas as pd

# Load the bin map
bin_map_file = "data/bin_map_human_100000.bed"  
bin_map = pd.read_csv(bin_map_file, sep="\t", header=None, names=["chr", "start", "end", "bin_index"])


ttn_chrom = "chr2" 
ttn_start = 178525989  # TTN start
ttn_end = 178807423    # TTN end

# Get all bins that overlap TTN
ttn_bins = bin_map[(bin_map["chr"] == ttn_chrom) & 
                   (bin_map["start"] <= ttn_end) & 
                   (bin_map["end"] >= ttn_start)]["bin_index"].values

print(f"TTN spans bins: {ttn_bins}")

# Load Hi-C contact matrix (update with your format)
hic_matrix = np.load("/Users/ethan/Desktop/chromatin-heart-dev/samples/contact_matrix_100kb_zeroed.npy") 

# Compute Super Bin Contact Profile
super_bin_contacts = np.mean(hic_matrix[ttn_bins, :], axis=0)

# Save the results
np.save("ttn_super_bin_contacts.npy", super_bin_contacts)

print("Super bin contacts computed and saved.")