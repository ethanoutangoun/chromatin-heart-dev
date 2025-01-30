import cooler
import numpy as np

# Load the cooler object at 100kb resolution
mcool_path = "4DNFISWHXA16.mcool"
clr = cooler.Cooler(f"{mcool_path}::resolutions/100000")
matrix = clr.matrix(balance=True)[:]
matrix = np.nan_to_num(matrix)


np.save("contact_matrix_100kb_balanced.npy", matrix)
print("Matrix saved successfully!")