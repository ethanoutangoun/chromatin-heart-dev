# misc.py
import numpy as np

# Generate a sample symettric Hi-C like contact matrix with dense regions
def generate_sample_matrix_bins(n_bins):
    np.random.seed(42)
    
    dense_regions = [
        (0, n_bins // 10),   
        (n_bins // 10, n_bins // 5), 
        (n_bins // 5, n_bins // 2) 
    ]

    contact_matrix = np.random.rand(n_bins, n_bins) * 2

    for start, end in dense_regions:
        for i in range(start, end + 1):
            for j in range(start, end + 1):
                if i != j:
                    contact_matrix[i, j] = np.random.rand() * 8 + 2  

    contact_matrix = (contact_matrix + contact_matrix.T) / 2
    np.fill_diagonal(contact_matrix, 0)

    return contact_matrix
