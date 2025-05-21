import numpy as np

mat = np.load("data/hic/full_contacts/knockout_500kb_balanced.npy")     
with open("contacts.tsv", "w") as out:
    N = mat.shape[0]
    for i in range(N):
        for j in range(i, N):
            c = mat[i, j]
            if c > 1e-10:  # adjust threshold as needed
                out.write(f"{i}\t{j}\t{c:.6f}\n")  # write float with precision
