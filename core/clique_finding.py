# clique_finding.py 

import numpy as np
from tqdm import tqdm


def find_greedy_clique(contact_matrix, n, target_bin=None):
    N = contact_matrix.shape[0]

    current = np.random.randint(N) if target_bin is None else target_bin
    clique = [current]

    excluded = np.zeros(N, dtype=bool)
    excluded[max(0, current - 10):min(N, current + 11)] = True

    while len(clique) < n:
        if len(clique) == 1:
            sums = contact_matrix[:, clique[0]]
        else:
            sums = np.sum(contact_matrix[:, clique], axis=1)
        means = sums / len(clique)

        invalid = excluded.copy()
        invalid[clique] = True
        means[invalid] = -np.inf

        max_node = np.argmax(means)
        if means[max_node] == -np.inf:
            break

        clique.append(max_node)
        excluded[max(0, max_node - 10):min(N, max_node + 11)] = True

    return clique

# Helper function to build neighbors and CDFs for random walk
def build_walk_index(contact_matrix):
    N = contact_matrix.shape[0]
    neighbors = [None]*N
    cdfs      = [None]*N

    for i in tqdm(range(N)):
        w = contact_matrix[i]
        idx = np.nonzero(w)[0]
        if idx.size == 0:
            neighbors[i] = np.empty(0, dtype=int)
            cdfs[i]      = np.empty(0, dtype=float)
        else:
            probs = w[idx] / w[idx].sum()
            neighbors[i] = idx
            cdfs[i]      = np.cumsum(probs)
    return neighbors, cdfs

def random_walk(contact_matrix, start_node, n,
                     neighbors, cdfs,
                     num_molecules=100, alpha=0.1):
    """
    Random walk on a contact matrix, starting from start_node. 
    Uses precomputed neighbors and CDFs for efficiency.
    """
    N = contact_matrix.shape[0]
    visit_count = np.zeros(N, dtype=int)

    for _ in range(num_molecules):
        cur = start_node
        while True:
            visit_count[cur] += 1
            if np.random.rand() < alpha or neighbors[cur].size == 0:
                break
            r = np.random.rand()
            j = np.searchsorted(cdfs[cur], r, side='right')
            cur = neighbors[cur][j]

    return np.argsort(visit_count)[-n:][::-1]

def analytical_diffusion_clique(contact_matrix: np.ndarray,
                                start_node: int,
                                n: int,
                                alpha: float = 0.1):

    N = contact_matrix.shape[0]

    P = np.zeros((N, N), dtype=float)
    row_sums = contact_matrix.sum(axis=1)
    for i in range(N):
        if row_sums[i] > 0:
            P[i, :] = contact_matrix[i, :] / row_sums[i]
        else:
            P[i, i] = 1.0

    I = np.eye(N)
    F = np.linalg.inv(I - (1 - alpha) * P)

    visits = F[start_node, :]
    

    clique = np.argsort(visits)[-n:][::-1]
    return clique, visits

# computes steady state diffusion for all nodes in the contact matrix
def all_analytical_diffusions(contact_matrix: np.ndarray, alpha: float = 0.1, output_prefix=False):
    N = contact_matrix.shape[0]

    P = np.zeros((N, N), dtype=float)
    row_sums = contact_matrix.sum(axis=1)
    for i in range(N):
        if row_sums[i] > 0:
            P[i, :] = contact_matrix[i, :] / row_sums[i]
        else:
            P[i, i] = 1.0 

    I = np.eye(N)
    F = np.linalg.inv(I - (1 - alpha) * P)


    if output_prefix:
        np.save(f'{output_prefix}.npy', F)

    return F 