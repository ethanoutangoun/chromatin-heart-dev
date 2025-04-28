# clique_finding.py 

import numpy as np
from tqdm import tqdm

def find_greedy_clique(contact_matrix, n, target_bin=None):
    """
    Greedy clique of size n, optimized via NumPy vector operations.
    Assumes contact_matrix[i, j] == 0 whenever i and j are on the same chromosome.
    """
    N = contact_matrix.shape[0]
    # 1) pick a starting node
    if target_bin is None:
        current = np.random.randint(N)
    else:
        current = target_bin

    clique = [current]

    # 2) boolean array of bins we cannot pick
    excluded = np.zeros(N, dtype=bool)
    # exclude ±10 around the start
    lo, hi = max(0, current - 10), min(N, current + 11)
    excluded[lo:hi] = True

    # 3) greedy growing
    while len(clique) < n:
        # a) compute total interaction of every node with the current clique
        #    this gives an (N,) array of sums
        sums = contact_matrix[:, clique].sum(axis=1)

        # b) mean = sums / |clique|
        means = sums / len(clique)

        # c) mask out already excluded or already in clique
        means[excluded] = -np.inf
        means[clique]   = -np.inf

        # d) pick the best new node
        max_node = int(np.argmax(means))
        if means[max_node] == -np.inf:
            # no more valid candidates
            break

        # e) add to clique and update exclusion window
        clique.append(max_node)
        lo, hi = max(0, max_node - 10), min(N, max_node + 11)
        excluded[lo:hi] = True

    return clique

def find_clique_greedy_fast(contact_matrix, n, target_bin=None):
    """
    Greedy clique of size n, optimized via NumPy vector operations.
    Assumes contact_matrix[i, j] == 0 whenever i and j are on the same chromosome.
    """
    N = contact_matrix.shape[0]
    # 1) pick a starting node
    if target_bin is None:
        current = np.random.randint(N)
    else:
        current = target_bin

    clique = [current]

    # 2) boolean array of bins we cannot pick
    excluded = np.zeros(N, dtype=bool)
    # exclude ±10 around the start
    lo, hi = max(0, current - 10), min(N, current + 11)
    excluded[lo:hi] = True

    # 3) greedy growing
    while len(clique) < n:
        # a) compute total interaction of every node with the current clique
        #    this gives an (N,) array of sums
        sums = contact_matrix[:, clique].sum(axis=1)

        # b) mean = sums / |clique|
        means = sums / len(clique)

        # c) mask out already excluded or already in clique
        means[excluded] = -np.inf
        means[clique]   = -np.inf

        # d) pick the best new node
        max_node = int(np.argmax(means))
        if means[max_node] == -np.inf:
            # no more valid candidates
            break

        # e) add to clique and update exclusion window
        clique.append(max_node)
        lo, hi = max(0, max_node - 10), min(N, max_node + 11)
        excluded[lo:hi] = True

    return clique

def random_walk(contact_matrix, start_node, n, num_molecules=100, alpha=0.1, verbose=False):
    num_nodes = contact_matrix.shape[0]
    visit_count = np.zeros(num_nodes, dtype=int)
    
    iterator = tqdm(range(num_molecules)) if verbose else range(num_molecules)
    
    for _ in iterator: 
        current_node = start_node
        
        while True:
            visit_count[current_node] += 1  # Track visits per molecule
            
            if np.random.rand() < alpha:
                break
            
            neighbors = np.where(contact_matrix[current_node] > 0)[0]
            if len(neighbors) == 0:
                break
            
            weights = contact_matrix[current_node, neighbors]
            probabilities = weights / np.sum(weights)
            
            next_node = np.random.choice(neighbors, p=probabilities)
            current_node = next_node

    clique = np.argsort(visit_count)[-n:][::-1]


    
    return clique

def random_walk_fast(contact_matrix, start_node, n,
                     neighbors, cdfs,
                     num_molecules=100, alpha=0.1):
    """
    A much faster random‐walk using prebuilt neighbor/CDF lists.
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
            # find next index in CDF
            j = np.searchsorted(cdfs[cur], r, side='right')
            cur = neighbors[cur][j]

    # top-n visited
    return np.argsort(visit_count)[-n:][::-1]
