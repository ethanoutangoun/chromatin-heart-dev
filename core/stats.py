from scipy import stats
import numpy as np

def calculate_avg_interaction_strength(contact_matrix, clique):
    total_interaction_strength = 0
    num_edges = 0

    # Loop through each unique pair of bins in the clique to get score of each edge
    for i in range(len(clique)):
        for j in range(i + 1, len(clique)):
            bin1 = clique[i]
            bin2 = clique[j]
            total_interaction_strength += contact_matrix[bin1, bin2]
            num_edges += 1

    # Calculate average interaction strength
    avg_interaction_strength = total_interaction_strength / num_edges if num_edges > 0 else 0
    return avg_interaction_strength


def t_test(observed_score, score_list):
    t_stat, p_value = stats.ttest_1samp(score_list, observed_score)
    return p_value


def empirical_p_value(observed_score, random_scores):
    random_scores = np.asarray(random_scores)
    n = len(random_scores)
    count = np.sum(random_scores >= observed_score)
    return (count + 1) / (n + 1)