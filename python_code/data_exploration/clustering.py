import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

from python_code.definitions import imgt_regions
from python_code.utils import flatten_list


def clustering_test(dataset, target_nucleotides=[]):
    n_sequences = dataset.shape[0]

    # --- Look only on V gene --- #
    v_gene_end = imgt_regions['FR3'][1]
    dataset.mutations_all = dataset.mutations_all.apply(lambda x: [m for m in x if m < v_gene_end])
    dataset.ancestor_alignment = dataset.ancestor_alignment.apply(lambda x: x[:v_gene_end])
    
    import ipdb; ipdb.set_trace()
    # --- Optionally, filter by target nucleotides --- #
    if target_nucleotides:  # 
        dataset.mutations_all = dataset.apply(lambda row: [pos for pos in row.mutations_all if row.ancestor_alignment[pos] in target_nucleotides], axis=1)

    # --- Drop sequences with no mutations --- #
    dataset = dataset[dataset.mutations_all.astype(bool)] 

    # --- Drop sequences with too many mutations --- #
    max_mutations_per_seq = 8
    mutations_per_sequence = dataset.mutations_all.apply(len)
    dataset = dataset[mutations_per_sequence < max_mutations_per_seq]

    # --- Infer distribution of mutability by position --- #
    mutations_list = flatten_list(dataset.mutations_all)
    mutations_per_position = pd.value_counts(mutations_list).sort_index()
    mutations_per_position_aligned = pd.Series(index=range(v_gene_end), data=0)
    mutations_per_position_aligned[mutations_per_position.index] = mutations_per_position
    normalized_mutability = (mutations_per_position_aligned / n_sequences).to_numpy()

    # --- Generate mutation with the above distribution --- #
    simulated_data = np.random.binomial(1, normalized_mutability, size=(n_sequences * 10, v_gene_end))

    # --- Infer distribution of mutations count per sequence --- #
    mutations_per_sequence_dist = mutations_per_sequence.value_counts()[:max_mutations_per_seq]
    
    # --- Filter simulated data so it will follow the above distribution --- #
    simulated_data_corrected = []
    simulated_data_mutations_count = simulated_data.sum(axis=1)
    for number, count in mutations_per_sequence_dist.iteritems():
        filter = simulated_data_mutations_count == number
        print(f'{number} - {count} - {filter.sum()}')
        simulated_data_corrected.append(simulated_data[filter][:count])
    
    import ipdb; ipdb.set_trace()
    simulated_data = np.concatenate(simulated_data_corrected)

    # --- Filter real data to cope with simulated data distribution limits --- #
    import ipdb; ipdb.set_trace()

    # --- Calc pairwise dists for real data --- #
    real_data_dists = dataset.mutations_all.apply(lambda x: pdist(np.array([x]).T))
    real_data_dists = pd.value_counts(flatten_list(real_data_dists)).sort_index()

    # --- Calc pairwise dists for simulated_data data --- #
    simulated_data_dists = [np.where(x)[0] for x in simulated_data]
    simulated_data_dists = [pdist(x[:, np.newaxis]) for x in simulated_data_dists]
    simulated_data_dists = pd.value_counts(flatten_list(simulated_data_dists)).sort_index()

    import ipdb; ipdb.set_trace()
    pass


