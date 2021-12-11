import numpy as np
import pandas as pd

from python_code.definitions import imgt_regions
from python_code.utils import flatten_list


def clustering_test(dataset, target_nucleotides=[]):
    n_sequences = dataset.shape[0]

    # --- Look only on V gene --- #
    v_gene_end = imgt_regions['FR3'][1]
    dataset.mutations_all = dataset.mutations_all.apply(lambda x: [m for m in x if m < v_gene_end])
    dataset.ancestor_alignment = dataset.ancestor_alignment.apply(lambda x: x[:v_gene_end])
    
    # --- Optionally, filter by target nucleotides --- #
    if target_nucleotides:  # 
        dataset.mutations_all = dataset.apply(lambda row: [pos for pos in row.mutations_all if row.ancestor_alignment[pos] in target_nucleotides], axis=1)

    # --- Infer distribution of mutability  by position --- #
    mutations_list = flatten_list(dataset.mutations_all)
    mutations_per_position = pd.value_counts(mutations_list).sort_index()
    mutations_per_position_aligned = pd.Series(index=range(v_gene_end), data=0)
    mutations_per_position_aligned[mutations_per_position.index] = mutations_per_position
    normalized_mutability = (mutations_per_position_aligned / n_sequences).to_numpy()

    # --- Generate mutation with the above distribution --- #
    sample = np.random.binomial(1, normalized_mutability, size=(n_sequences, v_gene_end))
    import ipdb; ipdb.set_trace()
    pass


