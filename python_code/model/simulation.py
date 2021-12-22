import numpy as np
import torch


def simulation(sequence, n_mutations, model):

    # --- Get probabilities --- #
    with torch.no_grad():
        replication_probs, ber_or_mmr_probs = model(sequence)
    
    # --- Stack and normalize all probs together --- #
    probs = np.concatenate([replication_probs, ber_or_mmr_probs])
    probs = probs / probs.sum()
    index = np.arange(probs.size)
    
    # --- Sample targets from distributions --- #
    targets = np.random.choice(a=index, size=n_mutations, p=probs, replace=False)
    targets = np.unravel_index(indices=targets, shape=(2, int(probs.size / 2)))

    # --- Mutated sampled targets --- #
    mutated_sequence = list(sequence)
    for type_of_target, position_of_target in zip(*targets):
        if type_of_target == 0:  # TODO: replace with enum
           mutated_sequence[position_of_target] = 'T'
        elif type_of_target == 1:
           mutated_sequence[position_of_target] = np.random.choice(['G', 'T', 'A'])

    mutated_sequence = ''.join(mutated_sequence)
    return mutated_sequence
