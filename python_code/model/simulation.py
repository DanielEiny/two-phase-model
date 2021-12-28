import numpy as np
import torch


def simulation(sequence, n_mutations, model):

    # --- Get probabilities --- #
    with torch.no_grad():
        targeting_probs, replication_probs = model(sequence)
    
    # --- Sample targets from distributions --- #
    index = np.arange(len(targeting_probs))
    targets = np.random.choice(a=index, size=n_mutations, p=targeting_probs, replace=False)

    # --- Mutated sampled targets --- #
    mutated_sequence = list(sequence)
    for position in targets:
        replication = np.random.binomial(1, replication_probs[position])
        if replication:  
           mutated_sequence[position] = 'T'
        else:
           mutated_sequence[position] = np.random.choice(['G', 'T', 'A'])

    mutated_sequence = ''.join(mutated_sequence)
    return mutated_sequence
