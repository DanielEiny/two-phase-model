import numpy as np
import torch

from python_code.definitions import nucleotides

replication_substitution = {'C': 'T', 'G': 'A'}

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
           mutated_sequence[position] = replication_substitution[sequence[position]]
        else:
           possible_substitutions = list(set(nucleotides) - set(sequence[position]))
           mutated_sequence[position] = np.random.choice(possible_substitutions)

    mutated_sequence = ''.join(mutated_sequence)
    return mutated_sequence
