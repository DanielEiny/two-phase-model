import torch

from python_code.data_preprocess.count_mutations import mismatch_positions, filter_synonymous
from python_code.model.model_utils import count_possible_synonymous_mutations


def calc_synonymous_mutation_likelihood(possible_synonymous_error_repair, possible_synonymous_replication, targeting_probs, replication_probs):
    each_nucleotid_substitution_prob = 1 / 3
    error_repair_probs = targeting_probs * (1 - replication_probs) * each_nucleotid_substitution_prob
    synonymous_error_repair_likelihood = possible_synonymous_error_repair.dot(error_repair_probs) 
    
    replication_probs = targeting_probs * replication_probs
    synonymous_replication_likelihood = possible_synonymous_replication.dot(replication_probs)
    likelihood = synonymous_error_repair_likelihood + synonymous_replication_likelihood

    return -torch.log(likelihood)

def calc_target_likelihood(ancestor_nucleotide, descendant_nucleotide, targeting_prob, replication_prob):
    likelihood = torch.tensor([0.], requires_grad=True) 
    
    each_nucleotid_substitution_prob = 1 / 3
    likelihood = likelihood + targeting_prob * (1 - replication_prob) * each_nucleotid_substitution_prob
    
    # Cases of possibly replication
    if (ancestor_nucleotide == 'C' and descendant_nucleotide == 'T') or \
       (ancestor_nucleotide == 'G' and descendant_nucleotide == 'A'):  
        likelihood = likelihood + targeting_prob * replication_prob

    return -torch.log(likelihood)

def calc_sequence_likelihood(ancestor_sequence, descendant_sequence, targeting_probs, replication_probs, only_synonymous=False):
    likelihood = torch.tensor([0.], requires_grad=True) 
    targets = mismatch_positions(ancestor_sequence, descendant_sequence)

    if only_synonymous:
        targets = filter_synonymous(ancestor_sequence, descendant_sequence, targets)
        if len(targets) == 0:
            return likelihood, 0
        possible_synonymous_error_repair, possible_synonymous_replication = count_possible_synonymous_mutations(ancestor_sequence)
        
    for t in targets:
        likelihood = likelihood + calc_target_likelihood(ancestor_sequence[t], 
                                                         descendant_sequence[t],
                                                         targeting_probs[t],
                                                         replication_probs[t])
        if likelihood.isinf():
            import ipdb; ipdb.set_trace()

        if only_synonymous:
            likelihood = likelihood - calc_synonymous_mutation_likelihood(possible_synonymous_error_repair,
                                                                          possible_synonymous_replication,
                                                                          targeting_probs,
                                                                          replication_probs)
        # Compensate no-replacement effect
        targeting_probs = targeting_probs.index_fill(0, torch.LongTensor([t]), 0)
        targeting_probs = targeting_probs / targeting_probs.sum()

    return likelihood, len(targets)

