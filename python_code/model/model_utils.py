from itertools import product
import torch
import torch.nn.functional as F

from python_code.definitions import nucleotides, uipac_ambiguity_codes
from python_code.data_utils.utils import is_synonymous


def parse_ambiguity_motif(ambiguity_motif):
    each_positions_possibilities = [uipac_ambiguity_codes[l].split('|') for l in ambiguity_motif]
    combinations = list(product(*each_positions_possibilities))
    return [''.join(x) for x in combinations]


def motif_ambiguity_to_regex(ambiguity_motif):
    motif_regex = ''.join(['[' + uipac_ambiguity_codes[symbol] + ']' for symbol in ambiguity_motif])
    return motif_regex

def assign_motif_probs(sequence, motif_list, anchor_dict, regex_dict, motif_idx_dict, prob_array):
    assigned_probs = torch.zeros(len(sequence))

    for motif in motif_list:
        offset = anchor_dict[motif]
        regex = regex_dict[motif]
        prob = prob_array[motif_idx_dict[motif]]
        positions = [match.start() + offset for match in regex.finditer(sequence)]
        assigned_probs[positions] = prob

    return assigned_probs

def assign_fivemer_probs(sequence, motif_idx_dict, motifs_prob):
    assigned_probs = torch.zeros(len(sequence))
    sequence = 'NN' + sequence + 'NN'  # Padding for valid probs assignment

    for i in range(len(sequence) - 4):
        motif = sequence[i:i+5]
        prob = motifs_prob[motif_idx_dict[motif]]
        assigned_probs[i] = prob

    return assigned_probs
    
def assign_n_mer_probs(sequence, motif_idx_dict, motifs_prob):
    assigned_probs = torch.zeros(len(sequence))
    motif_len = len(list(motif_idx_dict)[0])
    pad_size = int((motif_len - 1) / 2)
    pad = ''.join('N' * pad_size)
    sequence = pad + sequence + pad  # Padding for valid probs assignment

    for i in range(len(sequence) - pad_size * 2):
        motif = sequence[i:i+motif_len]
        prob = motifs_prob[motif_idx_dict[motif]]
        assigned_probs[i] = prob

    return assigned_probs

def normalize(tensor, dim=[]):
    denominatore = tensor.sum(dim)
    if dim:
        denominatore[denominatore == 0] = 1  # Numerical safty
        denominatore = denominatore.unsqueeze(dim)
    return tensor / denominatore

def probablize(tensor):
    ''' cast vector into probability parameter space: between (0, 1) and sum up to 1 '''
    eps = 1e-27
    tensor = F.relu(tensor) - F.relu(tensor - 1)
    tensor = tensor + eps  # for numerical stability
    if tensor.shape[0] > 1:
        tensor = normalize(tensor)
    return tensor

def count_possible_synonymous_mutations(sequence):
    sequence_len = len(sequence)
    error_repair = torch.zeros(sequence_len)
    replication = torch.zeros(sequence_len)
    
    for pos in range(2, sequence_len):
        reading_frame_pos = (pos // 3) * 3
        if reading_frame_pos + 3 > len(sequence):
            continue
        original_codon = sequence[reading_frame_pos:reading_frame_pos + 3]
        possible_substitutons = list(set(nucleotides) - set(sequence[pos]))
        for s in possible_substitutons:
           mutated_codon = original_codon[:pos % 3] + s + original_codon[pos % 3 + 1:]
           if is_synonymous(original_codon, mutated_codon):
                error_repair[pos] += 1

                # Cases of possibly replication
                if (sequence[pos] == 'C' and s == 'T') or \
                   (sequence[pos] == 'G' and s == 'A'):  
                       replication[pos] += 1

    return error_repair, replication

