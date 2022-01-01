from itertools import product
import torch

from python_code.definitions import uipac_ambiguity_codes


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

def normalize(tensor, dim=[]):
    denominatore = tensor.sum(dim)
    if dim:
        denominatore = denominatore.unsqueeze(dim)
    return tensor / denominatore
