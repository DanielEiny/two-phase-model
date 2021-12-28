from itertools import product

from python_code.definitions import uipac_ambiguity_codes


def parse_ambiguity_motif(ambiguity_motif):
    each_positions_possibilities = [uipac_ambiguity_codes[l].split('|') for l in ambiguity_motif]
    combinations = list(product(*each_positions_possibilities))
    return [''.join(x) for x in combinations]


def motif_ambiguity_to_regex(ambiguity_motif):
    motif_regex = ''.join(['[' + uipac_ambiguity_codes[symbol] + ']' for symbol in ambiguity_motif])
    return motif_regex
