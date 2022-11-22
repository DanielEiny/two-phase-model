import os
import time
import numpy as np
import pandas as pd

from python_code.definitions import *
from python_code.data_utils.utils import is_synonymous


def mismatch_positions(seq_a, seq_b):
    assert len(seq_a) == len(seq_b)
    seq_a = np.array(list(seq_a))
    seq_b = np.array(list(seq_b))
    filter_a = (seq_a != '.') & (seq_a != '-') & (seq_a != 'N')
    filter_b = (seq_b != '.') & (seq_b != '-') & (seq_b != 'N')
    compare = seq_a != seq_b
    position = np.where(filter_a & filter_b & compare)[0]
    return list(position.astype('int32'))

def filter_synonymous(original_sequence, mutated_sequence, mutations):
    synonymous_mutations = []
    for pos in mutations:
        reading_frame_pos = (pos // 3) * 3
        if reading_frame_pos + 3 > len(original_sequence):
            continue
        codon_a = original_sequence[reading_frame_pos:reading_frame_pos + 3]
        codon_b = codon_a[:pos % 3] + mutated_sequence[pos] + codon_a[pos % 3 + 1:]
        if is_synonymous(codon_a, codon_b):
            synonymous_mutations.append(pos)

    return synonymous_mutations

def filter_full_synonymous(original_sequence, mutations):
    full_synonymous_mutations = []

    for pos in mutations:
        is_all_synonymous = True
        reading_frame_pos = (pos // 3) * 3
        codon = original_sequence[reading_frame_pos:reading_frame_pos + 3]

        for nuc in nucleotides:
            possible_mut_codon = codon[:pos % 3] + nuc + codon[pos % 3 + 1:]
            if not is_synonymous(codon, possible_mut_codon):
                is_all_synonymous = False

        if is_all_synonymous:
            full_synonymous_mutations.append(pos)

    return full_synonymous_mutations

def count_mutations(input_file_path, output_file_path):
    # --- Read data --- #
    repertoire = pd.read_csv(input_file_path, sep='\t')

    # --- Add clone size column --- #
    file_name = os.path.basename(input_file_path)
    sample_name = file_name.replace('_cloned_w_filtered_seqs.tsv', '')
    print(f'{time.ctime()} | {sample_name}: Clone sizes ')

    clone_counts = repertoire.clone_id.value_counts()
    clone_size = clone_counts[repertoire.clone_id]
    repertoire = repertoire.assign(clone_size=clone_size.values)

    # --- Mark mutations --- #
    print(f'{time.ctime()} | {sample_name}: Mark mutations ')
    mutations_all = repertoire.apply(lambda x: mismatch_positions(x.sequence_alignment,
                                                                  x.ancestor_alignment),
                                                                  axis=1)
    repertoire = repertoire.assign(mutations_all=mutations_all)

    # --- Mark synonymous mutations --- #
    print(f'{time.ctime()} | {sample_name}: Mark synonymous mutations ')
    mutations_synonymous = repertoire.apply(lambda x: filter_synonymous(x.ancestor_alignment,
                                                                        x.sequence_alignment,
                                                                        x.mutations_all),
                                                                        axis=1)
    repertoire = repertoire.assign(mutations_synonymous=mutations_synonymous)

    # --- Mark full synonymous mutations --- #
    print(f'{time.ctime()} | {sample_name}: Mark full synonymous mutations ')
    mutations_full_synonymous = repertoire.apply(lambda x: filter_full_synonymous(x.ancestor_alignment,
                                                                                  x.mutations_all),
                                                                                  axis=1)
    repertoire = repertoire.assign(mutations_full_synonymous=mutations_full_synonymous)

    # --- Save output --- #
    print(f'{time.ctime()} | {sample_name}: Save output ')
    repertoire.to_feather(output_file_path.replace('tsv', 'feather'))


if __name__ == "__main__":
    count_mutations("data/tmp.tsv",
                    "tmp/P4_I1_S1_cloned_w_filtered_seqs.tsv")

