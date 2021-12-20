import os
import re
import time
import numpy as np
import pandas as pd
from itertools import product
from pathos.multiprocessing import ProcessPool

from python_code.definitions import nucleotides, uipac_ambiguity_codes, imgt_regions


def calc_motif_mutability(dataset, motif, motif_anchor, substitution=[]):
    # --- Carefulness --- #
    dataset = dataset.copy()

    # --- Look only on V gene --- #
    v_gene_end = imgt_regions['FR3'][1]
    dataset.ancestor_alignment = dataset.ancestor_alignment.apply(lambda x: x[:v_gene_end])
    dataset.mutations_all = dataset.mutations_all.apply(lambda x: [m for m in x if m < v_gene_end])

    # --- Find mofif occurences --- #
    regex_motif = re.compile(motif)
    motif_anchors = dataset.ancestor_alignment.apply(regex_motif.finditer)
    motif_anchors = motif_anchors.apply(lambda x: [match.start() + motif_anchor for match in x]) 

    # --- Sum occurences --- #
    dataset['motif_anchors'] = motif_anchors
    motif_count = dataset.motif_anchors.apply(len).sum()
    
    # --- Intersect with mutations --- #
    if substitution:  # optionally, filter by substitution nucleotides
        dataset.mutations_all = dataset.apply(lambda row: [pos for pos in row.mutations_all if row.sequence_alignment[pos] in substitution], axis=1)
    mutation_count = dataset.apply(lambda row: len(set.intersection(set(row.motif_anchors),
                                                                    set(row.mutations_all))), axis=1).sum()
    if mutation_count > 0:
        ratio = mutation_count / motif_count
    return motif_count, mutation_count, ratio

def generate_motifs(anchor_nucleotide, positions_left, positions_right, symbols=nucleotides):
    for l in range(positions_left + 1):
        for r in range(positions_right + 1):
            anchor = l
            left_wing_combs = [''.join(x) for x in product(symbols, repeat=l)]
            right_wing_combs = [''.join(x) for x in product(symbols, repeat=r)]
            for left_wing in left_wing_combs:
                for right_wing in right_wing_combs:
                    motif = left_wing + anchor_nucleotide + right_wing
                    yield motif, anchor

def calc_and_save(csv_to_append, dataset, motif, motif_anchor, substitution=[]):
    print(f'{time.ctime()} | motif: {motif} | start')
    motif_count, mutation_count, _ = calc_motif_mutability(dataset, motif, motif_anchor, substitution)
    df = pd.DataFrame({'motif': motif, 'anchor': motif_anchor, 'motif_count': motif_count, 'mutation_count': mutation_count}, index=[0])
    df.to_csv(csv_to_append, mode='a', header=not os.path.isfile(csv_to_append), index=False)
    print(f'{time.ctime()} | motif: {motif} | saved')

def calc_test(csv_to_append, motif, motif_anchor, substitution=[]):
    print(f'{time.ctime()} | motif: {motif} | start')
    res = len([x for x in range(int(10e7))])
    with open('data/tmp.txt', 'a+') as f:
        f.write(f'{motif} | {res}')
    print(f'{time.ctime()} | motif: {motif} | saved')

def calc_multiprocess(csv_to_append, dataset, anchor_nucleotide, positions_left, positions_right):
    motifs = list(generate_motifs(anchor_nucleotide, positions_left, positions_right))
    pool = ProcessPool(10)
    #pool.map(lambda x: calc_and_save(csv_to_append, dataset.copy(), motif=x[0], motif_anchor=x[1]),
    #        motifs)
    pool.map(lambda x: calc_test(csv_to_append, motif=x[0], motif_anchor=x[1]),
             motifs)

def calc_loop(csv_to_append, dataset, anchor_nucleotide, positions_left, positions_right, substitution=[]):
    motifs = list(generate_motifs(anchor_nucleotide, positions_left, positions_right))
    for m in motifs:
        calc_and_save(csv_to_append, dataset.copy(), motif=m[0], motif_anchor=m[1], substitution=substitution)

def calc_ambiguity_codes(in_csv, out_csv, anchor_nucleotide, positions_left, positions_right):
    in_table = pd.read_csv(in_csv)
    motif_list, anchor_list = zip(*list(generate_motifs(anchor_nucleotide, 
                                                        positions_left, 
                                                        positions_right, 
                                                        uipac_ambiguity_codes)))
    out_table = pd.DataFrame(data={'motif': motif_list,
                                   'anchor': anchor_list,
                                   'motif_count': 0, 
                                   'mutation_count': 0})
    for i in range(len(motif_list)):
        anchor = out_table.anchor[i]
        motif = ''
        for symbol in out_table.motif[i]:
            motif = motif + '[' + uipac_ambiguity_codes[symbol] + ']'
        motif_regex = re.compile('^' + motif + '$')
        match = in_table.motif.apply(motif_regex.search).astype(bool)
        match_slice = in_table[match & (in_table.anchor == anchor)]
        out_table.motif_count[i] = match_slice.motif_count.sum()
        out_table.mutation_count[i] = match_slice.mutation_count.sum()

    out_table.to_csv(out_csv, index=False)

# TODO: pandas DataFrame parallelism framework, sharing df using namespace
# https://stackoverflow.com/questions/29748481/how-to-use-pass-by-reference-for-data-frame-in-python-pandas
# https://stackoverflow.com/questions/22487296/multiprocessing-in-python-sharing-large-object-e-g-pandas-dataframe-between
