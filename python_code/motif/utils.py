import pickle
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from python_code.definitions import nucleotides, uipac_ambiguity_codes, imgt_regions


def create_frequent_motif_vocab_old_version(min_freq, ignore=[]):
    steps = []
    ignore.append('N')
    motifs = pd.read_csv('results/motifs/mutability/fivmers-mutability-no-N.csv')
    motifs = motifs[motifs.motif.apply(lambda x: x[2] not in ignore)]
    motifs['freq'] = motifs.motif_count / motifs.motif_count.sum()
    motifs = motifs.drop(['motif_count', 'mutation_count', 'mutability'], axis=1)
    motifs = motifs.sort_values(by='freq')
    original_size = len(motifs)
    print(f' -> current min frq =  {motifs.freq.iloc[0]:.5f} | steps = {original_size - len(motifs)} | vocabulary size = {len(motifs)} <- ', end='\r') 
    # Iterate over motifs and merge lowest freq ones
    while(motifs.freq.iloc[0] < min_freq):
        merged = merge_motifs(motifs.iloc[0], motifs.iloc[1])
        steps.append((motifs.iloc[0].motif, motifs.iloc[1].motif, merged.motif))
        motifs = motifs[2:]  # drop these 2 lower freq motifs
        # Merge all motifs contained in the new motif
        for _, m in motifs.iterrows():
            if contains(merged.motif, m.motif):
                merged = merge_motifs(merged, m)
                motifs = motifs.drop(motifs[motifs.motif == m.motif].index)
                steps.append((merged.motif, m.motif))
        motifs = motifs.append(merged)
        motifs = motifs.sort_values(by='freq')
        print(f' -> current min frq =  {motifs.freq.iloc[0]:.5f} | steps = {original_size - len(motifs)} | vocabulary size = {len(motifs)} <- ', end='\r') 
    return motifs, steps


def merge_motifs(a, b):
    merged = a.copy()

    merged_motif = []
    for sa, sb in zip (a.motif, b.motif):
        sa = expand_ambiguity_code(sa)
        sb = expand_ambiguity_code(sb)
        union = set(sa) | set(sb)
        joind_sorted_union = ''.join(sorted(list(union)))
        union_symbol = reversed_uipac_codes[joind_sorted_union]
        merged_motif.append(union_symbol)

    merged.motif = ''.join(merged_motif)
    merged.freq = a.freq + b.freq

    return merged

def reverse_uipac_codes():
    return {''.join(expand_ambiguity_code(k)): k for k in uipac_ambiguity_codes}

def expand_ambiguity_code(symbol):
    regex = uipac_ambiguity_codes[symbol]
    split = regex.split('|')
    sort = sorted(split)
    return sort


reversed_uipac_codes = reverse_uipac_codes()


def contains(a, b):
    if a == b:
        return False
    per_symbol = 0
    for sa, sb in zip(a, b):
        sa = expand_ambiguity_code(sa)
        sb = expand_ambiguity_code(sb)
        intersection = set(sa) & set(sb)
        if len(intersection) == len(sb):
            per_symbol += 1
    if per_symbol == len(a):
        return True
    else:
        return False

def calc_merge_freq(args):
    vocab_symbols, vocab_freqs, merge_symbol = args
    contained = [contains(merge_symbol, symbol) for symbol in vocab_symbols if symbol != merge_symbol]

    if sum(contained) > 1:
        return vocab_freqs[contained].sum()
    else: 
        return 1.0  # If not really merge, only replacement.

def all_possible_motifs(anchor_type):
    if anchor_type == 'all':
        return [''.join(x)for x in product(uipac_ambiguity_codes.keys(), repeat=5)]
    elif anchor_type == 'only_CG':
        ambiguity_codes_only_CG = {k: v for k, v in uipac_ambiguity_codes.items() if uipac_ambiguity_codes[k].count('A') + uipac_ambiguity_codes[k].count('T') == 0}
        iterables = [
                    uipac_ambiguity_codes.keys(),
                    uipac_ambiguity_codes.keys(),
                    ambiguity_codes_only_CG.keys(),
                    uipac_ambiguity_codes.keys(),
                    uipac_ambiguity_codes.keys(),
                    ]
        return [''.join(x)for x in product(*iterables)]
    


def create_frequent_motif_vocab(target_min_freq, anchor_type, debug=False):
    if anchor_type == 'all':
        vocab = pd.read_csv('results/motifs/mutability/fivmers-mutability-no-N.csv')
        vocab['freq'] = vocab.motif_count / vocab.motif_count.sum()

    elif anchor_type == 'only_CG':
        vocab = pd.read_csv('results/motifs/mutability/fivmers-only_CG-no-N-synonymous_mutations_count.csv')
        vocab = pd.read_csv('results/motifs/merged_vocabularies_by_mutations_freq_only_CG/min_freq_0.0118431090393701_size_28.csv')

    vocab = vocab.drop(['motif_count', 'mutation_count', 'mutability'], axis=1, errors='ignore') 

    possible_merges = list(set(all_possible_motifs(anchor_type))- set(vocab.motif))
 
    original_vocab_len = len(vocab)
    steps = []
 
    while vocab.freq.min() < target_min_freq and len(steps) < original_vocab_len:
 
        vocab_symbols = vocab.motif.to_list()
        vocab_freqs = vocab.freq.to_numpy()
        args = [(vocab_symbols, vocab_freqs, ms) for ms in possible_merges]
        possible_merges_freq = process_map(calc_merge_freq, args, max_workers=50, chunksize=1)
        # possible_merges_freq = np.zeros(len(args))
        # for i, arg in enumerate(tqdm(args)):
        #     possible_merges_freq[i] = calc_merge_freq(arg)
        # import ipdb; ipdb.set_trace()
        lowest_freq_merge_idx =  np.argmin(possible_merges_freq)

        chosen_merge = possible_merges[lowest_freq_merge_idx]
        contained = [contains(chosen_merge, x) for x in vocab_symbols]
        merge_freq = vocab.freq[contained].sum()
        
        remove = vocab.motif[contained].to_list() + [chosen_merge]
        possible_merges = [x for x in possible_merges if x not in remove]
 
        if debug:
            steps.append({'merge': chosen_merge, 'to_drop': vocab.motif[contained]})
        else:
            steps.append(None)
 
        vocab = vocab[~np.array(contained)]
        vocab = vocab.append({'motif': chosen_merge, 'freq': merge_freq}, ignore_index=True)

        vocab.to_csv(path_or_buf=f'results/motifs/merged_vocabularies_by_mutations_freq_{anchor_type}/min_freq_{vocab.freq.min()}_size_{len(vocab)}.csv', index=False)
        with open('results/motifs/merged_vocabularies/possible_merges.pkl', 'wb') as f:
            pickle.dump(possible_merges, f)

    return vocab, steps

def create_frequent_motif_vocab_v3(target_min_freq, anchor_type, debug=False):
    if anchor_type == 'all':
        vocab = pd.read_csv('results/motifs/mutability/fivmers-mutability-no-N.csv')
        vocab['freq'] = vocab.motif_count / vocab.motif_count.sum()

    elif anchor_type == 'only_CG':
        vocab = pd.read_csv('results/motifs/mutability/fivmers-only_CG-no-N-synonymous_mutations_count.csv')

    vocab = vocab.drop(['motif_count', 'mutation_count', 'mutability'], axis=1, errors='ignore') 

 
    vocab['group_id'] = -1
    group_id = 0
    group_freq = 0

    if debug:
        group_freq_dict = {}
 
    for i in range(len(vocab)):
 
        min_freq_idx = vocab.freq[vocab.group_id == -1].idxmin()
        vocab.group_id[min_freq_idx] = group_id
        group_freq += vocab.freq[min_freq_idx]

        if group_freq > target_min_freq:
            if debug:
                group_freq_dict[group_id] = group_freq
            group_id += 1
            group_freq = 0
 

    vocab.to_csv(path_or_buf=f'results/motifs/merged_vocabularies_v3_{anchor_type}/min_freq_{target_min_freq}_size_{group_id}.csv', index=False)
    return 

if __name__ == "__main__":
    create_frequent_motif_vocab_v3(target_min_freq=0.001, anchor_type='only_CG', debug=True)
