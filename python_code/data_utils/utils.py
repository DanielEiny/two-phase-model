import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from python_code.definitions import imgt_regions, nucleotides, codon_table


def to_numpy(line):
    return np.array([float(x) for x in line.strip('[]').split(' ') if x])

def is_synonymous(codon_a, codon_b):
    for symbol in codon_a + codon_b:
        if symbol not in nucleotides:
            return False

    return codon_table[codon_a] == codon_table[codon_b]

def plot_hist(value_counts, file='plot.png', title=[], inches=(16, 9)):
    plt.clf()
    plt.figure(figsize=inches)
    if title:
        plt.title(title)
    value_counts.plot(kind='bar')
    plt.savefig(file)

def load_multiple_sets(db_paths: list, columns: list = None):
    # sets = [pd.read_csv(x, sep='\t', usecols=columns) for x in db_paths]

    sets = []
    for db in db_paths:
        print(f'loading db: {db}')
        sets.append(pd.read_feather(db.replace('tsv', 'feather'), columns=columns))

    return pd.concat(sets)

def clone_size_distribution(list_of_sets, 
                            clone_column='clone_id', 
                            min_dupcount=0, 
                            min_conscount=0):
    stacked = pd.Series()

    for set_path in list_of_sets:
       repertoire = pd.read_csv(set_path, sep='\t')
       # TODO: filter function
       repertoire = repertoire[(~repertoire.clone_id.isna()) & \
                               (~repertoire.sequence_id.str.contains('FAKE')) & \
                               (repertoire.consensus_count >= min_conscount) & \
                               (repertoire.duplicate_count >= min_dupcount)]
       clone_sizes = repertoire.clone_id.value_counts()
       stacked =  pd.concat([stacked, clone_sizes])

    return stacked.value_counts()


def mutability_by_position(dataset):
    mutations_list = dataset.mutations_all[dataset.ancestor_origin != "GERMLINE"].to_list()
    mutations_list = [item for sublist in mutations_list for item in sublist]
    mutations_per_position = pd.value_counts(mutations_list).sort_index()
    mutations_per_position = mutations_per_position[mutations_per_position.index < 450]
    mutations_per_position_aligned = pd.Series(index=range(450), data=0)
    mutations_per_position_aligned[mutations_per_position.index] = mutations_per_position
    mutations_per_position_aligned.plot(kind="bar")
    plt.locator_params(nbins=49, axis='x')
    plt.locator_params(nbins=10, axis='y')
    plt.axvspan(*imgt_regions['CDR1'], facecolor='gray', alpha=0.5)
    plt.axvspan(*imgt_regions['CDR2'], facecolor='gray', alpha=0.5)


    
def mutability_by_nucleotide(dataset):
    nucleotide_lists = dataset.apply(lambda row: [row.ancestor_alignment[pos] for pos in row.mutations_all],
                                     axis=1)
    all_nucleotides = [item for sublist in nucleotide_lists for item in sublist]
    counts = pd.value_counts(all_nucleotides)
    counts.plot(kind='pie')
