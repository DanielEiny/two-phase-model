import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('Solarize_Light2')


def plot_hist(value_counts, file='plot.png', title=[], inches=(16, 9)):
    plt.clf()
    plt.figure(figsize=inches)
    if title:
        plt.title(title)
    value_counts.plot(kind='bar')
    plt.savefig(file)

def load_multiple_sets(db_paths: list, columns: list):
    sets = [pd.read_csv(x, sep='\t', usecols=columns) for x in db_paths]
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
