import pandas as pd

from python_code.definitions import nucleotides
from python_code.data_utils.utils import load_multiple_sets
from python_code.data_exploration.motifs import calc_multiprocess
from python_code.data_exploration.motifs import calc_loop


column_list = ['sequence_alignment', 
               'ancestor_alignment', 
               'mutations_all'] 

final_sets = pd.read_csv('data/final_sets.csv')
paths = final_sets[final_sets.sample_id != 'P4_I19_S1'].path

data = load_multiple_sets(paths, column_list)

target = 'T'
for substitution in nucleotides:
    if target != substitution:
        calc_loop(csv_to_append=f'results/substitution_{target.lower()}_to_{substitution.lower()}.csv',
                  dataset=data, 
                  anchor_nucleotide=target, 
                  positions_left=2, 
                  positions_right=2,
                  substitution=substitution)


