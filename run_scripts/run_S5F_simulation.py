import pandas as pd
from python_code.data_utils.utils import load_multiple_sets
from python_code.data_exploration.s5f_simulation import s5f_simulation

column_list = ['sequence_alignment', 
               'ancestor_alignment', 
               'mutations_all'] 

final_sets = pd.read_csv('data/final_sets.csv')
paths = final_sets[final_sets.study == 'mg'].path

dataset = load_multiple_sets(paths, column_list)
mutability_table = pd.read_csv('results/fivmers-mutability.csv')
mutability = mutability_table.mutability
mutability.index = mutability_table.motif

s5f_simulation(dataset.ancestor_alignment, mutability)
