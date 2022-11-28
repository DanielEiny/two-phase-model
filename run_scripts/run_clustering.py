import pandas as pd
from python_code.data_utils.utils import load_multiple_sets
from python_code.data_exploration.clustering import clustering_test

column_list = ['sequence_alignment', 
               'ancestor_alignment', 
               'sample_id',
               'clone_id',
               'mutations_all',
               'mutations_synonymous'] 

final_sets = pd.read_csv('data/final_sets.csv')
paths = final_sets[final_sets.study == 'mg'].path

dataset = load_multiple_sets(paths, column_list)
clustering_test(dataset)
