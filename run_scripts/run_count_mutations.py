import os
import pandas as pd
from python_code.data_preprocess.count_mutations import count_mutations


samples_table = pd.read_csv('data/final_sets.csv')
samples_table = samples_table[samples_table.sample_id != 'P4_I19_S1']
samples_table = samples_table[samples_table.sample_id == 'P3_I2_S2']

samples_table.apply(lambda row: count_mutations(row.path, 
                                                row.path),
                                                axis = 1)

