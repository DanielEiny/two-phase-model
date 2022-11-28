import os
import pandas as pd
from python_code.data_preprocess.define_clones import define_clones

 
missing_list = [
          #'P9_I2_S1',
          #'P9_I5_S1',
          #'P9_I16_S1',
          #'P9_I16_S2',
          #'P9_I16_S3',
          #'IGHCov_6',
          '100428_H'
          ]

samples_table = pd.read_csv('data/original_sets.csv')

samples_table = samples_table[samples_table.sample_id.isin(missing_list)]
#samples_table = samples_table[samples_table.sample_id == 'P9_I7_S2']

samples_table.apply(lambda row: define_clones(row.path, 
                                              os.path.join('data', row.study_id)),
                                              axis = 1)

