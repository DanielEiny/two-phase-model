import glob
import pandas as pd

from python_code.data_utils.utils import load_multiple_sets

 
column_list = ['sequence_alignment', 
               'sequence_origin',
               'ancestor_alignment', 
               'ancestor_origin',
               'mutations_all', 
               'mutations_synonymous', 
               'mutations_full_synonymous', 
               'clone_id',
               'clone_size']

final_sets = pd.read_csv('data/final_sets.csv')

influ = load_multiple_sets(final_sets.path[final_sets.study == 'influenza'], column_list)
#hcv = load_multiple_sets(final_sets.path[final_sets.study == 'hcv'], column_list)
mg = load_multiple_sets(final_sets.path[final_sets.study == 'mg'], column_list)
ms = load_multiple_sets(final_sets.path[final_sets.study == 'ms'], column_list)
covid = load_multiple_sets(final_sets.path[final_sets.study == 'covid'], column_list)
cancer = load_multiple_sets(final_sets.path[final_sets.study == 'cancer'], column_list)

# --- Original data sets --- #
#hcv_sets = glob.glob('/work/peresay/vdjbase/V9/P4/*/*_cloned_w_filtered_seqs.tsv')
#influ_sets = glob.glob('/work/peresay/vdjbase/V9/P3/*/*_cloned_w_filtered_seqs.tsv')
#mg_sets = glob.glob('/work/peresay/vdjbase/V9/P9/*/*_cloned_w_filtered_seqs.tsv')
#ms_sets = glob.glob('/work/peresay/vdjbase/V9/P10/*/*_cloned_w_filtered_seqs.tsv')
#covid_sets = glob.glob('/work/smodi/OurCovid/28_10_2021/IGHCov*.tsv')
#cancer_sets = glob.glob('/work/data/colon_cancer/*_H/*seqs.tsv')
