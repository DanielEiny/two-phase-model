import glob
import pandas as pd


# --- Single sample --- #
hcv = pd.read_csv('/work/peresay/vdjbase/V9/P4/P4_I1_S1/P4_I1_S1_cloned_w_filtered_seqs.tsv', sep='\t')
influ = pd.read_csv('/work/peresay/vdjbase/V9/P3/P3_I1_S1/P3_I1_S1_cloned_w_filtered_seqs.tsv', sep='\t')
mg = pd.read_csv('/work/peresay/vdjbase/V9/P9/P9_I19_S3/P9_I19_S3_cloned_w_filtered_seqs.tsv', sep='\t')
ms = pd.read_csv('/work/peresay/vdjbase/V9/P10/P10_I2_S1/P10_I2_S1_cloned_w_filtered_seqs.tsv', sep='\t')
covid = pd.read_csv('/work/smodi/OurCovid/28_10_2021/IGHCov_71_cloned_w_filtered_seqs.tsv', sep='\t')
cancer = pd.read_csv('/work/data/colon_cancer/100794_H/100794_H_cloned_w_filtered_seqs.tsv', sep='\t')



# --- Complete data sets --- #
hcv_sets = glob.glob('/work/peresay/vdjbase/V9/P4/*/*_cloned_w_filtered_seqs.tsv')
influ_sets = glob.glob('/work/peresay/vdjbase/V9/P3/*/*_cloned_w_filtered_seqs.tsv')
mg_sets = glob.glob('/work/peresay/vdjbase/V9/P9/*/*_cloned_w_filtered_seqs.tsv')
ms_sets = glob.glob('/work/peresay/vdjbase/V9/P10/*/*_cloned_w_filtered_seqs.tsv')
covid_sets = glob.glob('/work/smodi/OurCovid/28_10_2021/IGHCov*.tsv')
cancer_sets = glob.glob('/work/data/colon_cancer/*_H/*seqs.tsv')
 
