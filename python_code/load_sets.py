 --- Single sample ---
HCV = pd.read_csv('/work/peresay/vdjbase/V9/P4/P4_I1_S1/P4_I1_S1.tsv', sep='\t')
INF = pd.read_csv('/work/peresay/vdjbase/V9/P3/P3_I1_S1/P3_I1_S1.tsv', sep='\t')
MS = pd.read_csv('/work/peresay/vdjbase/V9/P10/P10_I2_S1/P10_I2_S1.tsv', sep='\t')
COVID = pd.read_csv('/work/smodi/OurCovid/28_10_2021/IGHCov_71_cloned_w_filtered_seqs.tsv', sep='\t')
CANCER = pd.read_csv('/work/data/colon_cancer/100794_H/100794_H_cloned_w_filtered_seqs.tsv', sep='\t')



 --- Complete data sets ---
hcv_sets = glob.glob('/work/peresay/vdjbase/V9/P4/*/*S1.tsv')
inf_sets = glob.glob('/work/peresay/vdjbase/V9/P3/*/*S1.tsv')
ms_sets = glob.glob('/work/peresay/vdjbase/V9/P10/*/*S1.tsv')
covid_sets = glob.glob('/work/smodi/OurCovid/28_10_2021/IGHCov*.tsv')
cancer_sets = glob.glob('/work/data/colon_cancer/*_H/*seqs.tsv')
 
