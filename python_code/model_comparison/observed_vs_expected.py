import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import pearsonr


from python_code.data_utils.utils import load_multiple_sets
from python_code.definitions import imgt_regions
from python_code.data_preprocess.count_mutations import mismatch_positions
from python_code.model.s5f import S5F
from python_code.model.shmoof import SHMOOF
from python_code.model.tpm_wrapper import TPM


# model_list = {'s5f': S5F(mutability_matrix_path="results/s5f/mutability_90%_data.csv"),
#               'shmoof': SHMOOF(),
#               'tpm': TPM(params_path='results/model/tpm/vocab_size_100/04_02_2024-21_29_43v_gene_family-all-study-all-v_only-synonymous-no_mmr-90%/state_dict_30000'),
#               'tpm_simulated': TPM(params_path='results/model/convergence_test_by_mutations_freq_fix_aid/07_11_2023-19_54_05_model_version-merged_vocab_size_100_n_sequences-1000000/state_dict_48800')}

model_list = {'tpm': TPM(params_path='results/model/tpm/v3/99_164/14_04_2024-18_15_20v_gene_family-all-study-all-v_only-synonymous-no_mmr-90%/state_dict_50000')}

data_source = '10%' 

if data_source == 'oof':
    data_files = glob.glob('data/shm_oof_french_research/_alignment_updated/*')
    dataset = pd.concat([pd.read_csv(x, sep='\t') for x in data_files])
    dataset = dataset[~dataset.ancestorseq.isna()]
    ANCESTOR_COLUMN = 'ancestorseq'
    DESCENDANT_COLUMN = 'ALIGNMENT'

elif data_source =='10%':
    columns_list = ['sequence_alignment', 'ancestor_alignment', 'mutations_all']
    all_sets = pd.read_csv('data/final_sets.csv')
    paths = all_sets[all_sets.sample_id != 'P4_I19_S1'].path
    dataset = load_multiple_sets(paths, columns_list)
    dataset = dataset[~(dataset.index.values % 10).astype(bool)]
    dataset.ancestor_alignment = dataset.ancestor_alignment.str.replace('.', 'N')
    ANCESTOR_COLUMN = 'ancestor_alignment'
    DESCENDANT_COLUMN = 'sequence_alignment'

# # Look only on V gene
# v_gene_end = imgt_regions['FR3'][1]
# dataset[ANCESTOR_COLUMN] = dataset[ANCESTOR_COLUMN].apply(lambda x: x[:v_gene_end])
# dataset[DESCENDANT_COLUMN] = dataset[DESCENDANT_COLUMN].apply(lambda x: x[:v_gene_end])

# Allocate arrays
maxlen = dataset[DESCENDANT_COLUMN].str.len().max()
observed = np.zeros(maxlen)
expected = {model_name: np.zeros(maxlen) for model_name in model_list.keys()}

for i in tqdm(range(len(dataset))):
    try:
        # Observed
        actual_targets = mismatch_positions(dataset[DESCENDANT_COLUMN].iloc[i], 
                                            dataset[ANCESTOR_COLUMN].iloc[i])
        observed[actual_targets] += 1

        # Expected
        for model_name in model_list.keys():
            targeting_probs = model_list[model_name].predict(dataset[ANCESTOR_COLUMN].iloc[i])
            expected[model_name][:len(targeting_probs)] += targeting_probs
    except:
        pass

# # Normalization
# observed = observed / observed.sum()
# for model_name in model_list.keys():
#     expected[model_name] /= expected[model_name].sum()

# Save results
results = {'observed': observed} | expected
with open(f'results/comparison/v3/vocab_size_99_164-{data_source}_data.pkl', 'wb') as f:
    pickle.dump(results, f)

