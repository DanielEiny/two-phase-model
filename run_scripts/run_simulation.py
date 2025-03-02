import os
import sys
import pandas as pd
from tqdm import tqdm

from python_code.data_utils.utils import load_multiple_sets
from python_code.model.simulation import simulation


# --- Set run params --- #
model_version = 'v3_61_88'
os.environ['MODEL_VERSION'] = model_version
from python_code.model.tpm_wrapper import TPM

# --- Load data --- #
columns_list = ['sequence_alignment', 'ancestor_alignment', 'mutations_all']
all_sets = pd.read_csv('data/final_sets.csv')
paths = all_sets[all_sets.sample_id != 'P4_I19_S1'].path
dataset = load_multiple_sets(paths, columns_list)

# Filter sequences with too many mutations
dataset = dataset[dataset.mutations_all.apply(len) < 9]
dataset.ancestor_alignment = dataset.ancestor_alignment.str.replace('.', 'N')

# Take 10%
dataset = dataset[~(dataset.index.values % 10).astype(bool)]

# --- Load model --- #
model = TPM(params_path='results/model/tpm/v3/61_88/14_04_2024-00_05_22v_gene_family-all-study-all-v_only-synonymous-no_mmr-90%/state_dict_48700')

# --- Run simulation & Save simulated data --- #
tqdm.pandas()
dataset['simulated_sequence'] = dataset.progress_apply(lambda row: simulation(sequence=row.ancestor_alignment,
                                                                              n_mutations=len(row.mutations_all),
                                                                              model=model), axis=1)
save_path = 'results/simulation/tpm/v3/61_88_simulated_data.csv'
dataset.to_csv(save_path, index=False)
