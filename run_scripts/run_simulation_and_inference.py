import pandas as pd

from python_code.data_utils.utils import load_multiple_sets
from python_code.model.simulation_and_inference import simulation_and_inference


# --- Load data --- #
columns_list = ['sequence_alignment', 'ancestor_alignment', 'mutations_all']
all_sets = pd.read_csv('data/final_sets.csv')
#paths = all_sets[all_sets.study == 'influenza'].path
paths = all_sets[all_sets.sample_id != 'P4_I19_S1'].path
paths = paths[:1]
dataset = load_multiple_sets(paths, columns_list)
dataset = dataset[:]
dataset.ancestor_alignment = dataset.ancestor_alignment.str.replace('.', 'N')

simulation_and_inference(dataset, only_synonymous=True, log_postfix='_tmp_deleteme')
