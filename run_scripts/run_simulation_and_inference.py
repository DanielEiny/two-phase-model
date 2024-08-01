import os
import sys
import pandas as pd

# --- Set run params --- #
model_version = sys.argv[1]
os.environ['MODEL_VERSION'] = model_version
n_sequences = int(sys.argv[2])

from python_code.data_utils.utils import load_multiple_sets
from python_code.model.simulation_and_inference import simulation_and_inference

# --- Load data --- #
columns_list = ['sequence_alignment', 'ancestor_alignment', 'mutations_all']
all_sets = pd.read_csv('data/final_sets.csv')
#paths = all_sets[all_sets.study == 'influenza'].path
paths = all_sets[all_sets.sample_id != 'P4_I19_S1'].path
#paths = paths[:1]
dataset = load_multiple_sets(paths, columns_list)
dataset = dataset[:n_sequences]
# Filter sequences with too many mutations
dataset = dataset[dataset.mutations_all.apply(len) < 9]
dataset.ancestor_alignment = dataset.ancestor_alignment.str.replace('.', 'N')

simulation_and_inference(dataset, only_synonymous=True, log_postfix=f'_model_version-{model_version}_n_sequences-{n_sequences}')
