import pandas as pd

from python_code.data_utils.utils import load_multiple_sets
from python_code.definitions import imgt_regions
from python_code.model.model import TwoPhaseModel
from python_code.model.inference import inference


# --- Load data --- #
columns_list = ['sequence_alignment', 'ancestor_alignment', 'germline_v_call', 'mutations_all', 'mutations_synonymous']
all_sets = pd.read_csv('data/final_sets.csv')
paths = all_sets[all_sets.sample_id != 'P4_I19_S1'].path

# Filter by study
#study = 'covid'
study = 'all'
if study != 'all':
    paths = all_sets[all_sets.study == study].path

dataset = load_multiple_sets(paths, columns_list)
dataset.ancestor_alignment = dataset.ancestor_alignment.str.replace('.', 'N')
dataset.descendant_alignment = dataset.ancestor_alignment.str.replace('.', 'N')


# Filter by V gene family
v_gene_family = 'IGHV' + '4'
v_gene_family = 'all'
if v_gene_family != 'all':
    dataset = dataset[dataset.germline_v_call.apply(lambda x: x.split('-')[0] == v_gene_family)]

# Filter sequences with too many mutations
dataset = dataset[dataset.mutations_all.apply(len) < 9]

# Look only on V gene
v_gene_end = imgt_regions['FR3'][1]
dataset.sequence_alignment = dataset.sequence_alignment.apply(lambda x: x[:v_gene_end])
dataset.ancestor_alignment = dataset.ancestor_alignment.apply(lambda x: x[:v_gene_end])

# Take only 90%
dataset = dataset[(dataset.index.values % 10).astype(bool)]

# Print final dataset stats 
n_sequences = dataset.shape[0]
n_mutations = dataset.mutations_synonymous.apply(len).sum()
print(f'Start inference! total number of sequences: {n_sequences}, total number of mutations: {n_mutations}')

# --- Init model --- #
tpm = TwoPhaseModel()

# --- Infer model params --- #
inference(tpm, 
          dataset, 
          ancestor_column='ancestor_alignment', 
          descendant_column='sequence_alignment', 
          only_synonymous=True,
          log_postfix=f'v_gene_family-{v_gene_family}-study-{study}-v_only-synonymous-no_mmr-90%',
          max_iter=50000)
