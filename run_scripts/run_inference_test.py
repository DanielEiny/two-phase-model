import pandas as pd

from python_code.data_utils.utils import load_multiple_sets
from python_code.definitions import imgt_regions
from python_code.model.model import TwoPhaseModel
from python_code.model.inference import inference


# --- Load data --- #
dataset = pd.read_csv('results/tmp_many_lp_ber_motifs.csv')

# --- Filter sequences with too many mutations --- #
#dataset = dataset[dataset.mutations_all.apply(len) < 9]

# --- Filter by V gene family --- #
#v_gene_family = 'IGHV' + '3'
#dataset = dataset[dataset.germline_v_call.apply(lambda x: x.split('-')[0] == v_gene_family)]

# --- Look only on V gene --- #
#v_gene_end = imgt_regions['FR3'][1]
#dataset.simulated_sequence = dataset.simulated_sequence.apply(lambda x: x[:v_gene_end])
#dataset.ancestor_alignment = dataset.ancestor_alignment.apply(lambda x: x[:v_gene_end])

n_sequences = dataset.shape[0]
print(f'Start inference! total number of sequences: {n_sequences}')

# --- Init model --- #
tpm = TwoPhaseModel()

# --- Infer model params --- #
inference(tpm, 
          dataset, 
          ancestor_column='ancestor_alignment', 
          descendant_column='simulated_sequence', 
          only_synonymous=True,
          log_postfix=f'_inference_test_many_lp_ber_motifs')
