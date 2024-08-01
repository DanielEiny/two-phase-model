import numpy as np
import pandas as pd
import torch

from python_code.model.model import TwoPhaseModel


DEFAULT_PARAMS = 'results/model/tpm/vocab_size_100/04_02_2024-21_17_57v_gene_family-all-study-all-v_only-synonymous-no_mmr/state_dict_30000'

class TPM:
    def __init__(self, params_path=DEFAULT_PARAMS):
        self.model = TwoPhaseModel()
        params_path = params_path
        params = torch.load(params_path)
        self.model.load_state_dict(params)


    def predict(self, sequence):
        targeting_probs = self.model(sequence)[0]
        return targeting_probs.detach().numpy()
