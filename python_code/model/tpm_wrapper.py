import numpy as np
import pandas as pd
import torch

from python_code.model.model import TwoPhaseModel


class TPM:
    def __init__(self):
        self.model = TwoPhaseModel()
        params_path = 'results/model/tpm/vocab_size_100/27_11_2023-17_19_16v_gene_family-all-study-influenza-v_only-synonymous-no_mmr/state_dict_51100'
        params = torch.load(params_path)
        self.model.load_state_dict(params)


    def predict(self, sequence):
        return self.model(sequence).detach().numpy()
