import torch
import pandas as pd

from python_code.model.model import TwoPhaseModel
from python_code.model.simulation import simulation
from python_code.model.inference import inference



def simulation_and_inference(dataset):
    # --- Init model --- #
    tpm = TwoPhaseModel()
    
    # --- Set model params --- #
    parameters = tpm.state_dict()
    parameters['phase1.motifs_prob'] = torch.tensor([0.2, 0.7, 0.1])
    parameters['phase2.replication_prob'] = torch.tensor([0.5])
    tpm.load_state_dict(parameters)
    print(f' *** simulation model params: {tpm.state_dict()["phase2.replication_prob"]} *** ')
    
    # --- Simulate mutations --- #
    dataset['simulated_sequence'] = dataset.apply(lambda row: simulation(sequence=row.ancestor_alignment,
                                                                         n_mutations=len(row.mutations_all),
                                                                         model=tpm), axis=1)
    
    # --- Reset model params --- #
    parameters['phase1.motifs_prob'] = torch.tensor([0.3, 0.3, 0.4])
    parameters['phase2.replication_prob'] = torch.tensor([0.9])
    tpm.load_state_dict(parameters)
    print(f' *** inference initials model params: {tpm.state_dict()["phase2.replication_prob"]} *** ')
    
    
    # --- Infer model params --- #
    inference(tpm, dataset, ancestor_column='ancestor_alignment', descendant_column='simulated_sequence')
    print(f' *** estimated model params: {tpm.state_dict()["phase2.replication_prob"]} *** ')
