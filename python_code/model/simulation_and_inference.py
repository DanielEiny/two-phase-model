import os
import torch
import pandas as pd
from tqdm import tqdm

from python_code.model.model import TwoPhaseModel
from python_code.model.simulation import simulation
from python_code.model.inference import inference
from python_code.model.model_utils import normalize, quasi_random_fivemer_probs, randomize_and_save_params


MODEL_VERSION = os.environ['MODEL_VERSION']  # 'fivemers' or 'simple'
log_path = 'results/model/convergence_test_by_mutations_freq_fix_aid_remove_ambiguous_correct_vocab/'
log_path = 'results/model/convergence_test_v3/'
log_path = 'results/model/convergence_test_recap/'
# log_path = 'results/model/tpm_tmp_dev/'


def simulation_and_inference(dataset, only_synonymous=False, log_postfix=''):
    # --- Init model --- #
    tpm = TwoPhaseModel()
    
    # --- Set model params --- #
    parameters = tpm.state_dict()

    if MODEL_VERSION == 'simple':
        parameters['phase1.motifs_prob'] = torch.tensor([0.1, 0.4, 0.05, 0.1, 0.3, 0.05])
        parameters['phase2.replication_prob'] = torch.tensor([0.35])
        parameters['phase2.short_patch_ber_prob'] = torch.tensor([0.2])
        parameters['phase2.lp_ber.profile'] = torch.concat([torch.zeros(11), torch.ones(9) / 9, torch.zeros(11)])
        parameters['phase2.lp_ber.motifs_prob'] = torch.tensor([1e-19, 1e-19, 0.15, 0.20, 0.30, 0.35])

    elif MODEL_VERSION == 'fivemers':
        parameters['phase1.motifs_prob'] = normalize(torch.concat([torch.zeros(625), torch.ones(1250), torch.zeros(625)]))
        parameters['phase1.motifs_prob'] = quasi_random_fivemer_probs(os.path.join(log_path, log_postfix[1:] + '_phase1_motif_probs.npy'), ignore=['A', 'T'])
        parameters['phase2.replication_prob'] = torch.tensor([0.35])
        parameters['phase2.short_patch_ber_prob'] = torch.tensor([0.2])
        parameters['phase2.lp_ber.profile'] = normalize(torch.concat([torch.zeros(11), torch.ones(9), torch.zeros(11)]))
        parameters['phase2.lp_ber.motifs_prob'] = normalize(torch.concat([torch.arange(625), torch.zeros(1250) + 1, torch.arange(625)]))
        parameters['phase2.lp_ber.motifs_prob'] = quasi_random_fivemer_probs(os.path.join(log_path, log_postfix[1:] + '_phase2_lp_ber_motifs_prob.npy'))

    elif MODEL_VERSION.count('merged_vocab') or MODEL_VERSION.count('v3'):
        save_path = os.path.join(log_path, f'original_parameters_{log_postfix[1:]}.pkl')
        randomize_and_save_params(parameters, save_path=save_path)

    tpm.load_state_dict(parameters)
    
    # --- Simulate mutations --- #
    tqdm.pandas()
    dataset['simulated_sequence'] = dataset.progress_apply(lambda row: simulation(sequence=row.ancestor_alignment,
                                                                                  n_mutations=len(row.mutations_all),
                                                                                  model=tpm), axis=1)
    #dataset.to_csv('results/tmp_fivemers_all_data.csv', index=False)
    #dataset = pd.read_csv('results/tmp_fivemers_all_data.csv')

    # --- Reset model params --- #
    tpm = TwoPhaseModel()
    #state_dict_path = 'results/model/log/13_07_2022-15:34:59_fivmers_alldata_ADAM_bs50_lr4/state_dict_50000'
    #tpm.load_state_dict(torch.load(state_dict_path))
    
    # --- Infer model params --- #
    inference(tpm, dataset, ancestor_column='ancestor_alignment', descendant_column='simulated_sequence', only_synonymous=only_synonymous, log_postfix=log_postfix, max_iter=500000)
