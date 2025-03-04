import os
import glob
import pickle
import numpy as np
from datetime import datetime
import torch
import scipy


def relative_entropy(p, q):
    if len(p) == len(q) == 1:
        p = np.array([p, 1-p])
        q = np.array([q, 1-q])
    else:
        zeros = q == 0
        p = p[~zeros]
        p = p / p.sum()
        q = q[~zeros]
        q = q / q.sum()
    return scipy.special.rel_entr(p, q)


def read_and_save(log_dir, gt_params_pkl):

    data = {'gt': {}, 'est': {}}

    # Read gt params
    with open(gt_params_pkl, 'rb') as f:
        gt_params = pickle.load(f)

    gt_params.pop('phase2.mmr.motifs_prob')
    param_keys = list(gt_params.keys())

    for k in param_keys:
        data['gt'][k] = gt_params[k].numpy()

    # Read est params
    state_dict_list = glob.glob(os.path.join(log_dir, 'state_dict_*'))
    sd_file = state_dict_list[-1]
    est_params = torch.load(sd_file)
    for k in param_keys:
        data['est'][k] = est_params[k].numpy()

    # Save data
    model_version = log_dir.split('-')[2].split('_n')[0]
    n_sequences = log_dir.split('-')[-1]

    data['model_version'] = model_version
    data['n_sequences'] = n_sequences

    data_dir = 'final_data'
    save_path = os.path.join(os.path.dirname(gt_params_pkl), data_dir, model_version + '+' + n_sequences + '.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    base_dir = 'results/model/convergence_test_v3'

    log_dirs = [
        '03_03_2025-15_37_06_model_version-v3_9_9_n_sequences-10000',
        '03_03_2025-15_43_43_model_version-v3_18_19_n_sequences-10000',
        '03_03_2025-15_53_18_model_version-v3_61_88_n_sequences-100000',
        '03_03_2025-15_53_50_model_version-v3_99_164_n_sequences-100000',
        '03_03_2025-17_55_00_model_version-v3_18_19_n_sequences-1000000',
        '04_03_2025-12_33_50_model_version-v3_61_88_n_sequences-10000',
        '14_04_2024-00_04_53_model_version-v3_18_19_n_sequences-100000',
        '14_04_2024-00_05_30_model_version-v3_9_9_n_sequences-100000',
        '14_04_2024-01_06_09_model_version-v3_61_88_n_sequences-1000000',
        '14_04_2024-13_04_21_model_version-v3_188_486_n_sequences-1000000',
        '14_04_2024-13_17_26_model_version-v3_99_164_n_sequences-1000000',
               ]

    gt_params_pkls = [
        'original_parameters_model_version-v3_9_9_n_sequences-10000.pkl',
        'original_parameters_model_version-v3_18_19_n_sequences-10000.pkl',
        'original_parameters_model_version-v3_61_88_n_sequences-100000.pkl',
        'original_parameters_model_version-v3_99_164_n_sequences-100000.pkl',
        'original_parameters_model_version-v3_18_19_n_sequences-1000000.pkl',
        'original_parameters_model_version-v3_61_88_n_sequences-10000.pkl',
        'original_parameters_model_version-v3_18_19_n_sequences-100000.pkl',
        'original_parameters_model_version-v3_9_9_n_sequences-100000.pkl',
        'original_parameters_model_version-v3_61_88_n_sequences-1000000.pkl',
        'original_parameters_model_version-v3_188_486_n_sequences-1000000.pkl',
        'original_parameters_model_version-v3_99_164_n_sequences-1000000.pkl',
                     ]

    for lg, gpp in zip(log_dirs, gt_params_pkls):
        print(lg)
        read_and_save(os.path.join(base_dir, lg),
                      os.path.join(base_dir, gpp))


