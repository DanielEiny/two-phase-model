import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import scipy


fig_kw = {'figsize': (19.20, 10.80),
          'dpi': 200}

def get_date(file_path):
    timestamp = os.stat(file_path).st_ctime
    date = datetime.fromtimestamp(timestamp)
    return date

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


def generate_plots(log_dir, gt_params_pkl, savefig=False):

    # Read gt params
    with open(gt_params_pkl, 'rb') as f:
        gt_params = pickle.load(f)

    param_keys = list(gt_params.keys())
    param_keys.remove('phase2.mmr.motifs_prob')

    # Read est params
    dates = []
    kl_divs = {k: [] for k in param_keys}

    state_dict_list = glob.glob(os.path.join(log_dir, 'state_dict_*'))
    plot_limit = -1
    step = 10
    for sd_file in state_dict_list[:plot_limit:step]:
        dates.append(get_date(sd_file))
        est_params = torch.load(sd_file)
        for k in param_keys:
            # to_compare = (est_params[k].numpy(), gt_params[k])
            to_compare = (est_params[k].numpy(), gt_params[k].numpy())
            kl_divs[k].append(relative_entropy(*to_compare).sum())

    idx = [int(x.split('/')[-1].split('_')[-1]) for x in state_dict_list[:plot_limit:step]]

    # Plot convergence curves
    fig, ax = plt.subplots(**fig_kw)
    for k in param_keys:
        label = f'parameter group: {k}, number of params: {gt_params[k].shape[0]}'
        ax.plot(idx, kl_divs[k], label=label)
        
    ax.legend()
    fig.suptitle(f'{os.path.basename(log_dir)} | parameter estimation convergence')

    if savefig:
        fig.savefig(f'{log_dir}_convergence.png')
    else:
        plt.show(block=True)


    # Plot profile comparison
    gt_prof = gt_params['phase2.lp_ber.profile']
    est_prof = est_params['phase2.lp_ber.profile']

    fig, axes = plt.subplots(2, 1, **fig_kw)
    axes[0].bar(range(31), est_prof)
    axes[1].bar(range(31), gt_prof)
    fig.suptitle(f'{os.path.basename(log_dir)} | estimated lp_per profile')

    if savefig:
        fig.savefig(f'{log_dir}_profile.png')
    else:
        plt.show(block=True)


if __name__ == '__main__':
    base_dir = 'results/model/convergence_test_by_mutations_freq_fix_aid/'

    log_dirs = [
                '07_11_2023-19_54_05_model_version-merged_vocab_size_100_n_sequences-1000000/',
               ]

    gt_params_pkls = [
                      'original_parameters_model_version-merged_vocab_size_100_n_sequences-1000000.pkl', 
                     ]

    for lg, gpp in zip(log_dirs, gt_params_pkls):
        print(lg)
        generate_plots(os.path.join(base_dir, lg),
                       os.path.join(base_dir, gpp),
                       True)

