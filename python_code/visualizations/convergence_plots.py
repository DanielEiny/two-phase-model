import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import scipy


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


base_path = 'results/model/convergence_test/'
log_dirs = glob.glob(os.path.join(base_path, '*/'))
current_log_dir = log_dirs[0]

with open(os.path.join(current_log_dir, 'params_gt.pkl'), 'rb') as f:
    gt_params = pickle.load(f)
param_keys = gt_params.keys()

dates = []
kl_divs = {k: [] for k in param_keys}

state_dict_list = glob.glob(os.path.join(current_log_dir, 'state_dict_*'))
plot_limit = 100
for sd_file in state_dict_list[:plot_limit]:
    dates.append(get_date(sd_file))
    est_params = torch.load(sd_file)
    for k in param_keys:
        to_compare = (est_params[k].numpy(), gt_params[k])
        kl_divs[k].append(relative_entropy(*to_compare).sum())

idx = [int(x.split('/')[-1].split('_')[-1]) for x in state_dict_list]

fig, ax = plt.subplots()
for k in param_keys:
    label = f'parameter group: {k}, number of params: {gt_params[k].shape[0]}'
    ax.plot(idx[:plot_limit], kl_divs[k], label=label)
    
ax.legend()
ax.set_title(current_log_dir)
plt.show(block=False)
