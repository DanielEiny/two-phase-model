import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch


study = ['influenza', 'mg', 'covid']
v_gene = ['IGHV1', 'IGHV3', 'IGHV4']

source = study
param = 'phase2.lp_ber.profile'
profiles = {}

fig_kw = {'figsize': (19.20, 10.80),
          'dpi': 200}

fig, axes = plt.subplots(len(source), 1, **fig_kw)
fig.tight_layout(pad=10.0)
for i, key in enumerate(source):
    path = glob.glob(f'results/model/tpm/vocab_size_100/*{key}*/state_dict_30000')[0]
    profiles[key] = torch.load(path)[param]
    axes[i].bar(range(len(profiles[key])), profiles[key])
    axes[i].set_title(key)

plt.show()
