import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from python_code.definitions import imgt_regions


# Load results
path = 'results/comparison/tmp_dev/v3_61_88_-10%_of_train_data_only_v.pkl'
path = 'results/comparison/tmp_dev_new_method/v3_61_88_-10%_of_train_data_only_v.pkl'

with open(path, 'rb') as f:
    results = pickle.load(f)

# Normalization
for model_name in results.keys():
    results[model_name] /= results[model_name].sum()

fig, ax = plt.subplots(figsize=(16, 10))

# Plot frequency 
for model_name in results.keys():
    if model_name == 'observed':
        label = 'observed mutation rate'
    else:
        label = f'expected mutation rate: {model_name}'
    ax.plot(results[model_name], label=label)
ax.legend()

# Print Pearson correlation 
correlation = {mn: pearsonr(results['observed'], results[mn]).correlation for mn in results.keys()}
correlations_str = [f'{mn}: {correlation[mn]}' for mn in results.keys()]
correlations_str = '\n'.join(['Pearson correlation:'] + correlations_str)
ax.text(0.05, 0.55, correlations_str, transform=ax.transAxes)

# Add title
ax.set_title(path)

plt.show()
# plt.savefig(path.replace('pkl', 'png'), bbox_inches='tight')

