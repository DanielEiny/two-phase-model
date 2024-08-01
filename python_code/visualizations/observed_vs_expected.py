import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from python_code.definitions import imgt_regions

data_source = '10%' 

# Load results
with open(f'results/comparison/10%_data.pkl', 'rb') as f:
    results = pickle.load(f)

# Normalization
for model_name in results.keys():
    results[model_name] /= results[model_name].sum()

fig, ax = plt.subplots()

# Plot frequency 
ax.plot(results['observed'], label='observed mutation rate')
for model_name in results.keys():
    ax.plot(results[model_name], label=f'expected mutation rate: {model_name}')
ax.legend()

# Print Pearson correlation 
correlation = {mn: pearsonr(results['observed'], results[mn]).correlation for mn in results.keys()}
correlations_str = [f'{mn}: {correlation[mn]}' for mn in results.keys()]
correlations_str = '\n'.join(['Pearson correlation:'] + correlations_str)
ax.text(0.05, 0.85, correlations_str, transform=ax.transAxes)

plt.show()

