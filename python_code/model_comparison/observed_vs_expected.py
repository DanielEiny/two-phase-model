import numpy as np
import pandas as pd



dataset = pd.read_csv('data/shm_oof_french_research/_alignment_updated/316188_alignment_updated.tab', sep='\t')
dataset = dataset[~dataset.ancestorseq.isna()]

