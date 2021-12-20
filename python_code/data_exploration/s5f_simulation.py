import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

from python_code.definitions import imgt_regions
from python_code.utils import flatten_list


def s5f_simulation(germlines, mutability):

    # --- Look only on V gene --- #
    v_gene_end = imgt_regions['FR3'][1]
    germlines = germlines.apply(lambda seq: seq[:v_gene_end])
    germlines = germlines.apply(lambda seq: seq.replace('.', 'N'))
    germlines = germlines.apply(lambda seq: 'NN' + seq + 'NN')
    probs = np.zeros((germlines.shape[0], v_gene_end))
    for i, seq in enumerate(germlines):
        probs[i] = np.array([mutability[seq[x-2:x+3]] for x in range(2, v_gene_end + 2)])
    import ipdb; ipdb.set_trace()
    simulated_data = np.random.binomial(1, probs)
    pass
