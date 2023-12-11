import numpy as np
import pandas as pd

from python_code.definitions import nucleotides


def all_nucleotides(motif):
    return all([x in nucleotides for x in motif])


class S5F:
    def __init__(self):
        df_mutability = pd.read_csv("results/s5f/mutability_all_data.csv").set_index("motif")

        self.mutability = df_mutability.mutability

    def predict(self, sequence):
        assigned_probs = np.zeros(len(sequence))

        for loc in range(2, len(assigned_probs) -2):
            motif = sequence[loc-2:loc+3]
            if all_nucleotides(motif):
                assigned_probs[loc] = self.mutability[motif]

        return assigned_probs / assigned_probs.sum() 
