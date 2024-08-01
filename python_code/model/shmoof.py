import numpy as np
import pandas as pd

from python_code.definitions import nucleotides


def all_nucleotides(motif):
    return all([x in nucleotides for x in motif])


class SHMOOF:
    def __init__(self):
        df_context = pd.read_csv("results/shmoof/mutabilities_context.tsv", sep="\t").set_index("Motif")
        df_pos = pd.read_csv("results/shmoof/mutabilities_position.tsv", sep="\t").set_index("Position")

        self.mutability_context = df_context.Mutability
        self.mutability_pos = df_pos.Mutability

    def predict(self, sequence):
        assigned_probs = np.zeros(len(sequence))

        for loc in range(2, len(assigned_probs) - 2):
            motif = sequence[loc-2:loc+3]
            if all_nucleotides(motif):
                assigned_probs[loc] = self.mutability_context[motif] * self.mutability_pos[loc]

        return assigned_probs / assigned_probs.sum() 
