import re
import torch
from torch import nn


class Phase1(nn.Module):
    def __init__(self):
        super().__init__()
        self.motif = re.compile('C')

    def forward(self, sequence):
        targeting_probs = torch.zeros(len(sequence))
        positions = [m.start() for m in self.motif.finditer(sequence)]
        targeting_probs[positions] = 1
        targeting_probs = targeting_probs / targeting_probs.sum()
        return targeting_probs


class Phase2(nn.Module):
    def __init__(self):
        super().__init__()
        self.replication_prob = nn.Parameter(torch.rand(1))
        self.ung_prob = nn.Parameter(torch.tensor([1.]), requires_grad=False)  # torch.rand(1))
        self.mmr_prob = nn.Parameter(1 - self.ung_prob)
        self.short_patch_ber_prob = nn.Parameter(torch.tensor([1.]), requires_grad=False)  # torch.rand(1))
        self.long_patch_ber_prob = nn.Parameter(1 - self.short_patch_ber_prob)

    def forward(self, sequence, targeting_probs_phase1):
        replication_probs = targeting_probs_phase1 * self.replication_prob
        error_prone_repair_probs = targeting_probs_phase1 * (1 - self.replication_prob) 
        ung_probs = error_prone_repair_probs * self.ung_prob
        mmr_probs = error_prone_repair_probs * self.mmr_prob
        short_patch_ber_probs = ung_probs * self.short_patch_ber_prob
        long_patch_ber_probs = ung_probs * self.long_patch_ber_prob

        return torch.vstack([replication_probs, short_patch_ber_probs])


class TwoPhaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.phase1 = Phase1()
        self.phase2 = Phase2()

    def forward(self, sequence):
        targeting_probs_phase1 = self.phase1(sequence)
        targeting_probs_phase2 = self.phase2(sequence, targeting_probs_phase1)
        return targeting_probs_phase2
