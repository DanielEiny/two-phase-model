import re
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from python_code.model.model_utils import motif_ambiguity_to_regex


torch.set_default_dtype(torch.float64)

motifs_and_anchors = {'C': 0, 'WRC': 2, 'SYC': 2, 'G': 0, 'GYW': 0, 'GRS' : 0, 'WA': 1, 'TW': 0}
motifs_and_anchors = {'C': 0, 'WRC': 2, 'SYC': 2, 'G': 0, 'GYW': 0, 'GRS' : 0}
#motifs_and_anchors = {'C': 0, 'WRC': 2, 'SYC': 2}


class LongPatchBer(nn.Module):
    def __init__(self):
        super().__init__()
        profile = torch.rand(31)
        profile = profile / profile.sum()
        self.profile = nn.Parameter(profile) 

    def forward(self, lp_ber_centers_probs):
        profile = self.profile.unsqueeze(0).unsqueeze(0)
        lp_ber_centers_probs = lp_ber_centers_probs.unsqueeze(0).unsqueeze(0)
        lp_ber_targets_probs = F.conv1d(lp_ber_centers_probs, profile, padding='same')

        # Handle edges issue (wrongly...)
        if lp_ber_targets_probs.sum() > 0:
            lp_ber_targets_probs = lp_ber_targets_probs / lp_ber_targets_probs.sum()
            lp_ber_targets_probs = lp_ber_targets_probs * lp_ber_centers_probs.sum()
        return lp_ber_targets_probs.squeeze()

class Phase1(nn.Module):
    def __init__(self):
        super().__init__()
        self.motifs = motifs_and_anchors.keys()
        self.motifs_anchor = motifs_and_anchors
        self.motifs_regex = {m: re.compile(motif_ambiguity_to_regex(m)) for m in self.motifs}
        self.motifs_prob = torch.rand(len(self.motifs))
        self.motifs_prob = self.motifs_prob / self.motifs_prob.sum()
        self.motifs_prob = nn.Parameter(self.motifs_prob) 
        self.motifs_idx = {m: i for m, i in zip(self.motifs, range(len(self.motifs)))}

    def forward(self, sequence):
        targeting_probs = torch.zeros(len(sequence))

        for motif in self.motifs:
            offset = self.motifs_anchor[motif]
            regex = self.motifs_regex[motif]
            prob = self.motifs_prob[self.motifs_idx[motif]]
            positions = [match.start() + offset for match in regex.finditer(sequence)]
            targeting_probs[positions] = prob

        # Normalize
        targeting_probs = targeting_probs / targeting_probs.sum()
        return targeting_probs

class Phase2(nn.Module):
    def __init__(self):
        super().__init__()
        self.lp_ber = LongPatchBer()

        self.replication_prob = nn.Parameter(torch.rand(1))
        self.ung_prob = nn.Parameter(torch.tensor([1.]), requires_grad=False)  # torch.rand(1))
        self.short_patch_ber_prob = nn.Parameter(torch.rand(1))

    def forward(self, sequence, targeting_probs_phase1):
        replication_probs = targeting_probs_phase1 * self.replication_prob
        error_prone_repair_probs = targeting_probs_phase1 * (1 - self.replication_prob) 

        ung_probs = error_prone_repair_probs * self.ung_prob
        short_patch_ber_probs = ung_probs * self.short_patch_ber_prob
        long_patch_ber_probs = self.lp_ber(ung_probs * (1 - self.short_patch_ber_prob))

        mmr_probs = error_prone_repair_probs * (1 - self.ung_prob)

        targeting_probs = replication_probs + mmr_probs + short_patch_ber_probs + long_patch_ber_probs
        replication_probs = torch.zeros(len(sequence))
        replication_probs[targeting_probs_phase1 > 0] = self.replication_prob
        return targeting_probs, replication_probs

class TwoPhaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.phase1 = Phase1()
        self.phase2 = Phase2()

    def forward(self, sequence):
        targeting_probs_phase1 = self.phase1(sequence)
        return self.phase2(sequence, targeting_probs_phase1)
