import os
import regex as re
import glob
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from python_code.model.model_utils import motif_ambiguity_to_regex
from python_code.model.model_utils import assign_motif_probs
from python_code.model.model_utils import normalize
from python_code.definitions import nucleotides


torch.set_default_dtype(torch.float64)

MODEL_VERSION = os.environ['MODEL_VERSION']  # 'fivemers' or 'simple'

if MODEL_VERSION == 'simple':
    motifs_and_anchors_aid = {'C': 0, 'WRC': 2, 'SYC': 2, 'G': 0, 'GYW': 0, 'GRS' : 0}
    motifs_and_anchors_lp_ber = {'C': 0, 'G': 0, 'A': 0, 'T' : 0, 'WA': 1, 'TW': 1}
    motifs_and_anchors_mmr = {'C': 0, 'G': 0, 'A': 0, 'T' : 0}

elif MODEL_VERSION == 'fivemers':
    anchor_pos = 2
    motifs = pd.read_csv('results/motifs/mutability/fivmers-mutability-no-N.csv')
    # NOTE - in this csv the order is AGCT
    motifs = motifs[motifs.motif.apply(lambda x: x[anchor_pos] != 'N')]

    motifs_and_anchors = {m: anchor_pos for m in motifs.motif.values.tolist()}
    motifs_and_anchors_lp_ber = motifs_and_anchors
    motifs_and_anchors_mmr = motifs_and_anchors

    motifs = motifs[motifs.motif.apply(lambda x: x[anchor_pos] not in ['A', 'T'])]
    motifs_and_anchors = {m: anchor_pos for m in motifs.motif.values.tolist()}
    motifs_and_anchors_aid = motifs_and_anchors


elif MODEL_VERSION.count('merged_vocab'):
    anchor_pos = 2
    vocab_size = int(MODEL_VERSION.split('size_')[1])

    phase2_vocab_size = int(vocab_size / 2)
    vocab_csv_path = glob.glob(f'results/motifs/merged_vocabularies_by_mutations_freq/*_size_{phase2_vocab_size}.csv')[0]
    motifs = pd.read_csv(vocab_csv_path)

    motifs_and_anchors = {m: anchor_pos for m in motifs.motif.values.tolist()}
    motifs_and_anchors_lp_ber = motifs_and_anchors
    motifs_and_anchors_mmr = motifs_and_anchors

    vocab_csv_path = glob.glob(f'results/motifs/merged_vocabularies_by_mutations_freq_only_CG/*_size_{vocab_size}.csv')[0]
    motifs = pd.read_csv(vocab_csv_path)

    motifs_and_anchors_aid = {m: anchor_pos for m in motifs.motif.values.tolist()}

class MisMatchRepair(nn.Module):
    def __init__(self):
        super().__init__()
        self.motifs = motifs_and_anchors_mmr.keys()
        self.motifs_anchor = motifs_and_anchors_mmr
        self.motifs_regex = {m: re.compile(motif_ambiguity_to_regex(m)) for m in self.motifs}
        self.motifs_prob = nn.Parameter(normalize(torch.ones(len(self.motifs))))
        self.motifs_idx = {m: i for m, i in zip(self.motifs, range(len(self.motifs)))}
        
    def forward(self, sequence, mmr_centers_probs):
        mmr_motif_probs = assign_motif_probs(sequence, 
                                             self.motifs, 
                                             self.motifs_anchor, 
                                             self.motifs_regex, 
                                             self.motifs_idx, 
                                             self.motifs_prob)
        mmr_motif_probs = normalize(mmr_motif_probs)
        return mmr_motif_probs * mmr_centers_probs.sum()


class LongPatchBer(nn.Module):
    def __init__(self):
        super().__init__()
        self.profile = nn.Parameter(normalize(torch.ones(31))) 

        self.motifs = motifs_and_anchors_lp_ber.keys()
        self.motifs_anchor = motifs_and_anchors_lp_ber
        self.motifs_regex = {m: re.compile(motif_ambiguity_to_regex(m)) for m in self.motifs}
        self.motifs_prob = nn.Parameter(normalize(torch.ones(len(self.motifs))))
        self.motifs_idx = {m: i for m, i in zip(self.motifs, range(len(self.motifs)))}

        self.forward = self.forward_vectorized

    def forward_loop(self, sequence, lp_ber_centers_probs):
        lp_ber_motif_probs = assign_motif_probs(sequence, 
                                                self.motifs, 
                                                self.motifs_anchor, 
                                                self.motifs_regex, 
                                                self.motifs_idx, 
                                                self.motifs_prob)

        sequence_len = len(sequence)
        lp_ber_targets_probs = torch.zeros(sequence_len)

        profile = self.profile.unsqueeze(0).unsqueeze(0)

        for position in range(sequence_len):
            if lp_ber_centers_probs[position] == 0:
                continue
            specific_center = torch.zeros(sequence_len)
            specific_center[position] = 1.0
            specific_center = specific_center.unsqueeze(0).unsqueeze(0)
            specific_center_profile = F.conv1d(specific_center, profile, padding='same')
            specific_center_profile = specific_center_profile.squeeze()
            specific_center_targets_prob_accounting_motifs = specific_center_profile * lp_ber_motif_probs
            specific_center_targets_prob_accounting_motifs = normalize(specific_center_targets_prob_accounting_motifs)
            specific_center_targets_prob_accounting_motifs_and_center_prob = specific_center_targets_prob_accounting_motifs * lp_ber_centers_probs[position]

            lp_ber_targets_probs = lp_ber_targets_probs + specific_center_targets_prob_accounting_motifs_and_center_prob
            
        return lp_ber_targets_probs

    def forward_vectorized(self, sequence, lp_ber_centers_probs):
        lp_ber_motif_probs = assign_motif_probs(sequence, 
                                                self.motifs, 
                                                self.motifs_anchor, 
                                                self.motifs_regex, 
                                                self.motifs_idx, 
                                                self.motifs_prob)

        sequence_len = len(sequence)

        lp_ber_center_pos = lp_ber_centers_probs > 0
        centers = torch.eye(sequence_len)[lp_ber_center_pos].unsqueeze(1)
        profile = self.profile.unsqueeze(0).unsqueeze(0)
        centers_profiles = F.conv1d(centers, profile, padding='same').squeeze()
        
        centers_profiles_accounting_motifs = centers_profiles * lp_ber_motif_probs
        centers_profiles_accounting_motifs = normalize(centers_profiles_accounting_motifs, dim=1)

        lp_ber_targets_probs = torch.matmul(lp_ber_centers_probs[lp_ber_center_pos], 
                                            centers_profiles_accounting_motifs)
        return lp_ber_targets_probs

class Phase1(nn.Module):
    def __init__(self):
        super().__init__()
        self.motifs = motifs_and_anchors_aid.keys()
        self.motifs_anchor = motifs_and_anchors_aid
        self.motifs_regex = {m: re.compile(motif_ambiguity_to_regex(m)) for m in self.motifs}
        self.motifs_prob = nn.Parameter(normalize(torch.ones(len(self.motifs))))
        self.motifs_idx = {m: i for m, i in zip(self.motifs, range(len(self.motifs)))}

    def forward(self, sequence):
        # Assign probs to motifs
        targeting_probs = assign_motif_probs(sequence, 
                                             self.motifs, 
                                             self.motifs_anchor, 
                                             self.motifs_regex, 
                                             self.motifs_idx, 
                                             self.motifs_prob)
        # Allow targting of only C/G
        not_c_or_g = [i for i, x in enumerate(sequence) if x not in ['C', 'G']]
        targeting_probs[not_c_or_g] = 0.0

        # Normalize
        targeting_probs = normalize(targeting_probs)
        return targeting_probs

class Phase2(nn.Module):
    def __init__(self):
        super().__init__()
        self.lp_ber = LongPatchBer()
        self.mmr = MisMatchRepair()

        self.replication_prob = nn.Parameter(torch.tensor([.5]))
        self.ung_prob = nn.Parameter(torch.tensor([1.]), requires_grad=False)
        self.short_patch_ber_prob = nn.Parameter(torch.tensor([.5]))

    def forward(self, sequence, targeting_probs_phase1):
        replication_probs = targeting_probs_phase1 * self.replication_prob
        error_prone_repair_probs = targeting_probs_phase1 * (1 - self.replication_prob) 

        ung_probs = error_prone_repair_probs * self.ung_prob
        short_patch_ber_probs = ung_probs * self.short_patch_ber_prob
        long_patch_ber_probs = self.lp_ber(sequence, ung_probs * (1 - self.short_patch_ber_prob))

        mmr_probs = self.mmr(sequence, error_prone_repair_probs * (1 - self.ung_prob))

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
