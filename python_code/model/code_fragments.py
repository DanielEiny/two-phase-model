
def learning_rate_per_param(model):
    return [{'params': model.phase1.motifs_prob_trainable,        'lr': 1e-2},
            {'params': model.phase2.replication_prob,             'lr': 1e-4},
            {'params': model.phase2.short_patch_ber_prob,         'lr': 1e-4},
            {'params': model.phase2.lp_ber.profile,               'lr': 1e-4},
            {'params': model.phase2.lp_ber.motifs_prob_trainable, 'lr': 1e-2}]

optimizer = torch.optim.Adam(learning_rate_per_param(model))









# --------------------------------------------------------------------------------- #
# Penalty coefficients for probability domain and probability sum
pc_pd_aid_motifs =      1
pc_ps_aid_motifs =      100
pc_pd_replication =     1
pc_pd_short_patch_ber = 1
pc_pd_lp_ber_profile =  100
pc_ps_lp_ber_profile =  100
pc_pd_lp_ber_motifs =   100
pc_ps_lp_ber_motifs =   100
pc_pd_ung =             1
pc_pd_mmr_motifs =      1
pc_ps_mmr_motifs =      1


# Impose parameter space constrains using penalty
loss = loss + pc_pd_aid_motifs * probability_domain_penalty(model.phase1.motifs_prob)
loss = loss + pc_ps_aid_motifs * probability_sum_penalty(model.phase1.motifs_prob)
loss = loss + pc_pd_replication * probability_domain_penalty(model.phase2.replication_prob)
loss = loss + pc_pd_ung * probability_domain_penalty(model.phase2.ung_prob)
loss = loss + pc_pd_short_patch_ber * probability_domain_penalty(model.phase2.short_patch_ber_prob)
loss = loss + pc_pd_lp_ber_profile * probability_domain_penalty(model.phase2.lp_ber.profile)
loss = loss + pc_ps_lp_ber_profile * probability_sum_penalty(model.phase2.lp_ber.profile)
loss = loss + pc_pd_lp_ber_motifs * probability_domain_penalty(model.phase2.lp_ber.motifs_prob)
loss = loss + pc_ps_lp_ber_motifs * probability_sum_penalty(model.phase2.lp_ber.motifs_prob)
loss = loss + pc_pd_mmr_motifs * probability_domain_penalty(model.phase2.mmr.motifs_prob)
loss = loss + pc_ps_mmr_motifs * probability_sum_penalty(model.phase2.mmr.motifs_prob)

print(f'pc_pd_aid_motifs = {probability_domain_penalty(model.phase1.motifs_prob)}')
print(f'pc_ps_aid_motifs = {probability_sum_penalty(model.phase1.motifs_prob)}')
print(f'pc_pd_replication = {probability_domain_penalty(model.phase2.replication_prob)}')
print(f'pc_pd_ung = {probability_domain_penalty(model.phase2.ung_prob)}')
print(f'pc_pd_short_patch_ber = {probability_domain_penalty(model.phase2.short_patch_ber_prob)}')
print(f'pc_pd_lp_ber_profile = {probability_domain_penalty(model.phase2.lp_ber.profile)}')
print(f'pc_ps_lp_ber_profile = {probability_sum_penalty(model.phase2.lp_ber.profile)}')
print(f'pc_pd_lp_ber_motifs = {probability_domain_penalty(model.phase2.lp_ber.motifs_prob)}')
print(f'pc_ps_lp_ber_motifs = {probability_sum_penalty(model.phase2.lp_ber.motifs_prob)}')
print(f'pc_pd_mmr_motifs = {probability_domain_penalty(model.phase2.mmr.motifs_prob)}')
print(f'pc_ps_mmr_motifs = {probability_sum_penalty(model.phase2.mmr.motifs_prob)}')

