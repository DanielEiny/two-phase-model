a = pd.read_csv('results/motifs/mutability/a_motifs_mutability.csv')
c = pd.read_csv('results/motifs/mutability/c_motifs_mutability.csv')
g = pd.read_csv('results/motifs/mutability/g_motifs_mutability.csv')
t = pd.read_csv('results/motifs/mutability/t_motifs_mutability.csv')



a['mutability'] = a.mutation_count / a.motif_count
c['mutability'] = c.mutation_count / c.motif_count
g['mutability'] = g.mutation_count / g.motif_count
t['mutability'] = t.mutation_count / t.motif_count


a = a.set_index('motif')
c = c.set_index('motif')
g = g.set_index('motif')
t = t.set_index('motif')



globals_dict = globals()

for i in nucleotides:
    for j in nucleotides:
        if i != j:
            name = f'{i.lower()}_to_{j.lower()}'
            globals_dict[name] = pd.read_csv('results/motifs/substitution/substitution_' + name + '.csv')
            globals_dict[name]['substitution'] = globals_dict[name].mutation_count / globals_dict[name].motif_count
            globals_dict[name] = globals_dict[name].set_index('motif')

