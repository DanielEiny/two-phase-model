from python_code.data_exploration.motifs import calc_ambiguity_codes





calc_ambiguity_codes('results/t_motifs_mutability.csv',
                     'results/t_ambiguous_motifs_mutability.csv',
                     anchor_nucleotide='T', 
                     positions_left=2, 
                     positions_right=2)
