nucleotides = ['C', 'G', 'A', 'T']
nucleotides_with_N = ['C', 'G', 'A', 'T', 'N']

uipac_ambiguity_codes = {'A': 'A',
                         'G': 'G',
                         'C': 'C',
                         'T': 'T',
                         'M': 'A|C',   
                         'R': 'A|G',
                         'W': 'A|T',
                         'S': 'C|G',
                         'Y': 'C|T',
                         'K': 'G|T',
                         'V': 'A|C|G',
                         'H': 'A|C|T',
                         'D': 'A|G|T',
                         'B': 'C|G|T',
                         'N': 'G|A|T|C'}


codon_table = {'TTT': 'Phe', 'TTC': 'Phe', 'TTA': 'Leu', 'TTG': 'Leu',
               'TCT': 'Ser', 'TCC': 'Ser', 'TCA': 'Ser', 'TCG': 'Ser',
               'TAT': 'Tyr', 'TAC': 'Tyr', 'TAA': 'STOP', 'TAG': 'STOP',
               'TGT': 'Cyc', 'TGC': 'Cyc', 'TGA': 'STOP', 'TGG': 'Trp',
               'CTT': 'Leu', 'CTC': 'Leu', 'CTA': 'Leu', 'CTG': 'Leu',
               'CCT': 'Pro', 'CCC': 'Pro', 'CCA': 'Pro', 'CCG': 'Pro',
               'CAT': 'His', 'CAC': 'His', 'CAA': 'Gln', 'CAG': 'Gln',
               'CGT': 'Arg', 'CGC': 'Arg', 'CGA': 'Arg', 'CGG': 'Arg',
               'ATT': 'Ile', 'ATC': 'Ile', 'ATA': 'Ile', 'ATG': 'Met',
               'ACT': 'Thr', 'ACC': 'Thr', 'ACA': 'Thr', 'ACG': 'Thr',
               'AAT': 'Asn', 'AAC': 'Asn', 'AAA': 'Lis', 'AAG': 'Lis',
               'AGT': 'Ser', 'AGC': 'Ser', 'AGA': 'Arg', 'AGG': 'Arg',
               'GTT': 'Val', 'GTC': 'Val', 'GTA': 'Val', 'GTG': 'Val',
               'GCT': 'Ala', 'GCC': 'Ala', 'GCA': 'Ala', 'GCG': 'Ala',
               'GAT': 'Asp', 'GAC': 'Asp', 'GAA': 'Glu', 'GAG': 'Glu',
               'GGT': 'Gly', 'GGC': 'Gly', 'GGA': 'Gly', 'GGG': 'Gly'}

imgt_regions = {'FR1' : (0, 78),
                'CDR1': (78, 114),
                'FR2' : (114, 165),
                'CDR2': (165, 195),
                'FR3' : (195, 312)}
