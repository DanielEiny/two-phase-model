import re


def calc_motif_mutability(dataset, motif, motif_anchor, substitution=[]):
    # --- Carefulness ---
    dataset = dataset.copy()

    # --- Find mofif occurences ---
    regex_motif = re.compile(motif)
    motif_anchors = dataset.ancestor_alignment.apply(regex_motif.finditer)
    motif_anchors = motif_anchors.apply(lambda x: [match.start() + motif_anchor for match in x]) 

    # --- Sum occurences ---
    dataset['motif_anchors'] = motif_anchors
    motif_count = dataset.motif_anchors.apply(len).sum()
    
    # --- Intersect with mutations --- 
    if substitution:  # optionally, filter by substitution nucleotides
        dataset.mutations_all = dataset.apply(lambda row: [pos for pos in row.mutations_all if row.sequence_alignment[pos] in substitution], axis=1)
    mutation_count = dataset.apply(lambda row: len(set.intersection(set(row.motif_anchors),
                                                                    set(row.mutations_all))), axis=1).sum()
    ratio = mutation_count / motif_count
    return motif_count, mutation_count, ratio

