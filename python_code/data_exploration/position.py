

d = dict(zip(nucleotides_with_N, range(5)))
count = data.ancestor_alignment.apply(lambda seq: np.array([d[x] for x in seq]))
count = np.stack(count.values)
C = (count == 0).sum(axis=0)
G = (count == 1).sum(axis=0)
A = (count == 2).sum(axis=0)
T = (count == 3).sum(axis=0)
N = (count == 4).sum(axis=0)
enrichment = pd.DataFrame({'A': A, 'C': C, 'G':G, 'T': T, 'N': N})


dont = enrichment.apply(lambda row: enrichment.columns[row.argmax()] if row.max() < 2000000 else '%', axis=1)
where = data.ancestor_alignment.apply(lambda seq: not any([seq[i] == l for (i, l) in dont.iteritems()]))
