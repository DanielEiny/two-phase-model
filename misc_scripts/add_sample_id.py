import pandas as pd

final_sets = pd.read_csv('data/final_sets.csv')
final_sets = final_sets[final_sets.sample_id != 'P4_I19_S1']
final_sets  = final_sets [final_sets .sample_id == 'P3_I2_S2']


for _, row in final_sets.iterrows():
    sample_id = row.sample_id
    print(sample_id)
    idx = 0

    feather_path = row.path.replace('tsv', 'feather')
    feather = pd.read_feather(feather_path)
    feather.insert(loc=idx, column='sample_id', value=sample_id)
    feather.to_feather(feather_path)
    
    tsv_path = row.path
    tsv = feather.drop(columns=['mutations_all', 'mutations_synonymous', 'mutations_full_synonymous'])
    tsv.to_csv(tsv_path, sep='\t', index=False)

    

    

