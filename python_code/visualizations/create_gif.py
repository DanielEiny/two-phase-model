import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imageio
import glob

from python_code.data_utils.utils import to_numpy

def create_gif():
    run_id = '11_11_2022-00:02:33_simple_alldata_ADAM_bs50_lr3'
    df = pd.read_csv(f'results/model/log/{run_id}/log.csv')
    profile = df['phase2.lp_ber.profile'].apply(to_numpy)

    step = 1000
    for i, p in enumerate(profile[::step]):
        print(f'{i} --> {i * step}')
        plt.figure(figsize=(16, 9))
        plt.bar(range(31), p)
        plt.xticks(range(31))
        plt.yticks(np.arange(0, .41, .02))
        plt.title('Inference convergence: long-patch BER profile')
        plt.text(0, 0.34, f'step = {i * step}', fontsize=15)
        plt.savefig('results/model/convergence_gif/' + str(i).zfill(4) + '.png')
        plt.close()

    files = glob.glob('results/model/convergence_gif/*.png')
    with imageio.get_writer(f'results/model/convergence_gif/{run_id}.gif', mode='I') as writer:
        for filename in files:
            im = imageio.imread(filename)
            writer.append_data(im)

