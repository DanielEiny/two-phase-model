import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imageio
import glob
import os

from python_code.data_utils.utils import to_numpy

def create_gif():
    old_pngs = glob.glob('results/model/convergence_gif/*.png')
    [os.remove(x) for x in old_pngs]

    run_id = '14_04_2024-00_04_53_model_version-v3_18_19_n_sequences-100000'
    df = pd.read_csv(f'results/model/convergence_test_v3/{run_id}/log.csv')
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

    files = sorted(glob.glob('results/model/convergence_gif/*.png'))
    # with imageio.get_writer(f'results/model/convergence_gif/{run_id}.gif', mode='I') as writer:
    #     for filename in files:
    #         im = imageio.imread(filename)
    #         writer.append_data(im)
    frames = []
    for f in files:
        frames.append(imageio.read(f).get_data(0)[..., :3])

    # Save them as frames into a gif 
    imageio.mimsave(f'results/model/convergence_gif/{run_id}.gif', frames, 'GIF', loop=1, fps=2)

create_gif()
