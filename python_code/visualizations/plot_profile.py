import matplotlib.pyplot as plt
import pandas as pd

from python_code.data_utils.utils import to_numpy


logfile = 'results/model/log/28_07_2022-10:41:26_fivmers_alldata_ADAM_bs50_lr4/log.csv'
#logfile = 'results/model/log/29_07_2022-12:07:37_fivmers_alldata_ADAM_bs50_lr3/log.csv'
df = pd.read_csv(logfile)

profile = df['phase2.lp_ber.profile'].apply(to_numpy)

plt.bar(range(31), profile[profile.shape[0] - 1])
plt.plot()
