import matplotlib.pyplot as plt
import pandas as pd

logfile = 'results/model/log/28_07_2022-10:41:26_fivmers_alldata_ADAM_bs50_lr4/log.csv'
logfile = 'results/model/log/29_07_2022-12:07:37_fivmers_alldata_ADAM_bs50_lr3/log.csv'
df = pd.read_csv(logfile)


plt.plot(df.loss, 'c')
plt.ylim(0., 10.)
ax2 = plt.twinx()
ax2.plot(df['phase2.replication_prob'], 'm')
ax2.plot(df['phase2.short_patch_ber_prob'], 'y')
ax2.set_ylim(0.0, 1.1)

plt.plot()
