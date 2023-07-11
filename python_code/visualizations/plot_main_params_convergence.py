import matplotlib.pyplot as plt
import pandas as pd

logfile = 'results/model/convergence_test/30_11_2022-23_39_56_model_version-fivemers_n_sequences-100000/log.csv'
df = pd.read_csv(logfile)


plt.plot(df.loss, 'c')
plt.ylim(0., 10.)
ax2 = plt.twinx()
ax2.plot(df['phase2.replication_prob'], 'm')
ax2.plot(df['phase2.short_patch_ber_prob'], 'y')
ax2.set_ylim(0.0, 1.1)

plt.plot()
