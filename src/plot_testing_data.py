import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

data = pd.read_csv('data_extraction/output/networks/nocomm_testing.csv', sep='\t')

c = ['r', 'g', 'b', 'm']
error = data.groupby('adversaries')['consensus'].sem() * 1.96
data.groupby('adversaries')['consensus'].mean().plot(kind='bar', yerr=error, color=c)
plt.xticks(rotation='horizontal')
plt.title('ratio over adversaries')
plt.savefig('data_extraction/output/networks/ratio_over_adversaries.png')
plt.close()