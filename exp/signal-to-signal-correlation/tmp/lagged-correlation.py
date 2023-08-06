import sys
import os
import logging
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Set up file paths
file_path = str(Path().absolute())
project_path = str(Path().absolute().parent.parent)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"Current file directory: {file_path}")
logging.info(f"Current project directory: {project_path}")

os.chdir(project_path)
sys.path.append(project_path)

from src.utils import lagged_correlation

input_dir = f"{project_path}/exp/signal-to-signal-correlation&coherence/data"

files = os.listdir(input_dir)
for file in files:
    if file.endswith('.npy'):
        if file == 'ecog.npy':
            ecog_data = np.load(input_dir + '/' + file, allow_pickle=True)
        if file == 'probe1.npy':
            probe1_data = np.load(input_dir + '/' + file, allow_pickle=True)
        if file == 'probe2.npy':
            probe2_data = np.load(input_dir + '/' + file, allow_pickle=True)
        if file == 'times.npy':
            times = np.load(input_dir + '/' + file, allow_pickle=True)

# Compute average of ecog data along the 0th axis
ecog_data_avg = np.mean(ecog_data, axis=0)

# Probe 1
probe1_corr, probe1_lag = [], []

# Probe 2
probe2_corr, probe2_lag = [], []

# Loop over each probe data in probe 1
for probe_1_d, probe_2_d in zip(probe1_data, probe2_data):
    # Compute the lagged correlation between probe 1 and ecog data
    lag, corr = lagged_correlation(probe_1_d, ecog_data_avg)
    # Append the lag and correlation coefficient to the lists for probe 1
    probe1_corr.append(corr)
    probe1_lag.append(lag)
    
    # Compute the lagged correlation between probe 2 and ecog data
    lag, corr = lagged_correlation(probe_2_d, ecog_data_avg)
    # Append the lag and correlation coefficient to the lists for probe 2
    probe2_corr.append(corr)
    probe2_lag.append(lag)

# log the results
logging.info(f"Probe 1 correlation: {probe1_corr}")
logging.info(f"Probe 1 lag: {probe1_lag}")
logging.info(f"Probe 2 correlation: {probe2_corr}")
logging.info(f"Probe 2 lag: {probe2_lag}")

depths = np.arange(0, 1600, 100)

# plot the correlation coefficients
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

# create colormap for scatter plot for values between -1 and 1
cmap = plt.cm.get_cmap('viridis')

# scatter plot with color according lag
for i, (depth, corr, lag) in enumerate(zip(depths, probe1_corr, probe1_lag)):
    if i == 0:
        ax.scatter(depth, lag, c=cmap(corr), label='Probe 1', alpha=0.8, marker='o')
    else:
        ax.scatter(depth, lag, c=cmap(corr), alpha=0.8, marker='o')
for i, (depth, corr, lag) in enumerate(zip(depths, probe2_corr, probe2_lag)):
    if i == 0:
        ax.scatter(depth, lag, c=cmap(corr), label='Probe 2', alpha=0.8, marker='x')
    else:
        ax.scatter(depth, lag, c=cmap(corr), alpha=0.8, marker='x')

# create colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
sm._A = []
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Pearson correlation coefficient')

# set labels
ax.set_xlabel('Depth (um)')
ax.set_ylabel('Lag (ms)')
ax.set_title('Lag between probe data and ECoG data average')

# set legend
ax.legend()

# plt.show()
# save figure
plt.savefig(f"{project_path}/res/signal-to-signal-correlation&coherence/lagged-pcc-plot.png")
