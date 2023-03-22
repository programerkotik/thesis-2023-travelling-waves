import sys
import os
import logging
from pathlib import Path

import numpy as np
import quantities as pq
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

from src.loaders import load_dataset
from src.utils import define_upstate_regions
from src.plotting import (plot_filtered_data, plot_filtered_data_with_upstates,
                          plot_upstate_durations, plot_downstate_durations)

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

# compute the pearson correlation coefficient between the ecog signal and each probe and each channel signal
probe1_corrs = []
probe2_corrs = []

for i, ecog_d in enumerate(ecog_data):
    # probe 1
    probe1_corr = [pearsonr(probe_d, ecog_d) for probe_d in probe1_data]
    probe1_corrs.append(probe1_corr)

    # probe 2
    probe2_corr = [pearsonr(probe_d, ecog_d) for probe_d in probe2_data]
    probe2_corrs.append(probe2_corr)

# plot the correlation coefficients vs depth with std error bars
probe1_corrs = np.array(probe1_corrs)
probe2_corrs = np.array(probe2_corrs)

probe1_corrs_mean = np.mean(probe1_corrs, axis=0)
probe2_corrs_mean = np.mean(probe2_corrs, axis=0)

probe1_corrs_std = np.std(probe1_corrs, axis=0)
probe2_corrs_std = np.std(probe2_corrs, axis=0)

depths = np.arange(0, 1600, 100)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.errorbar(depths, probe1_corrs_mean[:, 0], yerr=probe1_corrs_std[:, 0], fmt='o', label='Probe 1', color='blue')
ax.errorbar(depths, probe2_corrs_mean[:, 0], yerr=probe2_corrs_std[:, 0], fmt='o', label='Probe 2', color='red')

ax.plot(depths, probe1_corrs_mean[:, 0], color='blue', alpha=0.2, linestyle='--')
ax.plot(depths, probe2_corrs_mean[:, 0], color='red', alpha=0.2, linestyle='--')

ax.set_title('Pearson correlation coefficient between ECoG signals and probe signals')
ax.set_xlabel('Depth (um)')
ax.set_ylabel('Pearson correlation coefficient')
ax.legend()

# plt.show()
# save figure
plt.savefig(project_path + '/res/signal-to-signal-correlation&coherence/pcc.png')