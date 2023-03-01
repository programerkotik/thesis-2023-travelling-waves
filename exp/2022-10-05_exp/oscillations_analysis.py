import numpy as np
import matplotlib.pyplot as plt
from neurodsp.spectral import compute_spectrum_welch
from fooof.plts.fm import plot_fm
from glob import glob
from fooof import FOOOF
import sys
import os
from pathlib import Path
from natsort import natsorted

# Set up file paths
file_path = str(Path().absolute())
project_path = str(Path().absolute().parent.parent)

print(file_path)
print(project_path)

os.chdir(project_path)
sys.path.append(project_path)

from src import *

# read the data
data = sorted(glob('raw/2022-09-27_data/1kHz/lfp*.ibw'))
# get 1st probe data
files_1st_probe = natsorted([d for d in data if int(d.split('/')[-1].split('.')[0].split('p')[-1]) >= 65 and int(d.split('/')[-1].split('.')[0].split('p')[-1]) <= 80])
# get 2nd probe data
files_2st_probe = natsorted([d for d in data if int(d.split('/')[-1].split('.')[0].split('p')[-1]) >= 97 and int(d.split('/')[-1].split('.')[0].split('p')[-1]) <= 112])

# load the data
data_1st_probe = [load_ibw_data(f)[0] for f in files_1st_probe]
data_2st_probe = [load_ibw_data(f)[0] for f in files_2st_probe]

# get the times
times = load_ibw_data(files_1st_probe[0])[1]

# get the sampling rate
fs = data_1st_probe[0].sampling_rate

# get the time series
ts_1st_probe = [np.squeeze(d.magnitude) for d in data_1st_probe]
ts_2nd_probe = [np.squeeze(d.magnitude) for d in data_2st_probe]


# filter the data for line noise
ts_1st_probe_filtered = [filter_line_noise(ts, fs.magnitude, 550) for ts in ts_1st_probe]
ts_2nd_probe_filtered = [filter_line_noise(ts, fs.magnitude, 550) for ts in ts_2nd_probe]

freq_range=[1, 100]
for j, probe in enumerate([ts_1st_probe_filtered, ts_2nd_probe_filtered]):
    # for each data file, compute the power spectrum density using welch method, fit fooof model and plot the periodic component

    # prepare matplotlib axes
    fig, axes = plt.subplots(int(len(probe)/2), 2, figsize=(10, 50), constrained_layout=True)
    axes = axes.flatten()
    for i, (ts, ax) in enumerate(zip(probe, axes)):
        # compute the power spectrum density
        freqs, powers = compute_spectrum_welch(ts, fs=1000)
        # fit fooof model
        fm = FOOOF(verbose=False)
        fm.fit(freqs, powers, freq_range=freq_range)
        # plot the periodic component
        plot_fm(fm, plot_peaks=None, plot_aperiodic=True, plt_log=False, add_legend=True, ax=ax)

    plt.tight_layout()
    # Save the figure to a folder named after the experiment
    exp_name = file_path.split('/')[-1].split('_')[0]
    fig.savefig(f'res/{exp_name}_res/oscillations_analysis/figures/fooof_model_probe_{j+1}.png')