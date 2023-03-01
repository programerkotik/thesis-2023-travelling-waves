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
from fooof.analysis import get_band_peak_fm
from fooof.plts.periodic import plot_peak_fits
from fooof.core.funcs import gaussian_function
from fooof.sim import gen_freqs


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

for j, probe in enumerate([ts_1st_probe_filtered, ts_2nd_probe_filtered]):
    # for each data file, compute the power spectrum density using welch method, fit fooof model and plot the periodic component

    # prepare matplotlib axes
    fig, axes = plt.subplots(int(len(probe)/2), 2, figsize=(10, 30))
    axes = axes.flatten()

    peak_vals_arr = []
    depths = []
    for i, (ts, ax) in enumerate(zip(probe, axes)):
        f, t, spec = signal.spectrogram(ts, fs=1000, nperseg=1000, noverlap=900)
        ids = np.where((f < 100) & (f > 1))
        spec = spec[ids]
        freq = f[ids]
        # convert time to minutes
        t = t/60
        # get spec in log scale
        spec_log = np.log10(spec)
        ax.pcolormesh(t, freq, spec_log, cmap='plasma')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [min]')
    # add colorbar
    fig.colorbar(ax.pcolormesh(t, freq, spec_log, cmap='plasma'), ax=axes.ravel().tolist(), label='Power')
    # plt.show()
    exp_name = file_path.split('/')[-1].split('_')[0]
    fig.savefig(f'res/{exp_name}_res/spectrogram/figures/spectrogram_probe_{j+1}.png')
