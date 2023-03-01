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

freq_range=[1, 100]
 
for k, probe in enumerate([ts_1st_probe_filtered, ts_2nd_probe_filtered]):
    # for each data file, compute the power spectrum density using welch method, fit fooof model and plot the periodic component
    # go through each electrode 
    center_freqs_all = []
    for i in range(len(probe)):
        ts = probe[i]
        # for each data file, go through the following steps
        # 1. create time intervals
        intervals = np.arange(0, int(np.max(times)), 150)
        # 2. then for each time interval
        center_freqs = []
        for j in range(len(intervals)-1):
            idx = np.where((times >= intervals[j]) & (times < intervals[j+1]))[0]
            # print('for interval from', intervals[j], 'to', intervals[j+1], 'seconds')
            freqs, powers = compute_spectrum_welch(ts[idx], fs=1000)

            fm = FOOOF(verbose=False)
            fm.fit(freqs, powers, freq_range)
            cf = get_band_peak_fm(fm, freq_range) # SHOULD I GET THE PEAKS FOR ALL BANDS?
            center_freqs.extend(cf)
        
        idx = np.where((times >= intervals[j+1]))[0]
        freqs, powers = compute_spectrum_welch(ts[idx], fs=1000)
        fm = FOOOF(verbose=False)
        fm.fit(freqs, powers, freq_range)

        cf = get_band_peak_fm(fm, freq_range) # SHOULD I GET THE PEAKS FOR ALL BANDS?
        center_freqs.extend(cf)
        
        center_freqs = np.array(center_freqs)    
        center_freqs_all.append(center_freqs)

    # plot histograms of center frequencies
    fig, ax = plt.subplots(8, 2, figsize=(10, 30))
    axes = ax.flatten()
    for i, ax in zip(range(len(center_freqs_all)), axes):
        ax.hist(center_freqs_all[i], bins=30, label='electrode {}'.format(i+1))
        ax.set_xlabel('Center Frequency (Hz)')
        ax.set_ylabel('Count')
        ax.set_xlim(freq_range)
        ax.legend()

    plt.tight_layout()
    exp_name = file_path.split('/')[-1].split('_')[0]
    plt.savefig(f'res/{exp_name}_res/max_peaks_time_resolved/figures/max_peaks_hist_time_wind_{k+1}.png')
