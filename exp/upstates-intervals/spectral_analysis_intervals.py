import sys
import os
import logging
from pathlib import Path
import numpy as np
from neurodsp.spectral import compute_spectrum_welch
from fooof import FOOOF
from fooof.analysis import get_band_peak_fm
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

# Set up file paths
file_path = str(Path().absolute())
project_path = str(Path().absolute().parent.parent)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"Current file directory: {file_path}")
logging.info(f"Current project directory: {project_path}")

os.chdir(project_path)
sys.path.append(project_path)

# Import custom modules
from src.utils import split_intervals, filter_line_noise

# Specify the studied experiments
exp = 'w12_07.spont'

# Specify input directory
input_dir = f'{project_path}/data/processed/{exp}/1kHz'

files = os.listdir(input_dir)

for file in files:
    if file.endswith('.npy'):
        if file == 'ts_filtered_ecog.npy':
            ecog_data = np.load(input_dir + '/' + file, allow_pickle=True)
        if file == 'ts_filtered_probe1.npy':
            probe1_data = np.load(input_dir + '/' + file, allow_pickle=True)
        if file == 'ts_filtered_probe2.npy':
            probe2_data = np.load(input_dir + '/' + file, allow_pickle=True)
        if file == 'times.npy':
            times = np.load(input_dir + '/' + file, allow_pickle=True)

#!! Don't forget that w12_07.spont is not line filtered
# Filter line noise
ecog_data = np.array([filter_line_noise(d, 1000, 550) for d in ecog_data])
probe1_data = np.array([filter_line_noise(d, 1000, 550) for d in probe1_data])
probe2_data = np.array([filter_line_noise(d, 1000, 550) for d in probe2_data])

# Read upstate event times
upstates = np.load(f"{input_dir}/event_times.npy", allow_pickle=True)

# Get downstates as intervals between upstates
downstates = []
for i in range(len(upstates) - 1):
    downstates.append([upstates[i][1], upstates[i + 1][0]])

# Split data into upstate and downstate intervals
ecog_data_upstates, ecog_data_downstates = split_intervals(ecog_data, times, upstates, downstates)
data = {'upstate': ecog_data_upstates, 'downstate': ecog_data_downstates}
spectral_properties = defaultdict(list)

for state, ecog_data in data.items():
    print('State:', state, '...')
    for channel, values in tqdm(ecog_data.items()):
        for i, interval in enumerate(values):
            
            # Skip if interval is too short
            if len(interval) < 500:
                continue

            # Save spectral properties
            spectral_properties['Interval id'].append(i)
            spectral_properties['State'].append(state)
            spectral_properties['Channel'].append(channel)

            # Compute power spectrum using Welch's method (probably, however, it will still include only 1 window of data (500 samples))
            freqs, powers = compute_spectrum_welch(interval, fs=1000, nperseg=500, f_range=(0, 100),  noverlap=0)
            # Get Fooof model
            fm = FOOOF(verbose=False)
            fm.fit(freqs, powers, freq_range=(1, 100))
            # Get peak frequency
            peak_freq = get_band_peak_fm(fm, [1, 100], select_highest=True)

            spectral_properties['Central frequencies'].append(peak_freq[0])
            spectral_properties['Peak powers'].append(peak_freq[1])
            spectral_properties['Bandwidths'].append(peak_freq[2])
            spectral_properties['Power spectrum (freqs)'].append(freqs)
            spectral_properties['Power spectrum (powers)'].append(powers)

# Save spectral properties
output_dir = f"{project_path}/res/upstates-intervals"

# create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

pd.DataFrame(spectral_properties).to_csv(f"{output_dir}/spectral_properties_{exp}.csv", index=False, float_format='%.3f')