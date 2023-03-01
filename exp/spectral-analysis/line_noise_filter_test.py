# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from neurodsp.spectral import compute_spectrum_welch
import sys
import os
from pathlib import Path
import yaml

# Set up file paths
file_path = str(Path().absolute())
project_path = str(Path().absolute().parent.parent)

# Print out file and project paths for debugging purposes
print(file_path)
print(project_path)

# Add project path to sys path to enable importing of custom modules
os.chdir(project_path)
sys.path.append(project_path)

# Import custom modules from src directory
from src import *

# Define file to read
file = 'raw/2022-09-27_data/1kHz/lfp68.ibw'

# Load the data
ts, time = load_ibw_data(file)

# Get the sampling rate
fs = ts.sampling_rate.magnitude

# Convert time series to a numpy array and filter out 60Hz line noise
ts = np.squeeze(ts.magnitude)

# Load the YAML file
with open('exp/2022-11-29_exp/line_noise_filter_test.yml', 'r') as file:
    parameters = yaml.safe_load(file)['params']

# Loop over the parameter sets and apply the filter
for params in parameters:
    q = params['q']

    ts_1st_probe_filtered = filter_line_noise(ts, fs, Q=q) # Q is the quality factor of the notch filter

    # Plot filtered and unfiltered time series, and their difference, to check that the filter worked + plot the power spectrum
    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    ax[0].plot(time, ts, label='unfiltered')
    ax[0].plot(time, ts_1st_probe_filtered, label='filtered')
    ax[0].plot(time, ts - ts_1st_probe_filtered, label='difference')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Voltage (uV)')
    ax[0].set_title('Time series')
    ax[0].legend()
    ax[0].set_xlim([5, 6])

    # Compute power spectrum using Welch's method and plot it
    freqs, psd = compute_spectrum_welch(ts, fs, nperseg=fs)
    freqs_1st_probe_filtered, psd_1st_probe_filtered = compute_spectrum_welch(ts_1st_probe_filtered, fs, nperseg=fs)
    ax[1].loglog(freqs, psd, label='unfiltered')
    ax[1].loglog(freqs_1st_probe_filtered, psd_1st_probe_filtered, label='filtered')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('Power spectral density (uV^2/Hz)')
    ax[1].set_title('Power spectrum')
    ax[1].legend()

    # Set plot layout and display it
    plt.tight_layout()
    plt.show()

    # Save the figure to a folder named after the experiment
    exp_name = file_path.split('/')[-1].split('_')[0]
    fig.savefig(f'res/{exp_name}_res/line_noise_filter_test/figures/line_noise_filter_test_q{q}.png')
    
