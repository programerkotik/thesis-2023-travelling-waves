import sys
import os
import logging
from pathlib import Path
import numpy as np
import quantities as pq
from neurodsp.filt import filter_signal
import yaml

# Set up file paths
file_path = str(Path().absolute())
project_path = str(Path().absolute().parent.parent)
exp_name = file_path.split('/')[-1]

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"Current file directory: {file_path}")
logging.info(f"Current project directory: {project_path}")

os.chdir(project_path)
sys.path.append(project_path)

# Import custom modules
from src.loaders import load_dataset
from src.utils import define_upstate_regions
from src.plotting import (plot_filtered_time_series_snippet, plot_filtered_ts_with_upstates,
                          plot_interval_durations)

# Read parameters from yml file
parameters_path = f"{file_path}/find_upstates.yml"

# Load the YAML file and read the parameters
with open(parameters_path) as f:
    d = yaml.load(f.read(), Loader=yaml.FullLoader)
    parameters = d['params']

freq_range = parameters[0]['f_range'] # Frequency range for low-pass filtering
threshold_scalar = parameters[1]['threshold_scalar'] # Threshold scalar for upstate detection

logging.info(f"Frequency range for low-pass filtering: {freq_range}")
logging.info(f"Threshold scalar for upstate detection: {threshold_scalar}")

# Specify input and output directories
input_dir = f"{project_path}/data/processed/2022-09-27-data/1kHz"
output_dir = f"{project_path}/data/processed/2022-09-27-data/1kHz"
fig_output_dir = f"{project_path}/res/{exp_name}/figs" # directory to save figures

# Create output directory if it does not exist
os.makedirs(fig_output_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Load only ECoG data
ecog_path = f"{input_dir}/ts_filtered_ecog.npy"
ecog_times_path = f"{input_dir}/times.npy"
ts_ecog = np.load(ecog_path)
times_ecog = np.load(ecog_times_path)

# Filter (low-pass) all channels of ECoG data to smoothe out high-frequency signal for upstate detection
sig_filt = np.array([filter_signal(sig, 1000 * pq.Hz, 'lowpass', freq_range) for sig in ts_ecog])

# Remove nan values from the filtered signal
sig_filt = np.nan_to_num(sig_filt)

# Define upstate regions
event_times, threshold_value = define_upstate_regions(sig_filt, times_ecog, threshold_scalar)

# Save event times to output directory as numpy array file
np.save(f"{output_dir}/event_times.npy", event_times)

# Compute upstate/downstate duration statistics
upstate_durations = []
downstate_durations = []
for i in range(len(event_times)):
    upstate_durations.append(event_times[i][1] - event_times[i][0])
    if i < len(event_times) - 1:
        downstate_durations.append(event_times[i+1][0] - event_times[i][1])

plot_interval_durations(upstate_durations, fig_output_dir, interval_type='upstate')
# plot_interval_durations(downstate_durations, fig_output_dir)

# Plot some snipshots of filtered ECoG data
xlims = [[120, 140], [240, 250], [470, 500]] # time range to plot
for i, xlim in enumerate(xlims):
    plot_filtered_time_series_snippet(ts_ecog, sig_filt, times_ecog, xlim=xlim, fig_output_dir=fig_output_dir)
    plot_filtered_ts_with_upstates(times_ecog, sig_filt, threshold_value, event_times, fig_output_dir, xlim=xlim)

