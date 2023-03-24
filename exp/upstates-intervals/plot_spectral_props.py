# Description: Plot spectral properties of upstates and downstates in ECoG data and save figures. 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import logging
from pathlib import Path
import matplotlib.patches as mpatches
from fooof.sim import gen_freqs

def gauss(mu,sigma,x):
    # lower normal part of gaussian
    normal = 1/(2.0 * np.pi * sigma**2)
    # calculating Gaussian filter
    gauss = np.exp(-((x-mu)**2 / (2.0 * sigma**2))) * normal
    return gauss

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
from src.utils import parse_spectrum

# Set input directory for data
input_dir = f"{project_path}/exp/{exp_name}/data"

# Set output directory for plots
output_dir = f"{project_path}/res/{exp_name}/spectral-properties"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read data
data = pd.read_csv(f'{input_dir}/spectral_properties.csv')

# Get upstate and downstate data
upstate_data = data[data['State'] == 'upstate']
downstate_data = data[data['State'] == 'downstate']

freqs_upstate_mean, powers_upstate_mean = [], []
freqs_downstate_mean, powers_downstate_mean = [], []

# Get mean values for each channel
for channel in upstate_data['Channel'].unique():
    upstate_data_ch = upstate_data[upstate_data['Channel'] == channel]
    downstate_data_ch = downstate_data[downstate_data['Channel'] == channel]

    # Get all center frequencies, bandwidths, and powers
    upstate_cf = upstate_data_ch['Central frequencies'].values
    upstate_bw = upstate_data_ch['Bandwidths'].values
    upstate_pow = upstate_data_ch['Peak powers'].values

    downstate_cf = downstate_data_ch['Central frequencies'].values
    downstate_bw = downstate_data_ch['Bandwidths'].values
    downstate_pow = downstate_data_ch['Peak powers'].values

    # Get power spectrum
    spectrum = upstate_data_ch['Power spectrum (freqs)'].values
    freqs_upstate, powers_upstate = parse_spectrum(spectrum, upstate_data_ch['Power spectrum (powers)'].values)
    freqs_upstate = np.array(freqs_upstate)
    powers_upstate = np.array(powers_upstate)

    # Get power spectrum
    spectrum = downstate_data_ch['Power spectrum (freqs)'].values
    freqs_downstate, powers_downstate = parse_spectrum(spectrum, downstate_data_ch['Power spectrum (powers)'].values)
    freqs_downstate = np.array(freqs_downstate)
    powers_downstate = np.array(powers_downstate)

    # Append to list
    freqs_upstate_mean.append(freqs_upstate)
    powers_upstate_mean.append(powers_upstate)
    
    freqs_downstate_mean.append(freqs_downstate)
    powers_downstate_mean.append(powers_downstate)

freqs_upstate_mean = np.array(freqs_upstate_mean)
powers_upstate_mean = np.array(powers_upstate_mean)
# Plot power spectrum
freqs_upstate_mean = np.mean(freqs_upstate_mean, axis=0) # mean across channels
powers_upstate_std = np.std(powers_upstate, axis=0) # std across channels
powers_upstate_mean = np.mean(powers_upstate_mean, axis=0) # mean across channels


freqs_downstate_mean = np.mean(freqs_downstate_mean, axis=0) # mean across channels
powers_downstate_std = np.std(powers_downstate_mean, axis=0)  # std across channels
powers_downstate_mean = np.mean(powers_downstate_mean, axis=0) # mean across channels

fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(np.mean(freqs_upstate_mean, axis=0), np.mean(powers_upstate_mean, axis=0), label='Upstate (mean)', color='tab:red')
ax.fill_between(np.mean(freqs_upstate_mean, axis=0), np.mean(powers_upstate_mean, axis=0) - np.std(powers_upstate_mean, axis=0), np.mean(powers_upstate_mean, axis=0) + np.std(powers_upstate_mean, axis=0), alpha=0.5, label='Upstate (std)', color='tab:red')
ax.plot(np.mean(freqs_downstate_mean, axis=0), np.mean(powers_downstate_mean, axis=0), label='Downstate (mean)', color='tab:blue')
ax.fill_between(np.mean(freqs_downstate_mean, axis=0), np.mean(powers_downstate_mean, axis=0) - np.std(powers_downstate_mean, axis=0), np.mean(powers_downstate_mean, axis=0) + np.std(powers_downstate_mean, axis=0), alpha=0.5, label='Downstate (std)', color='tab:blue')
ax.set_xlim(0, 60)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Mean power across channels and intervals')
ax.set_title('Power spectrum density')
ax.legend()
# Save figure
fig.savefig(f"{output_dir}/power_spectrum.png")


# Remove rows with NaN rows 
data = data.dropna()

# Get upstate and downstate data
upstate_data = data[data['State'] == 'upstate']
downstate_data = data[data['State'] == 'downstate']

# Plot 2d histogram of center frequencies and powers
fig, ax = plt.subplots(figsize=(20, 10))
h2 = ax.hist2d(downstate_data['Central frequencies'], downstate_data['Peak powers'], bins=100, cmap='Blues', alpha=0.7, label='Downstate')
h1 = ax.hist2d(upstate_data['Central frequencies'], upstate_data['Peak powers'], bins=100, cmap='Reds', alpha=0.7, label='Upstate')
ax.legend(handles=[mpatches.Patch(color='tab:red', label='Upstate', alpha=0.8), mpatches.Patch(color='tab:blue', label='Downstate', alpha=0.8)])
# Add arrow to indicate line noise at 60 Hz
ax.arrow(60, 2.5, 0, -0.2, head_width=1, head_length=0.1, fc='k', ec='k')
ax.text(57, 2.6, 'Line noise', color='k')
ax.set_xlabel('Center frequency (Hz)')
ax.set_ylabel('Power')
ax.set_title('Center frequency and power histogram in upstate and downstate signal intervals')
# Save figure
fig.savefig(f"{output_dir}/center_freq_power_hist.png")

# Plot only center frequency histogram
fig, ax = plt.subplots(figsize=(20, 10))
ax.hist(upstate_data['Central frequencies'], bins=100, color='tab:red', alpha=0.5, label='Upstate')
ax.hist(downstate_data['Central frequencies'], bins=100, color='Tab:blue', alpha=0.5, label='Downstate')
# Add arrow to indicate line noise at 60 Hz
ax.arrow(60, 1500, 0, -100, head_width=1, head_length=50, fc='k', ec='k')
ax.text(57, 1550, 'Line noise', color='k')
ax.set_xlabel('Center frequency (Hz)')
ax.set_ylabel('Count')
ax.set_title('Center frequency histogram in upstate and downstate signal intervals')
ax.legend()
# Save figure
fig.savefig(f"{output_dir}/center_freq_hist.png")


# Plot only periodic component spectrum
fig, ax = plt.subplots(figsize=(20, 10))
freqs = gen_freqs(freq_range=[0,100], freq_res=1)

upstate_peak_vals = []
for cf, bw, pw in zip(upstate_data['Central frequencies'], upstate_data['Bandwidths'], upstate_data['Peak powers']):
    peak_vals = gauss(cf,bw,freqs)
    upstate_peak_vals.append(peak_vals)
    
upstate_peak_vals = np.array(upstate_peak_vals)

downstate_peak_vals = []
for cf, bw, pw in zip(downstate_data['Central frequencies'], downstate_data['Bandwidths'], downstate_data['Peak powers']):
    peak_vals = gauss(cf,bw,freqs)
    downstate_peak_vals.append(peak_vals)
    
downstate_peak_vals = np.array(downstate_peak_vals)
downstate_peak_vals_mean = np.mean(downstate_peak_vals, axis=0)

upstate_peak_vals_mean = np.mean(upstate_peak_vals, axis=0)

plt.plot(freqs, downstate_peak_vals_mean, label = 'Downstate', color='tab:blue')
plt.plot(freqs, upstate_peak_vals_mean, label = 'Upstate', color='tab:red')
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Mean oscillatory component power across channels and intervals')
plt.title('Mean oscillatory component power across channels and intervals')
# Save figure
fig.savefig(f"{output_dir}/oscillatory_component_power.png")


# Get power inside beta band (12-30 Hz) for upstate and downstate
powers_beta_upstate = []
powers_beta_downstate = []
beta_band = [12, 30]
beta_band_idx = np.where((freqs >= beta_band[0]) & (freqs <= beta_band[1]))[0]
beta_band_power_downstate = [peak_vals[beta_band_idx] for peak_vals in downstate_peak_vals]
beta_band_power_upstate = [peak_vals[beta_band_idx] for peak_vals in upstate_peak_vals]

# Plot bar chart of power in beta band
fig, ax = plt.subplots(figsize=(20, 10))
sns.barplot(x=['Downstate', 'Upstate'], y=[np.mean(beta_band_power_downstate), np.mean(beta_band_power_upstate)], ax=ax, palette=['tab:blue', 'tab:red'], errorbar='sd')
ax.set_ylabel('Mean power in beta band (12-30 Hz)')
ax.set_title('Mean power in beta band (12-30 Hz) in upstate and downstate signal intervals')
# Save figure
fig.savefig(f"{output_dir}/beta_band_power.png")

