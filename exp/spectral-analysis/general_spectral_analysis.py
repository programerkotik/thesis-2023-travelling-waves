import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from neurodsp.spectral import compute_spectrum_welch
from matplotlib.gridspec import GridSpec
import seaborn as sns
from collections import defaultdict

# Set up file paths
file_path = str(Path().absolute())
project_path = str(Path().absolute().parent.parent)

os.chdir(project_path)
sys.path.append(project_path)

# Set an experiment name, input and output directory
exps = ['w12_18.spont', 'w12_07.spont']

data = defaultdict(dict)
for exp in exps:
    # Set input and output directories
    input_dir = f"{project_path}/data/processed/{exp}"

    files = os.listdir(input_dir)
    for file in files:
        if file.endswith('.npy'):
            if file == 'Probe1_lfps_spont.npy':
                ecog_data = np.load(input_dir + '/' + file, allow_pickle=True)
            if file == 'Probe2_lfps_spont.npy':
                probe1_data = np.load(input_dir + '/' + file, allow_pickle=True)
            if file == 'Probe3_lfps_spont.npy':
                probe2_data = np.load(input_dir + '/' + file, allow_pickle=True)
            if file == 'times.npy':
                times = np.load(input_dir + '/' + file, allow_pickle=True)
            if file == 'event_times.npy':
                event_times = np.load(input_dir + '/' + file, allow_pickle=True)
    
    data[exp]['ecog_data'] = ecog_data
    data[exp]['probe1_data'] = probe1_data
    data[exp]['probe2_data'] = probe2_data
    data[exp]['times'] = times
    data[exp]['event_times'] = event_times

output_dir = f"{project_path}/res/spectral-analysis"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set plotting style
sns.set_context('paper', font_scale=2, rc={'lines.linewidth': 2})
sns.set_palette('colorblind')
sns.set_style('white')

# Set color palette and style
color_palette = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
sns.set_palette(color_palette)

freqs_ecog_18, powers_ecog_18 = [], []
for ch in range(data['w12_18.spont']['ecog_data'].shape[0]):
    freq, power = compute_spectrum_welch(data['w12_18.spont']['ecog_data'][ch, :], fs=1000)
    freqs_ecog_18.append(freq)
    powers_ecog_18.append(power)

freqs_probe1_18, powers_probe1_18 = [], []
freqs_probe2_18, powers_probe2_18 = [], []
for ch in range(data['w12_18.spont']['probe1_data'].shape[0]):
    freq_probe1, power_probe1 = compute_spectrum_welch(data['w12_18.spont']['probe1_data'][ch, :], fs=1000)
    freqs_probe1_18.append(freq_probe1)
    powers_probe1_18.append(power_probe1)

    freq_probe2, power_probe2 = compute_spectrum_welch(data['w12_18.spont']['probe2_data'][ch, :], fs=1000)
    freqs_probe2_18.append(freq_probe2)
    powers_probe2_18.append(power_probe2)

freqs_ecog_07, powers_ecog_07 = [], []
for ch in range(data['w12_07.spont']['ecog_data'].shape[0]):
    freq, power = compute_spectrum_welch(data['w12_07.spont']['ecog_data'][ch, :], fs=1000)
    freqs_ecog_07.append(freq)
    powers_ecog_07.append(power)

freqs_probe1_07, powers_probe1_07 = [], []
freqs_probe2_07, powers_probe2_07 = [], []
for ch in range(data['w12_07.spont']['probe1_data'].shape[0]):
    freq_probe1, power_probe1 = compute_spectrum_welch(data['w12_07.spont']['probe1_data'][ch, :], fs=1000)
    freqs_probe1_07.append(freq_probe1)
    powers_probe1_07.append(power_probe1)

    freq_probe2, power_probe2 = compute_spectrum_welch(data['w12_07.spont']['probe2_data'][ch, :], fs=1000)
    freqs_probe2_07.append(freq_probe2)
    powers_probe2_07.append(power_probe2)


# Set GridSpec for the figure
fig = plt.figure(figsize=(20, 15))
gs = GridSpec(2, 3, figure=fig)

# Define the experiment titles and corresponding data
experiments = [
    ('ECoG experiment w12_18', freqs_ecog_18[0], powers_ecog_18),
    ('Probe 1 experiment w12_18', freqs_probe1_18[0], powers_probe1_18),
    ('Probe 2 experiment w12_18', freqs_probe2_18[0], powers_probe2_18),
    ('ECoG experiment w12_07', freqs_ecog_07[0], powers_ecog_07),
    ('Probe 1 experiment w12_07', freqs_probe1_07[0], powers_probe1_07),
    ('Probe 2 experiment w12_07', freqs_probe2_07[0], powers_probe2_07)
]

# Define the frequency bands with LaTeX-rendered Greek symbols and colors
bands = {'$\\alpha$': (8, 13, 'tab:blue'), '$\\beta$': (13, 30, 'tab:green'), '$\\gamma$': (30, 100, 'tab:red')}
letters = ['A', 'B', 'C', 'D', 'E', 'F']

# Plot the collective spectra
for i, (title, freqs, powers) in enumerate(experiments):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    ax.plot(freqs, np.mean(powers, axis=0), color='tab:blue', linewidth=2)
    ax.fill_between(freqs, np.mean(powers, axis=0) - np.std(powers, axis=0),
                    np.mean(powers, axis=0) + np.std(powers, axis=0), alpha=0.2, color='tab:blue')
    ax.set_title(title)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power')
    ax.set_yscale('log')
    ax.set_xlim(0, 100)
    ax.set_ylim(10e-15, 10e-8)
    
    # Add description of frequency bands
    for band, (start, end, color) in bands.items():
        ax.axvspan(start, end, color=color, alpha=0.2)
        ax.text((start + end) / 2, ax.get_ylim()[1] * 0.5, band, color=color, fontsize=15, ha='center')
        
    # Add arrow and annotation for 60 Hz frequency
    ax.annotate('Artefact from line noise filtering', xy=(60, 10e-12), xytext=(60, 10e-11),
                arrowprops=dict(arrowstyle='->', color='black'), fontsize=15, ha='center')
    
    # Add letter labels to subplots
    ax.text(-0.1, 1.1, letters[i], transform=ax.transAxes, fontsize=20)
plt.tight_layout()

# Save the figure
fig.savefig(f"{output_dir}/general_spectral_analysis.png", dpi=300)
