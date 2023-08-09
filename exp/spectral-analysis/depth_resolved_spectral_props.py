import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import seaborn as sns
import logging
from pathlib import Path
from collections import defaultdict

# Set up file paths
file_path = str(Path().absolute())
project_path = str(Path().absolute().parent.parent)
exp_name = file_path.split('/')[-1]

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"Current file directory: {file_path}")
logging.info(f"Current project directory: {project_path}")

# Set plotting style
sns.set_context('paper', font_scale=2, rc={'lines.linewidth': 2})
sns.set_palette('colorblind')
sns.set_style('white')

# Set color palette and style
color_palette = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
sns.set_palette(color_palette)

# Specify the input and output directories
input_dir = f'{project_path}/res/{exp_name}'

# Replace the file paths below with your actual data file paths
w12_18_path = f'{input_dir}/w12_07.spont/spectral_properties_w12_07.spont.csv'
w12_07_path = f'{input_dir}/w12_18.spont/spectral_properties_w12_18.spont.csv'

w12_18 = pd.read_csv(w12_18_path)
w12_07 = pd.read_csv(w12_07_path)
data_exp_names = ['w12_18.spont', 'w12_07.spont']

def gauss(mu, sigma, x):
    """Compute the Gaussian filter at given x values."""
    normal = 1 / (2.0 * np.pi * sigma**2)
    gauss = np.exp(-((x - mu)**2 / (2.0 * sigma**2))) * normal
    return gauss

def compute_gaussians(df, probe_type, state):
    """Compute the Gaussian filters for a specific probe type and state."""

    logging.info(f'Probe type: {probe_type}, State: {state}')
    x = np.linspace(0, 100, 10000)
    ys = []

    if probe_type == 'ECoG':
        depths = df['Channel'].unique()
    else:
        depths = df['Depth'].unique()

    for depth in depths:
        logging.info(f'Depth: {depth}')
        if probe_type == 'ECoG':
            df_depth = df[df['Channel'] == depth]
        else:
            df_depth = df[df['Depth'] == depth]

        centr_frequencies = df_depth['Central frequencies']
        peak_powers = df_depth['Peak powers']
        band_widths = df_depth['Bandwidths']

        y = np.zeros_like(x)
        for cf, pp, bw in zip(centr_frequencies, peak_powers, band_widths):
            y_i = gauss(cf, bw, x)
            y_i = y_i * pp

            if np.isnan(y_i).any():
                continue
            y += y_i

        y = y / len(centr_frequencies)
        ys.append(y)

    return np.array(ys), depths

def plot_colormesh(probe_type, state, ys, depths, ax):
    """Plot the colormesh for a specific probe type and state."""
    ax.pcolormesh(np.linspace(0, 100, 10000), depths, ys, cmap='viridis')
    ax.set_title(f'{probe_type}, {state}')
    ax.set_xlabel('Frequency (Hz)')
    if probe_type == 'ECoG':
        ax.set_ylabel('Channel')
    else:
        ax.set_ylabel('Depth (um)')


for df, name in zip([w12_18, w12_07], data_exp_names):
    output_dir = f'{project_path}/res/{exp_name}/{name}'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Set GridSpec for the figure
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 3, figure=fig, width_ratios=[3, 3, 0.1])  # Adjusted to add the colorbar

    ax1 = fig.add_subplot(gs[0, 0]) # ECoG, upstate
    ax2 = fig.add_subplot(gs[0, 1]) # ECoG, downstate
    ax3 = fig.add_subplot(gs[1, 0]) # Probe_1, upstate
    ax4 = fig.add_subplot(gs[1, 1]) # Probe_1, downstate
    ax5 = fig.add_subplot(gs[2, 0]) # Probe_2, upstate
    ax6 = fig.add_subplot(gs[2, 1]) # Probe_2, downstate
    cax = fig.add_subplot(gs[:, 2])  # Colorbar subplot

    # Iterate over all probe types
    probe_types = df['Name of probe'].unique()

    for probe_type in probe_types:
        # Take only df for the current probe type
        df_probe = df[df['Name of probe'] == probe_type]
        # Iterate over all states for the current probe type
        states = df_probe['State'].unique()
        for state in states:
            # Take only df for the current state and probe type
            df_state = df_probe[df_probe['State'] == state]
            ys, depths = compute_gaussians(df_state, probe_type, state)

            # Plot the colormesh based on probe type and state
            if probe_type == 'ECoG' and state == 'upstate':
                plot_colormesh(probe_type, state, ys, depths, ax1)

            elif probe_type == 'ECoG' and state == 'downstate':
                plot_colormesh(probe_type, state, ys, depths, ax2)

            elif probe_type == 'Probe_1' and state == 'upstate':
                plot_colormesh(probe_type, state, ys, -depths, ax3)

            elif probe_type == 'Probe_1' and state == 'downstate':
                plot_colormesh(probe_type, state, ys, -depths, ax4)
            
            elif probe_type == 'Probe_2' and state == 'upstate':
                plot_colormesh(probe_type, state, ys, -depths, ax5)
            
            elif probe_type == 'Probe_2' and state == 'downstate':
                plot_colormesh(probe_type, state, ys, -depths, ax6)

    # Also add letters to the subplots 
    ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, size=20)
    ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, size=20)
    ax3.text(-0.1, 1.1, 'C', transform=ax3.transAxes, size=20)
    ax4.text(-0.1, 1.1, 'D', transform=ax4.transAxes, size=20)
    ax5.text(-0.1, 1.1, 'E', transform=ax5.transAxes, size=20)
    ax6.text(-0.1, 1.1, 'F', transform=ax6.transAxes, size=20)

    # Add colorbar
    cbar = fig.colorbar(ax1.collections[0], cax=cax)
    cbar.set_label('FOOOF Power')
    cbar.ax.yaxis.set_label_position('left')

    # Add figure titleTo 
    fig.suptitle(f'{name} - Depth resolved spectral properties', fontsize=20)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/depth_resolved_spectral_props.png', dpi=300)
