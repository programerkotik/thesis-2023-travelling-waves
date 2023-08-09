import sys
import os
import logging
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import yaml
from matplotlib.gridspec import GridSpec
import seaborn as sns
import argparse

def plot_correlation(exp):
    # Set up file paths
    file_path = str(Path().absolute())
    project_path = str(Path().absolute().parent.parent)

    # Set plotting style
    sns.set_context('paper', font_scale=1.8, rc={'lines.linewidth': 2})
    sns.set_palette('colorblind')
    sns.set_style('white')

    # Set color palette and style
    color_palette = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    sns.set_palette(color_palette)

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Current file directory: {file_path}")
    logging.info(f"Current project directory: {project_path}")

    os.chdir(project_path)
    sys.path.append(project_path)

    input_dir = f"{project_path}/data/processed/{exp}"
    output_dir = f"{project_path}/res/signal-to-signal-correlation/{exp}"

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

    # Compute the pearson correlation coefficient between the ecog signal and each probe and each channel signal
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

    # Set GridSpec for the figure
    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(3, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0]) # Heatmap 1 for probe 1
    ax2 = fig.add_subplot(gs[1, 0]) # Heatmap 2 for probe 2

    # Create both heatmaps 
    for i, (probe, ax) in enumerate(zip([probe1_data, probe2_data], [ax1, ax2])):
        # Set aspect ratio to be square
        ax.set_aspect('equal', 'box')
        # On last axis plot the correlation within one probe as heatmap
        corr_matrix = np.corrcoef(probe)
        ax.imshow(corr_matrix, cmap='viridis')
        ax.set_title(f'Correlation between channels within probe {i+1}')
        ax.set_xlabel('Channel number')
        ax.set_ylabel('Channel number')
        # Set ticks to be ints
        ax.set_xticks(np.arange(0, len(probe), 3))
        ax.set_yticks(np.arange(0, len(probe), 3))
        # Add color bar
        cbar = plt.colorbar(ax.imshow(corr_matrix, cmap='viridis'))
        cbar.set_label('Pearson correlation coefficient')

    ax3 = fig.add_subplot(gs[2, :]) # PCC ECoG vs Probe 1 and 2

    # Create the plot for the correlation between ecog and probe 1 and 2
    ax3.errorbar(depths, probe1_corrs_mean[:, 0], yerr=probe1_corrs_std[:, 0], fmt='o', label='Probe 1', color='tab:purple')
    ax3.errorbar(depths, probe2_corrs_mean[:, 0], yerr=probe2_corrs_std[:, 0], fmt='o', label='Probe 2', color='tab:olive')

    ax3.plot(depths, probe1_corrs_mean[:, 0], alpha=0.2, linestyle='--', color='tab:purple', linewidth=2)
    ax3.plot(depths, probe2_corrs_mean[:, 0], alpha=0.2, linestyle='--', color='tab:olive', linewidth=2)

    ax3.set_title('Pearson correlation coefficient between ECoG signals and probe signals')
    ax3.set_xlabel('Depth (um)')
    ax3.set_ylabel('Pearson correlation coefficient')
    ax3.legend()

    ax5 = fig.add_subplot(gs[0, 1]) # Probe 1 + event times
    ax6 = fig.add_subplot(gs[1, 1]) # Probe 2 + event times

    for j, (probe, ax) in enumerate(zip([probe1_data, probe2_data], [ax5, ax6])):
        # On ax2 plot Probe1 channels with event times overlayed
        for i, probe_d in enumerate(probe):
            ax.plot(times, probe_d - 0.001*i, label=f'Channel {i}', color='tab:blue')

        for i in range(len(event_times)):
            ax.axvspan(event_times[i][0], event_times[i][1], alpha=0.2, color='tab:green')

        ax.set_title(f'Probe {j+1} channels with upstate times overlayed')
        ax.set_xlabel('Time (s)')
        # ax.set_ylabel('Channel number')
        ax.set_xlim(100, 125)
        #ax.legend()

    # Also add letters to the subplots 
    ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, size=20)
    ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, size=20)
    ax3.text(-0.1, 1.1, 'C', transform=ax3.transAxes, size=20)
    ax5.text(-0.1, 1.1, 'D', transform=ax5.transAxes, size=20)
    ax6.text(-0.1, 1.1, 'E', transform=ax6.transAxes, size=20)

    # save figure
    plt.savefig(f'{output_dir}/ecog_probes_correlation.png')

    fig = plt.figure(figsize=(25, 25))
    gs = GridSpec(3, 1, figure=fig)
    axes = []

    # Create subplots
    for i in range(3):
        axes.append(fig.add_subplot(gs[i, 0]))

    # Plot data on each subplot
    subplots = ['ECoG', 'Probe 1', 'Probe 2']
    data = [ecog_data, probe1_data, probe2_data]

    for j, subplot in enumerate(subplots):
        ax = axes[j]
        ax.set_title(f'{subplot} channels with upstate times overlayed')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Channel number')
        ax.set_xlim(100, 125)

        for i, channel_data in enumerate(data[j]):
            ax.plot(times, channel_data - 0.001*i, label=f'Channel {i}')

        for i in range(len(event_times)):
            ax.axvspan(event_times[i][0], event_times[i][1], alpha=0.2, color='tab:green')

    # Save figure
    plt.savefig(f'{output_dir}/supplementary_figure_all_probes_signal.png')

def main():
    parser = argparse.ArgumentParser(description="Plot signal-to-signal correlation figures.")
    parser.add_argument("exp", type=str, help="Experiment name.")
    args = parser.parse_args()

    plot_correlation(args.exp)

if __name__ == "__main__":
    main()