import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import logging
from pathlib import Path
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import euclidean
import argparse

def plot_ecog_correlation(exp):
    # Set up file paths
    file_path = str(Path().absolute())
    project_path = str(Path().absolute().parent.parent)

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Current file directory: {file_path}")
    logging.info(f"Current project directory: {project_path}")

    os.chdir(project_path)
    sys.path.append(project_path)

    # Set plotting style
    sns.set_context('paper', font_scale=1.3, rc={'lines.linewidth': 2})
    sns.set_palette('colorblind')
    sns.set_style('white')

    # Set color palette and style
    color_palette = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    sns.set_palette(color_palette)

    # Set input directory for data
    input_dir = f"data/processed/{exp}"
    output_dir = f"res/upstate_qc/{exp}"

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read ecog data
    ecog_path = f"{input_dir}/Probe1_lfps_spont.npy"
    ecog_times_path = f"{input_dir}/times.npy"
    ts_ecog = np.load(ecog_path)
    times_ecog = np.load(ecog_times_path)

    # Set GridSpec for the figure
    fig = plt.figure(figsize=(20, 22))
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, :]) # Time series
    ax2 = fig.add_subplot(gs[1, 0]) # Correlation heatmap
    ax3 = fig.add_subplot(gs[1, 1]) # Graph correlation by distance

    # Plot the time series
    for i, ch in enumerate(range(ts_ecog.shape[0])):
        ax1.plot(times_ecog, ts_ecog[ch] - 15e-5 * i, label=f'Channel {ch}', c='tab:red', alpha=0.5)

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Voltage')
    ax1.set_title(f'Time series of ECoG channels')
    ax1.set_xlim(140, 150)

    # Compute correlation matrix
    corr_matrix = np.corrcoef(ts_ecog)

    # Plot correlation matrix
    ax2.imshow(corr_matrix, cmap='viridis')
    # Set colorbar
    cbar = plt.colorbar(ax2.imshow(corr_matrix, cmap='viridis'), ax=ax2)
    # Set aspect ratio
    ax2.set_aspect('equal')
    ax2.set_title('Correlation matrix of ECoG channels')
    ax2.set_xlabel('Channel')
    ax2.set_ylabel('Channel')

    # organize data according to channel map
    channel_map = np.array([[49,50,51,52,59,58,57,56,55,54,53],
                            [60,61,62,63,32,43,44,45,46,47,65],
                            [42,41,40,39,38,33,34,35,36,37,65],
                            [26,25,24,23,22,17,18,19,20,21,65],
                            [12,13,14,15,16,27,28,29,30,31,65],
                            [1,2,3,4,11,10,9,8,7,6,5]])

    # Go through each pair of channels, compute distance between them, and plot correlation vs distance
    channels = range(1, 64)
    corrs = []
    distances = []
    for i in channels:
        for j in channels:
            if i == j:
                continue
            # Get coordinates of the two channels
            coord1 = np.where(channel_map == i)
            coord2 = np.where(channel_map == j)

            # Check if the channel is in the channel map
            if len(coord1[0]) == 0 or len(coord2[0]) == 0:
                continue

            # Concatenate coordinates
            coord1 = np.concatenate((coord1[0], coord1[1]))
            coord2 = np.concatenate((coord2[0], coord2[1]))
            # Compute distance
            distance = euclidean(coord1, coord2)
            # Compute correlation
            corr = corr_matrix[i-1, j-1]
            # Append to list
            corrs.append(corr)
            distances.append(distance)

    # Plot
    ax3.scatter(distances, corrs, alpha=0.1, c='tab:cyan')

    # also plot the mean correlation for each distance
    mean_corrs = []
    for d in np.unique(distances):
        mean_corrs.append(np.mean(np.array(corrs)[np.where(np.array(distances) == d)]))
    ax3.scatter(np.unique(distances), mean_corrs, c='tab:red', marker='x', s=150)

    # Connect the mean correlation points
    for i in range(len(np.unique(distances)) - 1):
        ax3.plot([np.unique(distances)[i], np.unique(distances)[i+1]],
                 [mean_corrs[i], mean_corrs[i+1]], linestyle='--', c='tab:red', alpha=0.5, linewidth=3)

    ax3.set_xlim(0.5, 11.5)
    ax3.set_xlabel('Distance')
    ax3.set_ylabel('Correlation')
    ax3.set_title('Correlation vs distance')

    # Add letters to subplots
    ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, fontsize=20)
    ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, fontsize=20)
    ax3.text(-0.1, 1.1, 'C', transform=ax3.transAxes, fontsize=20)

    # Save figure
    fig.savefig(f"{output_dir}/ecog_ecog_correlation.png", dpi=300)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Plot ECoG correlation.')
    parser.add_argument('exp', type=str, help='Experiment name')

    # Parse command-line arguments
    args = parser.parse_args()

    exp = args.exp
    plot_ecog_correlation(exp)

if __name__ == "__main__":
    main()
