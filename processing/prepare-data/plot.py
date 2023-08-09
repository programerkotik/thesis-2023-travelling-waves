import sys
import os
import logging
from pathlib import Path
from glob import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_time_series(exp):
    # Set up file paths
    file_path = str(Path().absolute())
    project_path = str(Path().absolute().parent.parent)

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Change working directory to project path
    os.chdir(project_path)
    sys.path.append(project_path)

    # Set plotting style
    sns.set_context('paper', font_scale=2, rc={'lines.linewidth': 2})
    sns.set_palette('colorblind')
    sns.set_style('white')

    # Set color palette and style
    color_palette = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    sns.set_palette(color_palette)

    # Create output directory if it does not exist
    output_dir = f"{project_path}/res/processing/{exp}"
    os.makedirs(output_dir, exist_ok=True)

    files = glob(f'data/processed/{exp}/Probe*.npy')
    times = np.load(f'data/processed/{exp}/times.npy')

    for file in files:
        # Get name of file
        filename = file.split('/')[-1].split('.')[0]
        # Load time series
        ts = np.load(file)

        # Plot time series of all channels
        # Set a figure size
        fig = plt.figure(figsize=(30, 10))

        # Plot the time series
        for i, ch in enumerate(range(ts.shape[0])):
            plt.plot(times, ts[ch] - 10e-4 * i, label=f'Channel {ch}')

        plt.xlabel('Time (s)')
        plt.ylabel('Voltage')
        plt.title(f'Time series')
        plt.xlim(0, 10)
        plt.savefig(f'{output_dir}/time_series_{filename}.png', dpi=300)
        plt.close(fig)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Plot time series.')
    parser.add_argument('exp', type=str, help='Experiment name')

    # Parse command-line arguments
    args = parser.parse_args()

    exp = args.exp
    plot_time_series(exp)

if __name__ == "__main__":
    main()
