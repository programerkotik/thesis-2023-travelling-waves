## ! Inverted upstate detection
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import sys
import os
import logging
from pathlib import Path
import numpy as np
import quantities as pq
import yaml
import argparse
from neurodsp.filt import filter_signal
import seaborn as sns

def find_upstates(exp_name):
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
    from src.utils import define_upstate_regions

    # Read parameters from yml file
    parameters_path = f"{file_path}/find_upstates.yml"

    # Set plotting style
    sns.set_context('paper', font_scale=2, rc={'lines.linewidth': 2})
    sns.set_palette('colorblind')
    sns.set_style('white')

    # Set color palette and style
    color_palette = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    sns.set_palette(color_palette)

    # Load the YAML file and read the parameters
    with open(parameters_path) as f:
        d = yaml.load(f.read(), Loader=yaml.FullLoader)
        parameters = d['params']

    freq_range = parameters[0]['f_range']
    threshold_scalar = parameters[1]['threshold_scalar']

    logging.info(f"Frequency range for low-pass filtering: {freq_range}")
    logging.info(f"Threshold scalar for upstate detection: {threshold_scalar}")

    # Specify input and output directories
    input_dir = f"{project_path}/data/processed/{exp_name}"
    output_dir = f"{project_path}/data/processed/{exp_name}"
    fig_output_dir = f"{project_path}/res/upstate_qc/{exp_name}"

    # Create output directory if it does not exist
    os.makedirs(fig_output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Load only ECoG data
    ecog_path = f"{input_dir}/Probe1_lfps_spont.npy"
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


    # Set GridSpec for the figure
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(2, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0:2]) # Filtered signal snippet 
    ax2 = fig.add_subplot(gs[0, 2]) # Interval durations histogram
    ax3 = fig.add_subplot(gs[-1, :]) # Upstates with filtered signal snippet

    # Plot filtered time series snippet
    ax1.plot(times_ecog, ts_ecog[0], label='Raw signal', alpha=0.4, color='tab:cyan')
    ax1.plot(times_ecog, sig_filt[0], label='Filtered signal', color='tab:red')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Voltage')
    ax1.set_title('Filtered ECoG time series snippet, channel 0')
    ax1.legend()
    ax1.set_xlim(0, 10)

    # Plot upstate/downstate duration histogram
    ax2.hist(upstate_durations, bins=100, label='Upstate duration', alpha=0.5)
    ax2.hist(downstate_durations, bins=100, label='Downstate duration', alpha=0.2)
    ax2.set_xlabel('Duration (s)')
    ax2.set_ylabel('Count')
    ax2.set_title('Upstate/downstate duration histogram')
    ax2.legend()

    # Plot upstates with filtered time series snippet
    ax3.plot(times_ecog, sig_filt[0], label='Filtered signal')
    # Add standard deviation lines
    ax3.fill_between(times_ecog, sig_filt[0] - np.std(sig_filt[0]), sig_filt[0] + np.std(sig_filt[0]), alpha=0.4, label='Standard deviation')
    # Plot upstate regions
    for i in range(len(event_times)):
        ax3.axvspan(event_times[i][0], event_times[i][1], color='green', alpha=0.2)
    # Add threshold line
    ax3.axhline(y=threshold_value, linestyle='--', label='Threshold', alpha=0.8, color='black')
    ax3.legend()
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Voltage')
    ax3.set_title('Upstates with filtered time series snippet')
    ax3.set_xlim(110, 120)

    # Add letters to subplots
    ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, size=20)
    ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, size=20)
    ax3.text(-0.1, 1.1, 'C', transform=ax3.transAxes, size=20)

    # Save figure
    plt.savefig(f"{fig_output_dir}/upstate_detection.png", dpi=300)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Find upstates.')
    parser.add_argument('exp_name', type=str, help='Experiment name')

    # Parse command-line arguments
    args = parser.parse_args()

    exp_name = args.exp_name
    find_upstates(exp_name)

if __name__ == "__main__":
    main()
