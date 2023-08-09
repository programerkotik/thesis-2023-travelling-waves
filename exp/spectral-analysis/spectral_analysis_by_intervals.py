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
import argparse

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
from src.utils import split_intervals

def main(args):
    # Specify the studied experiments
    exp = args.exp

    # Specify input directory
    input_dir = f'{project_path}/data/processed/{exp}'

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
            if file == 'event_times_inverted.npy':
                event_times = np.load(input_dir + '/' + file, allow_pickle=True)


    # Get downstates as intervals between event_times
    downstates = []
    for i in range(len(event_times) - 1):
        downstates.append([event_times[i][1], event_times[i + 1][0]])

    # Perform analysis twice for both probes and save the data
    data = [ecog_data, probe1_data, probe2_data]
    names = ['ECoG', 'Probe_1', 'Probe_2']

    # Split data into upstate and downstate intervals
    spectral_properties = defaultdict(list)

    for d, name in zip(data, names):
        # Logging
        logging.info(f'Probe: {name} ...')
        logging.info(f'Splitting intervals ...')
        # Split data into upstate and downstate intervals
        probe_upstates, probe_downstates = split_intervals(d, times, event_times, downstates)
        data_dict = {'upstate': probe_upstates, 'downstate': probe_downstates}

        for state, probe_data in data_dict.items():
            logging.info(f'State: {state} ...')

            for channel, values in tqdm(probe_data.items()):
                for i, interval in enumerate(values):

                    # Skip if interval is too short
                    if len(interval) < 500:
                        continue

                    # Save spectral properties
                    spectral_properties['Name of probe'].append(name)
                    spectral_properties['Interval id'].append(i)
                    spectral_properties['State'].append(state)
                    spectral_properties['Channel'].append(channel)

                    if name == 'ECoG':
                        spectral_properties['Depth'].append('0')
                    else:
                        spectral_properties['Depth'].append(int(channel * 100))

                    # Compute power spectrum using Welch's method (probably, however, it will still include only 1 window of data (500 samples))
                    freqs, powers = compute_spectrum_welch(interval, fs=1000, nperseg=500, f_range=(0, 100),  noverlap=0)

                    # Get total power in the 1-100 Hz range
                    total_power = np.where((np.array(freqs) >= 1) & (np.array(freqs) <= 100))
                    total_power = np.sum(np.array(powers)[total_power], dtype=np.float64)

                    # Get power in alpha (8-12 Hz) and beta (15-30 Hz) and gamma (30-80 Hz) bands
                    alpha_band = np.where((np.array(freqs) >= 8) & (np.array(freqs) <= 12))
                    alpha_power = np.mean(np.array(powers)[alpha_band], dtype=np.float64)

                    beta_band = np.where((np.array(freqs) >= 15) & (np.array(freqs) <= 30))
                    beta_power = np.mean(np.array(powers)[beta_band], dtype=np.float64)

                    gamma_band = np.where((np.array(freqs) >= 30) & (np.array(freqs) <= 80))
                    gamma_power = np.mean(np.array(powers)[gamma_band], dtype=np.float64)

                    spectral_properties['Alpha power'].append(alpha_power)
                    spectral_properties['Beta power'].append(beta_power)
                    spectral_properties['Gamma power'].append(gamma_power)
                    spectral_properties['Total power'].append(total_power)

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
    output_dir = f"{project_path}/res/spectral-analysis/{exp}"

    # create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pd.DataFrame(spectral_properties).to_csv(f"{output_dir}/spectral_properties_{exp}_inverted.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze spectral properties of upstate and downstate intervals in Probes data.")
    parser.add_argument("exp", type=str, help="The name of the experiment (e.g., 'w12_07.spont').")
    args = parser.parse_args()

    main(args)