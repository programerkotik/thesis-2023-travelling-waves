import sys
import os
from pathlib import Path
import numpy as np
from collections import defaultdict
import pandas as pd
import argparse

def compute_correlation(exp):
    # Set up file paths
    file_path = str(Path().absolute())
    project_path = str(Path().absolute().parent.parent)

    os.chdir(project_path)
    sys.path.append(project_path)

    from src.utils import split_intervals, lagged_correlation

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

    # Compute average of ecog data along the 0th axis
    ecog_data_avg = np.mean(ecog_data, axis=0)

    # Make sure all signals are the same length
    assert len(probe1_data[0]) == len(ecog_data_avg)
    assert len(probe2_data[0]) == len(ecog_data_avg)


    # Get downstates as intervals between event_times
    downstates = []
    for i in range(len(event_times) - 1):
        downstates.append((event_times[i][1], event_times[i + 1][0]))

    # Organize probe data into upstate/downstate intervals
    probe1_data_upstates, probe1_data_downstates = split_intervals(probe1_data, times, event_times, downstates)
    probe2_data_upstates, probe2_data_downstates = split_intervals(probe2_data, times, event_times, downstates)
    ecog_data_upstates, ecog_data_downstates = split_intervals(ecog_data_avg, times, event_times, downstates)
    ecog_data_upstates = ecog_data_upstates[0]
    ecog_data_downstates = ecog_data_downstates[0]


    # Create a dictionary to store correlation data
    correlation_data_upstate = defaultdict(list)

    # Loop over each probe
    for k, probe_data_upstates in enumerate([probe1_data_upstates, probe2_data_upstates]):
        # Loop over each channel in a probe
        for n, probe_data_upstates_ch in probe_data_upstates.items():
            # Loop over each upstate window in a channel
            for probe_wind, ecog_wind in zip(probe_data_upstates_ch, ecog_data_upstates):
                if len(probe_wind) < 2:
                    continue

                # Compute lag and correlation
                pcc, lag, max_pcc = lagged_correlation(probe_wind, ecog_wind)

                # Append results to lists
                correlation_data_upstate['pcc'].append(pcc)
                correlation_data_upstate['max_pcc'].append(max_pcc)
                correlation_data_upstate['lag'].append(lag)
                correlation_data_upstate['channel'].append(n + 1)
                correlation_data_upstate['probe'].append(k + 1)
                correlation_data_upstate['interval'] = 'upstate'

    correlation_data_downstate = defaultdict(list)

    # Loop over each probe
    for k, probe_data_downstates in enumerate([probe1_data_downstates, probe2_data_downstates]):
        # Loop over each channel in a probe
        for n, probe_data_downstates_ch in probe_data_downstates.items():
            # Loop over each downstate window in a channel
            for probe_wind, ecog_wind in zip(probe_data_downstates_ch, ecog_data_downstates):
                if len(probe_wind) < 2:
                    continue

                # Compute lag and correlation
                pcc, lag, max_pcc = lagged_correlation(probe_wind, ecog_wind)

                # Append results to lists
                correlation_data_downstate['pcc'].append(pcc)
                correlation_data_downstate['max_pcc'].append(max_pcc)
                correlation_data_downstate['lag'].append(lag)
                correlation_data_downstate['channel'].append(n + 1)
                correlation_data_downstate['probe'].append(k + 1)
                correlation_data_downstate['interval'] = 'downstate'

    # Concatenate upstate and downstate data
    correlation_data = pd.concat([pd.DataFrame(correlation_data_upstate), pd.DataFrame(correlation_data_downstate)])
    
    # Save table to CSV
    correlation_data.to_csv(f"{output_dir}/correlation_data.csv", index=False)

    print(correlation_data.head())

def main():
    parser = argparse.ArgumentParser(description="Compute signal-to-signal correlation.")
    parser.add_argument("exp", type=str, help="Experiment name.")
    args = parser.parse_args()

    compute_correlation(args.exp)

if __name__ == "__main__":
    main()