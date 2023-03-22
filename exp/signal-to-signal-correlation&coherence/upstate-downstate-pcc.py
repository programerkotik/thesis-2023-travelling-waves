import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
# Set up file paths
file_path = str(Path().absolute())
project_path = str(Path().absolute().parent.parent)

os.chdir(project_path)
sys.path.append(project_path)

from src.utils import split_intervals, lagged_correlation

input_dir = f"{project_path}/exp/signal-to-signal-correlation&coherence/data"

files = os.listdir(input_dir)
for file in files:
    if file.endswith('.npy'):
        if file == 'ecog.npy':
            ecog_data = np.load(input_dir + '/' + file, allow_pickle=True)
        if file == 'probe1.npy':
            probe1_data = np.load(input_dir + '/' + file, allow_pickle=True)
        if file == 'probe2.npy':
            probe2_data = np.load(input_dir + '/' + file, allow_pickle=True)
        if file == 'times_ecog.npy':
            times = np.load(input_dir + '/' + file, allow_pickle=True)
            times = times[0]
            
# Compute average of ecog data along the 0th axis
ecog_data_avg = np.mean(ecog_data, axis=0)

# Make sure all signals are the same length
assert len(probe1_data[0]) == len(ecog_data_avg)
assert len(probe2_data[0]) == len(ecog_data_avg)

# read event times
upstates = np.load(f"{input_dir}/event_times.npy", allow_pickle=True)
# Get downstates as intervals between upstates
downstates = []
for i in range(len(upstates) - 1):
    downstates.append((upstates[i][1], upstates[i + 1][0]))

# # Plot ecog_data_avg and upstate/downstate intervals
# fig, ax = plt.subplots()
# ax.plot(times[0], ecog_data_avg)
# for upstate in upstates:
#     ax.axvspan(upstate[0], upstate[1], color='green', alpha=0.2)
# for downstate in downstates:
#     ax.axvspan(downstate[0], downstate[1], color='red', alpha=0.2)
# plt.show()

# Organize probe data into upstate/downstate intervals
probe1_data_upstates, probe1_data_downstates = split_intervals(probe1_data, times, upstates, downstates)
probe2_data_upstates, probe2_data_downstates = split_intervals(probe2_data, times, upstates, downstates)
ecog_data_upstates, ecog_data_downstates = split_intervals(ecog_data_avg, times, upstates, downstates)
ecog_data_upstates = ecog_data_upstates[0]
ecog_data_downstates = ecog_data_downstates[0]


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
print(correlation_data.head())

# save table to csv
correlation_data.to_csv(f"{input_dir}/correlation_data.csv", index=False)
