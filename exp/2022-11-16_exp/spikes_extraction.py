import neo
from glob import glob
from utils import read_spikes_from_channels
from elephant.statistics import mean_firing_rate
from elephant.spike_train_generation import StationaryPoissonProcess
import quantities as pq
import matplotlib.pyplot as plt

# Read the file paths for all the data
lfp_data_paths = sorted(glob('data/w12_18_nlx_2021_11_12(20khz_sampl_rate_spontan)/*.ncs'))

# Identify the range of channel IDs for each probe
ids = range(65,113)

# Find the file paths for the LFP data for each channel on probe 1
lfp_paths = []
for channel_id in ids:
    filename_lfp_probe1 = f'data/w12_18_nlx_2021_11_12(20khz_sampl_rate_spontan)/csc{channel_id}_lfp.ncs'
    filename_lfp_probe2 = f'data/w12_18_nlx_2021_11_12(20khz_sampl_rate_spontan)/csc{channel_id}_lfp_0001.ncs'
    if filename_lfp_probe1 in lfp_data_paths:
        lfp_paths.append(filename_lfp_probe1)
    elif filename_lfp_probe2 in lfp_data_paths:
        lfp_paths.append(filename_lfp_probe2)

# Extract the spikes from the LFP data
spiketrains = read_spikes_from_channels(lfp_paths)

# Create a control data with the same firing rate but random spikes
spiketrains_control = []
for i in range(len(spiketrains)):
    firing_rate = mean_firing_rate(spiketrains[i])
    t_stop = spiketrains[i].t_stop
    t_start = spiketrains[i].t_start
    st = StationaryPoissonProcess(rate=firing_rate, t_start=t_start, t_stop=t_stop).generate_spiketrain()
    spiketrains_control.append(st)

# Get the filenames for each channel
filenames = [path.split('/')[-1].split('_')[0] for path in lfp_paths]

# Save the spikes to a file for each channel
for (spiketrain, spiketrain_control, filename) in zip(spiketrains, spiketrains_control, filenames):
    # Create a Neo Segment object to store the spike trains
    segment = neo.Segment()
    segment_control = neo.Segment()

    # Add the spike trains to the segment
    segment.spiketrains.append(spiketrain)
    segment_control.spiketrains.append(spiketrain_control)

    # Save the block to a file using AsciiSpikeTrainIO
    neo.io.AsciiSpikeTrainIO(filename='data/spiking_data/'+filename+'_spikes.txt').write_segment(segment)
    neo.io.AsciiSpikeTrainIO(filename='data/spiking_data_control/'+filename+'_spikes.txt').write_segment(segment_control)
