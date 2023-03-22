from .loaders import load_ncs_data
from .utils import filter_signal
import numpy as np
import quantities as pq
import neo
from elephant.spike_train_generation import peak_detection
from tqdm import tqdm
import elephant.spike_train_synchrony as sync

def read_spikes(path, sampling_rate=20000. * pq.Hz, threshold_cutoff=4.2):
    """
    Reads in data from a file and detects spikes in the data.
    """
    raw_signal = load_ncs_data(path, sampling_rate=sampling_rate)
   
    # Filter signal with highpass filter
    fs=sampling_rate.magnitude
    f_range = (300, 3000)  # range of frequencies to pass through filter, in Hz
    filtered_signal = filter_signal(np.squeeze(raw_signal.magnitude), fs, 'bandpass', f_range, return_filter=False)
    
    # Compute standard deviation of filtered signal
    filtered_signal = filtered_signal.astype('float64')
    filtered_signal = filtered_signal[~np.isnan(filtered_signal)]  # remove NaN values
    std = np.std(filtered_signal)
    
    # Convert signal to neo.AnalogSignal object
    hf_sig = neo.AnalogSignal(filtered_signal, units='mV', sampling_rate=raw_signal.sampling_rate, t_start=raw_signal.t_start, t_stop=raw_signal.t_stop)
    
    # Detect peaks below 6 times the standard deviation
    threshold = float(np.mean(filtered_signal) - (std * threshold_cutoff))* pq.mV
    st = peak_detection(hf_sig, threshold=threshold, sign='below')
    # Create a Neo SpikeTrain object from the spike times
    st = neo.SpikeTrain(st, t_stop=raw_signal.t_stop, t_start=raw_signal.t_start, units='s') 
    # Return detected peaks
    return st


def read_spikes_from_channels(path_list, save=False):
    """
    Reads in data from multiple files and detects spikes in the data.
    """
    spiketrains = []  # list to store detected spike trains
    # Loop through each channel's data
    for path in tqdm(path_list):
        # Call read_spikes function to detect spikes in this channel's data
        st = read_spikes(path)
        # Append detected spike train for this channel to list
        spiketrains.append(st)
    
    # Filter out highly synchronized spikes
    spiketrains = filter_synchronized_spikes(spiketrains)
    return spiketrains


def filter_synchronized_spikes(spiketrains):
    """
    Filters out highly synchronized spikes.
    """
    # Create a Synchrotool object with the input spike train
    synchrotool = sync.Synchrotool(spiketrains, sampling_rate=20000. * pq.Hz)

    # Filter out highly synchronized spikes
    filtered_spike_train = synchrotool.delete_synchrofacts(threshold=1.5, in_place=True, mode='delete')

    # Return filtered spike train
    return filtered_spike_train

def compute_waveform(lfp, times, dtctd, fs=20000, f_range=(300, 3000)):
    """
    Compute the waveforms of high frequency oscillations in the lfp signal around the spike times.
    
    Parameters:
    lfp (ndarray): The lfp signal.
    times (ndarray): Array of times corresponding to the lfp signal.
    dtctd (ndarray): Array of detected spike times.
    fs (int): The sampling frequency of the lfp signal (default: 20000).
    f_range (tuple): The range of frequencies to pass through the filter (default: (300, 3000)).
    
    Returns:
    waveforms (list): A list of waveforms.
    """
    # Filter the lfp frequencies from 300 to 3000 Hz
    lfp_filtered = filter_signal(np.squeeze(lfp), fs, 'highpass', f_range, return_filter=False)
    waveforms = []
    # Extract waveforms around the detected spike times
    for i in dtctd:
        # Get the index of the spike time
        idx = np.where(times == i)[0][0]
        # Get the waveform
        waveform = lfp_filtered[idx-30:idx+30]
        # Append the waveform to the list
        waveforms.append(waveform)
    return waveforms