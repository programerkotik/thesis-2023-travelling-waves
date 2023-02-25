import neo
import quantities as pq
from scipy import signal
from neurodsp.filt import filter_signal
from elephant.spike_train_generation import peak_detection
import elephant.spike_train_synchrony as sync
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.signal as signal  # import signal library

def filter_line_noise(data, fs, Q):
    """
    Removes specific frequencies (60 Hz and 120 Hz) from a signal using a notch filter.
    """
    freqs = [60.0, 120.0]  # frequencies to be removed from signal
    
    # Loop through each frequency to remove
    for freq in freqs:
        # Design notch filter
        b_notch, a_notch = signal.iirnotch(freq, Q, fs)
        
        # Apply notch filter to the data using signal.filtfilt
        data = signal.filtfilt(b_notch, a_notch, data)
    
    return data


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

def define_upstate_regions(data, times, threshold_scalar=2, binning_size=1, plot=False):
    """
    Define upstate regions throughout all channels based on threshold.

    Parameters
    ----------
    data : np.array
        Time series from multiple channels of EcoG recordings.
    threshold_criteria : float
        Threshold criteria for defining upstate regions.
    binning_value : int
        Binning value to bin the data.
    
    Returns
    -------
    regions : array
        Binary array of upstate regions (0 is downstate, 1 is upstate) of same size as data channel.
    """

    # data_normalized = (data - data.min()) / (data.max() - data.min()) # normalize data to 0-1. IS IT NEEDED? I suggest yes, because we will use one threshold for all channels. 

    
    # print(data.shape, np.min(data), np.max(data))
    threshold_values = np.mean(data, axis=1) + (np.std(data, axis=1) * threshold_scalar) # define threshold as mean + std
    # print(threshold_values)

    data_binorized = data.copy()
    for i, (d, th) in enumerate(zip(data_binorized, threshold_values)):
        d = np.where(d > th, 1, 0) # 0 is upstate, 1 is downstate
        data_binorized[i] = d
    data_binorized = np.array(data_binorized)

    # print(data_binorized.shape)
    pop_act = np.sum(data_binorized, axis=0)
    print(pop_act.shape, pop_act)

    zeros_ids = np.where(pop_act==0)[0]
    print(zeros_ids.shape)

    zeros_times = times[zeros_ids]
    print(zeros_times)
    
    event_times = []
    for i in tqdm(range(len(zeros_ids)-1)):
        if (zeros_ids[i+1] - zeros_ids[i]) > 1:
            event_times.append((zeros_times[i], zeros_times[i+1]))

    return event_times, threshold_values[0]
    