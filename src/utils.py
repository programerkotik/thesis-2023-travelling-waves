import neo
import quantities as pq
from scipy import signal
from neurodsp.filt import filter_signal
from elephant.spike_train_generation import peak_detection
import elephant.spike_train_synchrony as sync
import numpy as np
from tqdm import tqdm
import logging
from scipy.signal import correlate, correlation_lags
from scipy.stats import pearsonr
from collections import defaultdict
from tqdm import tqdm

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

def define_upstate_regions(data, times, threshold_scalar=2):
    """
    Define upstate regions throughout all channels based on threshold.

    Parameters
    ----------
    data : np.array
        Time series from multiple channels of EcoG recordings.
    times : np.array
        Time stamps, one array for all channels.
    threshold_scalar : int
        Scalar to multiply standard deviation by to define threshold.
    
    Returns
    -------
    event_times : list
        List of tuples containing start and end times of upstate regions.
    threshold_value : float
        Threshold value used to define upstate regions for first channel.
    """


    # data_normalized = (data - data.min()) / (data.max() - data.min()) # normalize data to 0-1. IS IT NEEDED? I suggest yes, because we will use one threshold for all channels. 

    
    # print(data.shape, np.min(data), np.max(data))
    
    # data_normalized = (data - data.min()) / (data.max() - data.min()) # normalize data to 0-1. IS IT NEEDED? I suggest yes, because we will use one threshold for all channels. 

    
    # print(data.shape, np.min(data), np.max(data))
    threshold_values = np.mean(data, axis=1) + (np.std(data, axis=1) * threshold_scalar) # define threshold as mean + std

    data_binorized = data.copy()

    for i, (d, th) in enumerate(zip(data_binorized, threshold_values)):
        logging.info(f"Channel {i} threshold: {th}")
        d = np.where(d > th, 1, 0) # 0 is upstate, 1 is downstate
        data_binorized[i] = d
    data_binorized = np.array(data_binorized)

    pop_act = np.sum(data_binorized, axis=0)
    zeros_ids = np.where(pop_act==0)[0]
    zeros_times = times[zeros_ids]
    event_times = []
    for i in tqdm(range(len(zeros_ids)-1)):
        if (zeros_ids[i+1] - zeros_ids[i]) > 1:
            event_times.append((zeros_times[i], zeros_times[i+1]))

    return event_times, threshold_values[0]



def lagged_correlation(probe_data, ecog_data):
    """
    Computes lagged correlation between probe data and ecog data.
    
    Parameters
    ----------
    probe_data : np.array
        Probe data from a single channel.
    ecog_data : np.array
        Averaged across channels EcoG data.
    
    Returns
    -------
    lag : int
        Lag with maximum correlation.
    corr : float
        Maximum correlation.
    """

    # Calculate Pearson correlation coefficient between probe data and ecog data average
    pcc = pearsonr(probe_data, ecog_data)
    # If the correlation coefficient is negative, flip the probe data
    if pcc[0] < 0:
        probe_data = -probe_data

    # Compute cross-correlation between the probe data and ecog data average
    correlation = correlate(probe_data, ecog_data, mode="full")
    # Compute lags using the correlation_lags function
    lags = correlation_lags(probe_data.size, ecog_data.size, mode="full")
    # Get the lag with the maximum correlation using argmax
    lag = lags[np.argmax(correlation)]
    # Compute Pearson correlation coefficient between probe data and ecog data average with the lag
    max_pcc = pearsonr(probe_data, np.roll(ecog_data, lag))[0]

    return pcc[0], lag, max_pcc

def make_splits(data, times, intervals):
        splits = defaultdict(list)
        # Split data into upstate/downstate intervals
        for interval in tqdm(intervals):
            # Find index of interval start and end
            i, j = np.searchsorted(times, interval)

            # Skip if interval is of length 0
            if i == j: continue
            
            # If there is only one channel
            if len(data.shape) == 1:
                # Split data into intervals
                splits[0].append(data[i:j])
            else: 
                # Split data into intervals
                for i, d in enumerate(data[:, i:j]):
                    splits[i].append(d)
        return splits

# Get signal in upstate/downstate intervals
def split_intervals(data, times, upstates, downstates):
    # Split data into upstate intervals
    upstate_splits = make_splits(data, times, upstates)
    # Split data into downstate intervals
    downstate_splits = make_splits(data, times, downstates)
    return upstate_splits, downstate_splits

def parse_spectrum(x, y):
    freqs = []
    for x_i in x:
        x_i = x_i.replace('[', '').replace(']', '').replace(' ', '').replace('\n', '').split('.')
        x_i = [int(i) for i in x_i if i != '']
        freqs.append(x_i)
    powers =[]
    for y_i in y:
        y_i = y_i.replace('[', '').replace(']', '').replace('\n', '').split(' ')
        y_i = [float(i) for i in y_i if i != '']
        powers.append(y_i)
    return freqs, powers
