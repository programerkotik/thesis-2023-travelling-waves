import neo
import quantities as pq
import os
from glob import glob
from natsort import natsorted
import numpy as np
from .utils import filter_line_noise
from .plotting import plot_time_series_snippet, plot_filtered_time_series_snippet

def load_ncs_data(path, sampling_rate=1000. * pq.Hz):
    '''
    Reads in data from a file and converts it to a neo.AnalogSignal object. Sampling rate might not be required since it is already in the file.
    '''
    # Read in analog signal from file
    reader = neo.io.NeuralynxIO(filename=path)
    block = reader.read_block()
    assert len(block.segments) == 1
    assert len(block.segments[0].analogsignals) == 1
    signal = block.segments[0].analogsignals[0]

    # Convert signal to neo.AnalogSignal object with correct sampling rate and times
    # times = (signal.times.magnitude / 1000) * pq.s # Convert times (milliseconds) to seconds
    times = signal.times.magnitude * pq.s #! It seems like the times are already in seconds
    t_start = signal.t_start 
    # t_stop = (signal.t_stop.magnitude / 1000) * pq.s # Convert times (milliseconds) to seconds
    t_stop = signal.t_stop.magnitude * pq.s #! It seems like the times are already in seconds

    analog_signal = neo.AnalogSignal(signal, sampling_rate=signal.sampling_rate, times=times, t_start=t_start, t_stop=t_stop)

    return analog_signal, times

def load_ibw_data(path, sampling_rate=1000. * pq.Hz):
    '''
    Reads in data from a file and converts it to a neo.AnalogSignal object.
    '''
    # Read in analog signal from file
    reader = neo.io.IgorIO(filename=path)
    block = reader.read_block()
    assert len(block.segments) == 1
    assert len(block.segments[0].analogsignals) == 1
    signal = block.segments[0].analogsignals[0]

    # Convert signal to neo.AnalogSignal object with correct sampling rate and times
    times = (signal.times.magnitude / 1000) * pq.s # convert times (milliseconds) to seconds
    t_start = signal.t_start 
    t_stop = (signal.t_stop.magnitude / 1000) * pq.s # convert times (milliseconds) to seconds
    analog_signal = neo.AnalogSignal(signal, sampling_rate=sampling_rate, times=times, t_start=t_start, t_stop=t_stop)

    return analog_signal, times

def process_data(files, format, fs = 1000. * pq.Hz, electrode='probe1', plot=True):
    '''
    Load the data and filter for line noise (60Hz, 120Hz).

    Parameters
    ----------
    files : list
        List of file paths to load
    format : str
        Format of the data
    fs : float
        Sampling rate of the data
    electrode : str
        Name of the electrode
    plot : bool
        Whether to plot the example of data
    '''

    if plot:
        # Create output directory if it does not exist 
        os.makedirs('res/processing', exist_ok=True)

    if format == 'ncs':
        # Load the ncs data
        data = []
        for f in files:
            print('Loading: ' + f)
            print(fs)
            d, t = load_ncs_data(f, fs)
            data.append(d)
        times = t
    elif format == 'ibw':
        # Load the ibw data
        data = []
        for f in files:
            print('Loading: ' + f)
            print(fs)
            d, t = load_ibw_data(f, fs)
            data.append(d)
        times = t

    ts = np.array([np.squeeze(d.magnitude) for d in data])

    if plot:
        # Generate random time range
        t_start = np.random.randint(0, len(times) - 50000)
        t_end = t_start + 50000

        # Plot the time series
        plot_time_series_snippet(ts, times, xlim=(times[t_start], times[t_end]), channel=np.random.randint(0, len(ts)), filename=str(electrode), fig_output_dir='res/processing')

    # Filter the data for line noise
    ts_filtered = np.array([filter_line_noise(ts, fs.magnitude, 550) for ts in ts])

    if plot:
        # Generate random time range
        t_start = np.random.randint(0, len(times) - 5000)
        t_end = t_start + 5000

        # Plot the filtered time series
        plot_filtered_time_series_snippet(ts, ts_filtered, times, xlim=(times[t_start], times[t_end]), channel=np.random.randint(0, len(ts)), filename=str(electrode), fig_output_dir='res/processing')

    return ts_filtered, times

def load_dataset(path, file_format, ecog = True, probe = True, probe2 = True, fs = 1000. * pq.Hz):
    '''
    Load the dataset; probe1, probe2, ecog data, assumes specific names and numeration of the channels, change if doesn't apply.

    Parameters
    ----------
    path : str
        Path to the data
    file_format : str
        File format of the data
    ecog : bool
        Whether to load the ecog data
    probe : bool
        Whether to load the probe data
    probe2 : bool
        Whether to load the probe2 data
    fs : float
        Sampling rate of the data

    Returns
    -------
    ts_filtered_probe1 : list
        Filtered (against 60Hz line noise) probe1 data
    times_probe1 : list
        Times of the probe1 data
    ts_filtered_probe2 : list
        Filtered (against 60Hz line noise) probe2 data
    times_probe2 : list
        Times of the probe2 data
    ts_filtered_ecog : list
        Filtered (against 60Hz line noise) ecog data
    times_ecog : list
        Times of the ecog data
    '''
    # Read the data
    data_path = sorted(glob(os.path.join(path, '*.' + file_format)))

    ts_filtered_probe1, times_probe1, ts_filtered_probe2, times_probe2, ts_filtered_ecog, times_ecog = None, None, None, None, None, None
    
    if file_format == 'ncs':
            if probe:
                # Get 1st probe data paths; 65-80
                files_1st_probe = natsorted([d for d in data_path if int(d.split('/')[-1].split('.')[0].split('_')[0].split('c')[-1]) >= 65 and int(d.split('/')[-1].split('.')[0].split('_')[0].split('c')[-1]) <= 80])
                ts_filtered_probe1, times_probe1 = process_data(files_1st_probe, file_format, fs, electrode='probe1')
                
            if probe2: 
                # Get 2nd probe data paths; 97-112
                files_2nd_probe = natsorted([d for d in data_path if int(d.split('/')[-1].split('.')[0].split('_')[0].split('c')[-1]) >= 97 and int(d.split('/')[-1].split('.')[0].split('_')[0].split('c')[-1]) <= 112])
                ts_filtered_probe2, times_probe2 = process_data(files_2nd_probe, file_format, fs, electrode='probe2')
            
            if ecog:
                # Get ecog data; 1-64
                files_ecog = natsorted([d for d in data_path if int(d.split('/')[-1].split('.')[0].split('_')[0].split('c')[-1]) >= 1 and int(d.split('/')[-1].split('.')[0].split('_')[0].split('c')[-1]) <= 64])
                ts_filtered_ecog, times_ecog = process_data(files_ecog, file_format, fs, electrode='ecog')
    
    if file_format == 'ibw':       
        if probe:
            # Get 1st probe data paths; 65-80
            files_1st_probe = natsorted([d for d in data_path if int(d.split('/')[-1].split('.')[0].split('p')[-1]) >= 65 and int(d.split('/')[-1].split('.')[0].split('p')[-1]) <= 80])
            ts_filtered_probe1, times_probe1 = process_data(files_1st_probe, file_format, fs, electrode='probe1')
            
        if probe2: 
            # Get 2nd probe data paths; 97-112
            files_2nd_probe = natsorted([d for d in data_path if int(d.split('/')[-1].split('.')[0].split('p')[-1]) >= 97 and int(d.split('/')[-1].split('.')[0].split('p')[-1]) <= 112])
            ts_filtered_probe2, times_probe2 = process_data(files_2nd_probe, file_format, fs, electrode='probe2')
        
        if ecog:
            # Get ecog data; 1-64
            files_ecog = natsorted([d for d in data_path if int(d.split('/')[-1].split('.')[0].split('p')[-1]) >= 1 and int(d.split('/')[-1].split('.')[0].split('p')[-1]) <= 64])
            ts_filtered_ecog, times_ecog = process_data(files_ecog, file_format, fs, electrode='ecog')

    return ts_filtered_probe1, times_probe1, ts_filtered_probe2, times_probe2, ts_filtered_ecog, times_ecog
    