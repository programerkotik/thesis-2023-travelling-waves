import neo
import quantities as pq
from pathlib import Path
import os
from glob import glob
from natsort import natsorted
import numpy as np
from .utils import filter_line_noise

def load_ncs_data(path, sampling_rate=1000. * pq.Hz):
    '''
    Reads in data from a file and converts it to a neo.AnalogSignal object.
    '''
    # Read in analog signal from file
    reader = neo.io.NeuralynxIO(filename=path)
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

def process_data(files, format, fs = 1000. * pq.Hz):
    '''
    Load the data and filter for line noise
    '''
    if format == 'ncs':
        # load the data
        data, times = [], []
        for f in files:
            print('Loading: ' + f)
            d, t = load_ncs_data(f)
            data.append(d)
            times.append(t)

    elif format == 'ibw':
        # load the data
        data, times = [], []
        for f in files:
            print('Loading: ' + f)
            d, t = load_ibw_data(f)
            data.append(d)
            times.append(t)
            
    ts = [np.squeeze(d.magnitude) for d in data]
    # filter the data for line noise
    ts_filtered = [filter_line_noise(ts, fs.magnitude, 550) for ts in ts]
    return ts_filtered, times

def load_dataset(path, file_format, ecog = True, probe = True, probe2 = True, fs = 1000. * pq.Hz):
    '''
    Load the first dataset
    '''
    # Set up file paths
    project_path = str(Path().absolute())
    data_path = os.path.join(project_path, path)
    # read the data
    data = sorted(glob(os.path.join(data_path, '*.' + file_format)))
    ts_filtered_probe1, times_probe1, ts_filtered_probe2, times_probe2, ts_filtered_ecog, times_ecog = None, None, None, None, None, None
    
    if probe:
        # get 1st probe data paths
        files_1st_probe = natsorted([d for d in data if int(d.split('/')[-1].split('.')[0].split('p')[-1]) >= 65 and int(d.split('/')[-1].split('.')[0].split('p')[-1]) <= 80])
        ts_filtered_probe1, times_probe1 = process_data(files_1st_probe, file_format, fs)
        
    if probe2: 
        # get 2nd probe data paths
        files_2nd_probe = natsorted([d for d in data if int(d.split('/')[-1].split('.')[0].split('p')[-1]) >= 97 and int(d.split('/')[-1].split('.')[0].split('p')[-1]) <= 112])
        ts_filtered_probe2, times_probe2 = process_data(files_2nd_probe, file_format, fs)
    
    if ecog:
        # get ecog data
        files_ecog = natsorted([d for d in data if int(d.split('/')[-1].split('.')[0].split('p')[-1]) >= 1 and int(d.split('/')[-1].split('.')[0].split('p')[-1]) <= 64])
        ts_filtered_ecog, times_ecog = process_data(files_ecog, file_format, fs)

    return ts_filtered_probe1, times_probe1, ts_filtered_probe2, times_probe2, ts_filtered_ecog, times_ecog
    