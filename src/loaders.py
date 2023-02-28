import neo
import quantities as pq

def load_ncs_data(path, sampling_rate=1000. * pq.Hz):
    """
    Reads in data from a file and converts it to a neo.AnalogSignal object.
    """
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
    """
    Reads in data from a file and converts it to a neo.AnalogSignal object.
    """
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

