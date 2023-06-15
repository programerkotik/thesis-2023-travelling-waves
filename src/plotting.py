import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import logging
import matplotlib.style as style
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_time_series_snippet(ts, time, xlim=None, channel=None, filename=None, fig_output_dir='res'):
    """
    Plot time series of ts. ts are multichannel (e.g. multiple electrodes). Plot random channel if channel is not specified. If channel is -1 then plot all channels.
 
    Parameters
    ----------
    ts : np.array
        Time series from multiple channels of recordings.
    time : np.array
        Time stamps, one array for all channels.
    xlim : tuple
        Tuple of floats, start and end time of the time series to plot.
    channel : int
        Channel to plot.
    file : str
        Name of the file ts is from.
    """

    # Set a figure size
    plt.figure(figsize=(20, 10))

    # Check if ts is 2D
    if len(ts.shape) != 2:
        logging.error('ts must be 2D (channels x time).')
        return
    
    # Check if time is 1D
    if len(time.shape) != 1:
        logging.error('time must be 1D.')
        return
    
    if channel is None:
        # If channel is not specified, plot a random channel
        channel = np.random.randint(ts.shape[0])

    if channel == -1:
        # If channel is -1, plot all channels
        channels = range(ts.shape[0])
    else:
        # Plot the specified channel
        channels = [channel]


    # Load style configuration from YAML file
    with open('plot_style.yml', 'r') as file:
        style_config = yaml.safe_load(file)

    # Set the plot style
    style.use(style_config['style'])
    plt.rcParams['font.size'] = style_config['font_size']
                        
    for i, ch in enumerate(channels):
        plt.plot(time, ts[ch] - 10e1*i*np.mean(ts[ch]), label=f'Channel {ch}')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage')
        plt.title(f'Time series')

        if xlim is not None:
            plt.xlim(xlim)

    plt.legend()
    # Save plot to file
    plt.savefig(f'{fig_output_dir}/time_series_snippet_{filename}.png')

def plot_filtered_time_series_snippet(ts, ts_filtered, time, xlim=None, channel=None, filename=None, fig_output_dir='res'):
    """
    Plot time series of ts and ts_filtered. ts and ts_filtered are multichannel (e.g. multiple electrodes). Plot random channel if channel is not specified. If channel is -1 then plot all channels.

    Parameters
    ----------
    ts : np.array
        Time series from multiple channels of recordings.
    ts_filtered : np.array
        Filtered time series from multiple channels of recordings.
    time : np.array
        Time stamps, one array for all channels.
    xlim : tuple
        Tuple of floats, start and end time of the time series to plot. 
    channel : int
        Channel to plot.
    file : str
        Name of the file ts is from.
    """
    # Set a figure size
    plt.figure(figsize=(20, 10))

    # Check if ts is 2D
    if len(ts.shape) != 2:
        logging.error('ts must be 2D (channels x time).')
        return
    
    # Check if ts_filtered is 2D
    if len(ts_filtered.shape) != 2:
        logging.error('ts_filtered must be 2D (channels x time).')
        return
    
    # Check if time is 1D
    if len(time.shape) != 1:
        logging.error('time must be 1D.')
        return
    
    # Load style configuration from YAML file
    with open('plot_style.yml', 'r') as file:
        style_config = yaml.safe_load(file)

    if channel is None:
        # If channel is not specified, plot a random channel
        channel = np.random.randint(ts.shape[0])

    if channel == -1:
        # If channel is -1, plot all channels
        channels = range(ts.shape[0])
    else:
        # Plot the specified channel
        channels = [channel]

    # Set the plot style
    style.use(style_config['style'])
    plt.rcParams['font.size'] = style_config['font_size']

    for i, ch in enumerate(channels):
        plt.plot(time, ts[ch] - 10e1*i*np.mean(ts[ch]), label=f'Channel {ch}', alpha=0.5)
        plt.plot(time, ts_filtered[ch] - 10e1*i*np.mean(ts_filtered[ch]), label=f'Channel {ch} filtered', alpha=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage')
        plt.title(f'Time series')

        if xlim is not None:
            plt.xlim(xlim)
        
    plt.legend()
    # Save plot to file
    plt.savefig(f'{fig_output_dir}/filtered_time_series_snippet_{filename}.png')

def plot_upstate_durations(upstate_durations, fig_output_dir):
    fig = plt.figure(figsize=(10, 10))
    sns.histplot(x=upstate_durations)
    plt.xlabel('Upstate duration (s)')
    plt.ylabel('Count')
    plt.title('Histogram of upstate durations')
    fig.savefig(fig_output_dir + '/upstate_durations_histogram.png')

def plot_downstate_durations(downstate_durations, fig_output_dir):
    fig = plt.figure(figsize=(10, 10))
    sns.histplot(x=downstate_durations)
    plt.xlabel('Downstate duration (s)')
    plt.ylabel('Count')
    plt.title('Histogram of downstate durations')
    fig.savefig(fig_output_dir + '/downstate_durations_histogram.png')

def plot_filtered_ts_with_upstates(time_ecog, sig_filt, threshold_value, event_time, fig_output_dir, xlim=None):
    sig_filt_mean = np.mean(sig_filt, axis=0)

    fig =  plt.figure(figsize=(60, 10))
    plt.plot(time_ecog[0], sig_filt_mean, label='filtered', color='red')
    plt.fill_between(time_ecog[0], sig_filt_mean - np.std(sig_filt, axis=0), sig_filt_mean + np.std(sig_filt, axis=0), color='red', alpha=0.2)
    for i in range(len(event_time)):
        plt.axvspan(event_time[i][0], event_time[i][1], alpha=0.2, color='green')
    plt.axhline(y=threshold_value, color='black', linestyle='--')
    plt.legend()
    plt.xlim(xlim)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (uV)')
    plt.title('Time series of filtered ECoG ts')
    fig.savefig(fig_output_dir + f'/filtered_ts_example_with_upstates_{str(xlim[0])}-{str((xlim[1]))}.png')
