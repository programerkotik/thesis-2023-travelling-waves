import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_time_series(data, times, xlim=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    print(len(data), len(times))

    for i, d in enumerate(data):
        ax.plot(times[i], d - 10e4*i*np.mean(data), label='Channel {}'.format(i+1))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage (mV)')
    ax.set_xlim(xlim)
    plt.show()

def plot_filtered_data(times_ecog, sig_filt, ts_filtered_ecog, fig_output_dir, xlim=None):
    sig_filt_mean = np.mean(sig_filt, axis=0)
    ts_filtered_ecog_mean = np.mean(ts_filtered_ecog, axis=0)

    fig =  plt.figure(figsize=(60, 10))
    plt.plot(times_ecog[0], sig_filt_mean, label='filtered', color='red')
    plt.fill_between(times_ecog[0], sig_filt_mean - np.std(sig_filt, axis=0), sig_filt_mean + np.std(sig_filt, axis=0), color='red', alpha=0.2)
    plt.plot(times_ecog[0], ts_filtered_ecog_mean, label='raw', color='blue')
    plt.fill_between(times_ecog[0], ts_filtered_ecog_mean - np.std(ts_filtered_ecog, axis=0), ts_filtered_ecog_mean + np.std(ts_filtered_ecog, axis=0), color='blue', alpha=0.2)
    plt.legend()
    plt.xlim(xlim)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (uV)')
    plt.title('Time series of filtered ECoG data')
    fig.savefig(fig_output_dir + f'/filtered_data_example_{str(xlim[0])}-{str((xlim[1]))}.png')

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

def plot_filtered_data_with_upstates(times_ecog, sig_filt, threshold_value, event_times, fig_output_dir, xlim=None):
    sig_filt_mean = np.mean(sig_filt, axis=0)

    fig =  plt.figure(figsize=(60, 10))
    plt.plot(times_ecog[0], sig_filt_mean, label='filtered', color='red')
    plt.fill_between(times_ecog[0], sig_filt_mean - np.std(sig_filt, axis=0), sig_filt_mean + np.std(sig_filt, axis=0), color='red', alpha=0.2)
    for i in range(len(event_times)):
        plt.axvspan(event_times[i][0], event_times[i][1], alpha=0.2, color='green')
    plt.axhline(y=threshold_value, color='black', linestyle='--')
    plt.legend()
    plt.xlim(xlim)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (uV)')
    plt.title('Time series of filtered ECoG data')
    fig.savefig(fig_output_dir + f'/filtered_data_example_with_upstates_{str(xlim[0])}-{str((xlim[1]))}.png')
