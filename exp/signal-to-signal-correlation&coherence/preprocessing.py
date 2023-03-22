import sys
import os
import logging
import yaml
import numpy as np
import neurodsp as ndsp
from pathlib import Path
import quantities as pq

# Set up file paths
file_path = str(Path().absolute())
project_path = str(Path().absolute().parent.parent)

print(file_path)
print(project_path)

os.chdir(project_path)
sys.path.append(project_path)

from src import load_dataset


def filter_data(data, f_range=(1, 100), fs=1000 * pq.Hz):
    """Filter a multi-channel signal with a bandpass filter."""
    filtered_data = []
    for channel in range(len(data)):
        filtered_channel = ndsp.filt.filter_signal(data[channel], f_range=f_range, fs=fs, pass_type='bandpass')
        filtered_data.append(filtered_channel)
    return np.array(filtered_data)


def remove_nans(data):
    """Remove any NaN values from a multi-channel signal."""
    cleaned_data = []
    for channel in range(len(data)):
        cleaned_channel = data[channel][~np.isnan(data[channel])]
        cleaned_data.append(cleaned_channel)
    return np.array(cleaned_data)


def update_times(times, offset):
    """Update the time array to match the length of the filtered signal."""
    updated_times = []
    for time_channel in times:
        updated_channel = time_channel[int(offset):-int(offset)]
        updated_times.append(updated_channel)
    return updated_times


def save_data(data, times, output_dir):
    """Save the processed data and times arrays to binary files."""
    # get ecog data and remove it from the dictionary
    ts_filtered_ecog = data.pop('ecog')
    times_ecog = times.pop('ecog')
    
    for key, value in data.items():
        np.save(output_dir / f'{key}.npy', value)
    np.save(output_dir / 'times.npy', times)

    # save ecog data and times separately
    np.save(output_dir / 'ecog.npy', ts_filtered_ecog)
    np.save(output_dir / 'times_ecog.npy', times_ecog)

def main(input_dir, output_dir):
    """Load, filter, and save ECoG and probe data."""
    # Load data
    ts_filtered_probe1, times_probe1, ts_filtered_probe2, times_probe2, ts_filtered_ecog, times_ecog = load_dataset(input_dir, 'ibw', ecog=True, probe=True, probe2=True, fs=1000 * pq.Hz)
    
    print(len(ts_filtered_ecog))
    # Filter data
    data = {'probe1': ts_filtered_probe1, 'probe2': ts_filtered_probe2}
    filtered_data = {key: filter_data(value) for key, value in data.items()}
    cleaned_data = {key: remove_nans(value) for key, value in filtered_data.items()}

    # process ecog data separately
    ts_filtered_ecog = filter_data(ts_filtered_ecog)
    ts_filtered_ecog = remove_nans(ts_filtered_ecog)
    print(len(ts_filtered_ecog))
    logging.info('Data filtered')

    # Update times
    offsets = [len(data[key][0]) - len(value[0]) for key, value in cleaned_data.items()]
    offsets = [offset / 2 for offset in offsets]
    updated_times = {key: update_times(value, offset) for key, value, offset in zip(data.keys(), [times_probe1, times_probe2], offsets)}


    updated_times['ecog'] = update_times(times_ecog, offsets[0])
    cleaned_data['ecog'] = ts_filtered_ecog
    logging.info('Times updated')
    
    print(len(cleaned_data['ecog']))
    # Save filtered data and updated times to binary files
    save_data(cleaned_data, updated_times, output_dir)

    # Save filter parameters as YAML file
    with open(output_dir / 'bandpass_kwargs.yml', 'w') as outfile:
        yaml.dump({'f_range': (1, 100), 'fs': 1000 * pq.Hz, 'pass_type': 'bandpass'}, outfile)

    logging.info('Data processing complete')


if __name__ == '__main__':

    input_dir = Path('raw/2022-09-27_data/1kHz')
    output_dir = Path('exp/ecog-probes-coherence/data')

    logging.basicConfig(level=logging.INFO)
    main(input_dir, output_dir)
    logging.info('Done')


