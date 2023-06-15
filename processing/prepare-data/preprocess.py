'''This script is supposed to load raw data (.ibw or .ncs), filter line noise and save as .npy file in the output directory. '''

# Import modules
import sys
import os
import logging
from pathlib import Path
import numpy as np
import quantities as pq
import yaml

# Set up file paths
file_path = str(Path().absolute())
project_path = str(Path().absolute().parent.parent)
data_name = '2022-09-27-data'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Log file path
logging.info(f"Current file directory: {file_path}")
logging.info(f"Current project directory: {project_path}")

# Change working directory to project path; this is done to make sure we can import custom modules and load data that lay upstream in the directory structure (i.e. in the src/ and raw/ directories)
os.chdir(project_path)
sys.path.append(project_path)

# Import custom modules
from src.loaders import load_dataset

# Read parameters from yml file
parameters_path = f"{file_path}/data_params.yml"

# Load the YAML file and read the parameters
with open(parameters_path) as f:
    d = yaml.load(f.read(), Loader=yaml.FullLoader)
    parameters = d['params']

sampling_rates = parameters[0]['sampling_rate'] # Sampling rate in Hz of the data
formats = parameters[1]['format'] # Format of the data

# Log parameters
logging.info(f"Sampling rate: {sampling_rates}")
logging.info(f"Format: {formats}")

for sr, ft in zip(sampling_rates, formats):
    print(f"Sampling rate: {sr}")
    print(f"Format: {ft}")

    # Specify input and output directories
    input_dir = f"{project_path}/data/raw/{data_name}/{int(sr/1000)}kHz"
    output_dir = f"{project_path}/data/processed/{data_name}/{int(sr/1000)}kHz"

    # Log input and output directories
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    if sr==1000:
        ecog, probe1, probe2 = True, True, True
    if sr==20000:
        ecog, probe1, probe2 = False, True, True
        
    #! Load data WARNING: ECOG DATA ARE TOO LARGE TO LOAD IN MEMORY IF 20 KHZ, DON'T EVEN TRY :) 
    ts_filtered_probe1, times_probe1, ts_filtered_probe2, times_probe2, ts_filtered_ecog, times_ecog = load_dataset(input_dir, ft, ecog=ecog, probe=probe1, probe2=probe2, fs=sr * pq.Hz)

    times = times_probe1 # Setting times to ecog channel, since it is the same for all channels and probes

    # Save data
    if probe1:
        np.save(f"{output_dir}/ts_filtered_probe1.npy", ts_filtered_probe1)
    if probe2:
        np.save(f"{output_dir}/ts_filtered_probe2.npy", ts_filtered_probe2)
    if ecog:
        np.save(f"{output_dir}/ts_filtered_ecog.npy", ts_filtered_ecog)
    
    np.save(f"{output_dir}/times.npy", times)

    logging.info(f"Data saved in {output_dir}")

    # Also save txt with data description
    with open(f"{output_dir}/data_description.txt", "w") as f:
        f.write(f"Sampling rate: {sr} Hz, Raw data format: {ft}. Data were filtered for line noise (60 Hz). \n Data were saved as .npy files: \n- ts_filtered_probe1 contains the filtered data of the first probe, \n- times_probe1 contains the corresponding timestamps, \n- ts_filtered_probe2 contains the filtered data of the second probe, \n- times_probe2 contains the corresponding timestamps, \n- ts_filtered_ecog contains the filtered data of the ecog, \n- times_ecog contains the corresponding timestamps.\nMore information about raw data can be found in the dscr.md file in the raw directory.")  

# Log end of script
logging.info("Data preprocessed and saved.")
