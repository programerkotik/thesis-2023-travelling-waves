import sys
import os
import logging
from pathlib import Path
from glob import glob
import argparse
import yaml
import quantities as pq
import numpy as np

def load_dataset(exp):
    # Set up file paths
    file_path = str(Path().absolute())
    project_path = str(Path().absolute().parent.parent)

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Log file path
    logging.info(f"Current file directory: {file_path}")
    logging.info(f"Current project directory: {project_path}")

    # Change working directory to project path
    os.chdir(project_path)
    sys.path.append(project_path)

    # Import custom modules
    from src.loaders import process_data

    # Read parameters from yml file
    parameters_path = f"{file_path}/data_params.yml"

    # Load the YAML file and read the parameters
    with open(parameters_path) as f:
        d = yaml.load(f.read(), Loader=yaml.FullLoader)
        parameters = d['params']

    sampling_rates = parameters[0]['sampling_rate']
    formats = parameters[1]['format']

    # Log parameters
    logging.info(f"Sampling rate: {sampling_rates}")
    logging.info(f"Format: {formats}")

    # Specify input and output directories
    input_dir = f"{project_path}/data/raw/{exp}"
    output_dir = f"{project_path}/data/processed/{exp}"

    # Log input and output directories
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    for sr, ft in zip(sampling_rates, formats):
        # Log start of script
        logging.info("Preprocessing data...")
        logging.info(f"Sampling rate: {sr} Hz, Format: {ft}")

        # Get all the files in the input directory
        files = glob(f"{input_dir}/*.{ft}")

        # Load and save data
        for file in files:
            name = file.split('/')[-1].split('.')[0]

            # Load data
            fs = sr * pq.Hz
            ts_filtered, times = process_data(file, ft, fs)

            # Save data
            np.save(f"{output_dir}/{name}.npy", ts_filtered)
            np.save(f"{output_dir}/times.npy", times)

            logging.info(f"Data saved in {output_dir}")

            # Log end of script
            logging.info("Data preprocessed and saved.")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Preprocess raw data.')
    parser.add_argument('exp', type=str, help='Experiment name')

    # Parse command-line arguments
    args = parser.parse_args()

    exp = args.exp
    load_dataset(exp)

if __name__ == "__main__":
    main()
