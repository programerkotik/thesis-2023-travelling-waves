import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import seaborn as sns
import logging
from pathlib import Path
from collections import defaultdict

# Set up file paths
file_path = str(Path().absolute())
project_path = str(Path().absolute().parent.parent)
exp_name = file_path.split('/')[-1]

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"Current file directory: {file_path}")
logging.info(f"Current project directory: {project_path}")

# Set plotting style
sns.set_context('paper', font_scale=2, rc={'lines.linewidth': 2})
sns.set_palette('colorblind')
sns.set_style('white')

# Set color palette and style
color_palette = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
sns.set_palette(color_palette)

# Specify the input and output directories
input_dir = f'{project_path}/res/{exp_name}'

# Replace the file paths below with your actual data file paths
w12_18_path = f'{input_dir}/w12_07.spont/spectral_properties_w12_07.spont.csv'
w12_07_path = f'{input_dir}/w12_18.spont/spectral_properties_w12_18.spont.csv'

w12_18 = pd.read_csv(w12_18_path)
w12_07 = pd.read_csv(w12_07_path)
data_exp_names = ['w12_18.spont', 'w12_07.spont']

for df, name in zip([w12_18, w12_07], data_exp_names):
    # get probes 
    probes = df['Name of probe'].unique()

    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(1, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0]) # For probe 1
    ax2 = fig.add_subplot(gs[0, 1]) # For probe 2

    output_dir = f'{project_path}/res/{exp_name}/{name}'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for probe, ax in zip(probes, [None, ax1, ax2]):
        if probe == "ECoG":
            continue
        else:
            # get dataframes for each probe
            df_probe = df[df['Name of probe'] == probe]
                
            # Recombine the df to organize Alpha power, Beta powerand Gamma power into one column and add a column with band label (Alpha, Beta, Gamma)
            df_probe = pd.melt(df_probe, id_vars=['Depth'], value_vars=['Alpha power', 'Beta power', 'Gamma power'], var_name='band', value_name='power')

            
            # Plot barlot with vertical orientation with depth on y-axis and combining bars for Alpha power column, Beta power column, and Gamma power column
            sns.barplot(x='power', y='Depth', hue='band', data=df_probe, orient='h', ax=ax)
            # Set y-axis label
            ax.set_ylabel('Depth')
            # Set x-axis label
            ax.set_xlabel('Power')
            # Set title
            ax.set_title(f'{name}:{probe}')
            # Set legend
            ax.legend(loc='upper right')
                
            # Add letter labels to subplots
            ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, fontsize=18)
            ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, fontsize=18)

            plt.savefig(f'{output_dir}/bands_resolved_spectral_props.png', dpi=300, bbox_inches='tight')