import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from pathlib import Path
import pandas as pd
from matplotlib.gridspec import GridSpec
import argparse
from scipy.stats import mannwhitneyu

def visualize_pcc():
    # Set up file paths
    file_path = str(Path().absolute())
    project_path = str(Path().absolute().parent.parent)

    os.chdir(project_path)
    sys.path.append(project_path)

    # Set plotting style
    sns.set_context('paper', font_scale=2, rc={'lines.linewidth': 2})
    sns.set_palette('colorblind')
    sns.set_style('white')

    # Set color palette and style
    color_palette = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    sns.set_palette(color_palette)

    output_dir = f"{project_path}/res/signal-to-signal-correlation"

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read data
    data_07 = pd.read_csv('res/signal-to-signal-correlation/w12_07.spont/correlation_data.csv')
    data_18 = pd.read_csv('res/signal-to-signal-correlation/w12_18.spont/correlation_data.csv')


    data_07['experiment'] = 'w12_07'
    data_18['experiment'] = 'w12_18'

    # Concatenate dataframes
    data = pd.concat([data_07, data_18])
    # Set GridSpec for the figure
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0]) # Violin plot for experiment 1
    ax2 = fig.add_subplot(gs[0, 1]) # Violin plot for experiment 2
    ax3 = fig.add_subplot(gs[1, 0:2]) # Graph for both experiments

    my_colors = {'upstate': 'tab:green', 'downstate': 'tab:orange'}
     
    # Set up violin plots
    # Plot histograms of lag values for upstate and downstate
    sns.distplot(data[data['interval'] == 'upstate']['max_pcc'], ax=ax1, kde=False, norm_hist=True, label='upstate', color='tab:green')
    sns.distplot(data[data['interval'] == 'downstate']['max_pcc'], ax=ax1, kde=False, norm_hist=True, label='downstate', color='tab:orange')

    # Set legend
    ax1.legend(title='Interval', loc='upper left')
    # Test for statistical significance for each experiment between upstate and downstate
    print('w12_07')
    stat, p = mannwhitneyu(data[data['interval'] == 'upstate']['max_pcc'], data[data['interval'] == 'downstate']['max_pcc'], alternative='less')
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')
    print('w12_18')

    # Test for statistical significance for each experiment between upstate and downstate
    stat, p = mannwhitneyu(data_18[data_18['interval'] == 'upstate']['max_pcc'], data_18[data_18['interval'] == 'downstate']['max_pcc'])
    print('Statistics=%.3f, p=%.3f' % (stat, p))

    # Add title and axis labels
    ax1.set_title('Distribution of maximum PCC values')
    # ax2.set_title('w12_18')
    ax1.set_xlabel('Maximum PCC values')


    # Plot histograms of lag values for upstate and downstate
    sns.distplot(data[data['interval'] == 'upstate']['lag'], ax=ax2, kde=False, norm_hist=True, label='upstate', color='tab:green')
    sns.distplot(data[data['interval'] == 'downstate']['lag'], ax=ax2, kde=False, norm_hist=True, label='downstate', color='tab:orange')

    # Set legend
    ax2.legend(title='Interval', loc='upper left')
    # Set title and axis labels
    ax2.set_title('Distribution of lag values')
    ax2.set_xlabel('Lag (ms)')    
    # Compute depth as channel number multiplied by 100 microns and add it as a column to the dataframe
    data['depth'] = data['channel'] * 100

    # Set up graph, on x axis put channel multiply by 100 microns to get depth, on y axis put box plots of PCC values, plot intervals with different colors and experiments with different markers
    sns.boxplot(x='depth', y='pcc', data=data, ax=ax3, hue='interval', showfliers=False, hue_order=['upstate', 'downstate'], palette=my_colors)

    # Add title and axis labels
    ax3.set_title('PCC values for all channels')
    ax3.set_xlabel('Depth (microns)')
    ax3.set_ylabel('Pearson correlation coefficient')
    # # Set x axis ticks to be every 100 microns
    # ax3.set_xticks(range(0, 1000, 100))
    # # Set x axis tick labels to be every 100 microns
    # ax3.set_xticklabels(range(0, 1000, 100))
    # # Set legend
    # ax3.legend(title='Interval', loc='lower left')

    # Also add letters to the subplots 
    ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, size=20)
    ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, size=20)
    ax3.text(-0.1, 1.1, 'C', transform=ax3.transAxes, size=20)

    # for each depth, compare significance between upstate and downstate
    for depth in data['depth'].unique():
        # Get upstate and downstate PCC values for this depth
        upstate_pcc = data[(data['depth'] == depth) & (data['interval'] == 'upstate')]['pcc']
        downstate_pcc = data[(data['depth'] == depth) & (data['interval'] == 'downstate')]['pcc']

        # Test for statistical significance for each experiment between upstate and downstate
        stat, p = mannwhitneyu(upstate_pcc, downstate_pcc)
        print(f"Depth: {depth}, Statistics={stat:.3f}, p={p:.3f}")

    # Save figure
    plt.savefig(f'{output_dir}/visualize_upstate_downstate_pcc.png', dpi=300, bbox_inches='tight')

def main():
    visualize_pcc()

if __name__ == "__main__":
    main()
