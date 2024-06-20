from typing import Hashable

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import re
import json
import seaborn as sns
from io import StringIO
import os
import time
import argparse


def main():
    parser = argparse.ArgumentParser(description="Process and visualize data")
    parser.add_argument("filename", help="The path to the data file", default="output")
    parser.add_argument("-s", "--smoothing", help="Smoothing method (max, mean, none)",
                        choices=['max', 'mean', 'none'], default="none")

    args = parser.parse_args()

    file_path = args.filename
    smoothing_method = args.smoothing
    print(smoothing_method)

    os.makedirs('graphics', exist_ok=True)
    sns.set_palette("tab10")

    df = read_and_filter_data(file_path)
    print(df)
    Generate_Statisitcs(df)
    plot_histogram(file_path + ".json")

    if smoothing_method in ['max', 'mean', 'none']:
        print(smoothing_method)
        plot_timeseries(df, smoothing_method)


def read_and_filter_data(file_path):
    """Reads and filters the file to extract CPU, Loop, and Latency (and optionally SMI count)."""
    pattern = re.compile(r'^\s*(\d+):\s+(\d+):\s+(\d+)(?:\s+(\d+))?\s*$')

    with open(file_path, 'r') as file:
        filtered_data = [
            [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4) or 0)]
            for line in file if (match := pattern.match(line))
        ]

    return pd.DataFrame(filtered_data, columns=['CPU', 'Tick', 'Latency', 'SMI Count'])


def plot_timeseries(df, sm='max'):
    """Generates a combined plot and individual CPU-specific plots with explicit color access."""

    custom_colors = sns.color_palette()

    plt.figure()
    plot_name = ""
    for cpu, group in df.groupby('CPU'):
        window_size = max(len(group) // 100, 1)
        if sm == 'max':
            latency_values = group['Latency'].rolling(window=window_size, min_periods=1).max()
            plot_name = "Max"
        elif sm == 'mean':
            latency_values = group['Latency'].rolling(window=window_size, min_periods=1).mean()
            plot_name = "Mean"
        else:
            latency_values = group['Latency']
            plot_name = "Unfiltered"

        plt.plot(group['Tick'], latency_values, linestyle='-', label=f'CPU {cpu}', alpha=0.7,
                 color=custom_colors[cpu - 1])

    plt.title(f'{plot_name} CPU Latencies over Time'.title())
    plt.xlabel('Loop')
    plt.ylabel(f'Latency (μs)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'graphics/timeseries_{plot_name}.pdf')
    plt.savefig(f'graphics/timeseries_{plot_name}.png')

    plt.close()

    for cpu, group in df.groupby('CPU'):
        plt.figure(figsize=(10, 6))
        window_size = max(len(group) // 100, 1)
        if sm == 'max':
            latency_values = group['Latency'].rolling(window=window_size, min_periods=1).max()
        elif sm == 'mean':
            latency_values = group['Latency'].rolling(window=window_size, min_periods=1).mean()
        else:
            latency_values = group['Latency']

        plt.plot(group['Tick'], latency_values, linestyle='-', label=f'CPU {cpu}', alpha=0.7,
                 color=custom_colors[cpu - 1])
        plt.title(f'CPU {cpu} {plot_name} Latency over Time'.title())
        plt.xlabel('Loop')
        plt.ylabel(f'Latency (μs)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'graphics/timeseries_{cpu}_{plot_name}.pdf')
        plt.savefig(f'graphics/timeseries_{cpu}_{plot_name}.png')

        plt.close()


def plot_histogram(file_path, step=50, maxrange=400):
    """Generates two histograms.
    @param maxrange: maximum value for the x-axis
    @param file_path: path to the json file
    @param step: step size for the x-axis labels
    """

    # Read the json file
    with open(file_path, 'r') as file:
        data = json.load(file) # We need the json data for the histogram labels

    # Create a datagram of all histograms (Thread | Latency | Count)
    histograms = []
    for thread_id, thread_data in data['thread'].items():
        for latency, count in thread_data['histogram'].items():
            histograms.append({'Thread': thread_id, 'Latency': int(latency), 'Count': count})

    dataframe = pd.DataFrame(histograms)
    grouped = dataframe.groupby('Thread')
    grouped_by_latency = dataframe.groupby('Latency')['Count'].sum()

    custom_colors = sns.color_palette()

    min_latency, max_latency, avg_latency, total_samples, overflows = {}, {}, {}, {}, {}
    legend_labels = []
    # Generate a histogram for each CPU
    thread_id: int
    for thread_id, group in grouped:
        fig_singles, ax_singles = plt.subplots()
        ax_singles.bar(group['Latency'], group['Count'], width=1, log=True, color=custom_colors[int(thread_id) - 1],
                       linewidth=1)
        ax_singles.set_title(f'CPU {thread_id} Latency Histogram'.title())
        ax_singles.set_xlabel('Latency (μs)')
        ax_singles.set_ylabel('Number of latency samples'.title())
        ax_singles.set_xlim(0, maxrange)
        ax_singles.set_xticks(np.arange(0, 401, step))
        ax_singles.grid(True)

        # Calculate the statistics once and save them for the legend
        min_latency[thread_id] = data['thread'][thread_id]['min']
        max_latency[thread_id] = data['thread'][thread_id]['max']
        avg_latency[thread_id] = data['thread'][thread_id]['avg']
        total_samples[thread_id] = sum(group['Count'])
        overflows[thread_id] = data['thread'][thread_id]['cycles'] - total_samples[thread_id]

        info_text = (f"CPU {thread_id}, Total: {total_samples[thread_id]}, Min: {min_latency[thread_id]}, "
                     f"Avg: {avg_latency[thread_id]}, Max: {max_latency[thread_id]}, Overflows: {overflows[thread_id]}")
        ax_singles.text(0.98, 0.95, info_text, horizontalalignment='right', verticalalignment='top',
                        transform=ax_singles.transAxes, fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

        plt.tight_layout()
        plt.savefig(f'graphics/histogram_{thread_id}.pdf')
        plt.savefig(f'graphics/histogram_{thread_id}.png')

        plt.close()

    # Generate a histogram with all CPUs summed up
    fig_merged, ax_merged = plt.subplots()
    ax_merged.bar(grouped_by_latency.index, grouped_by_latency, width=1, log=True, color=custom_colors[0],
                  linewidth=1)
    ax_merged.set_title('Latency Histogram (All CPUs summed up)'.title())
    ax_merged.set_xlabel('Latency (μs)')
    ax_merged.set_ylabel('Number of latency samples'.title())
    ax_merged.set_xlim(0, maxrange)
    ax_merged.set_xticks(np.arange(0, 401, step))
    ax_merged.grid(True)
    info_text = (f"Total: {sum(total_samples.values())}, Min: {min(min_latency.values())}, "
                 f"Avg: {sum(avg_latency.values())/len(avg_latency)}, Max: {max(max_latency.values())},"
                 f" Overflows: {sum(overflows.values())}")

    ax_merged.legend(info_text)

    plt.tight_layout()
    plt.savefig('graphics/histogram.pdf')
    plt.savefig('graphics/histogram.png')
    plt.close()


def Generate_Statisitcs(dataframe):
    """Generates quartiles for the dataframe containing the time series data."""
    quartiles = dataframe.groupby('CPU')['Latency'].quantile([0.25, 0.5, 0.75, 0.9, 0.99]).unstack()
    print(quartiles)
    quartiles.to_json('quartiles.json', orient='split')

if __name__ == "__main__":
    main()