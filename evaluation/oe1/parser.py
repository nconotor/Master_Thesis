# Description: This script is used to parse the output of the RTLA and cyclictest tools and generate boxplots from the data.
# Creating was assisted by the following resources: Pycharm IDE, GithubCopilot

import re
import matplotlib.pyplot as plt
import json
import numpy as np
import os
import seaborn as sns
import math
import pandas as pd
import itertools
from scipy import stats


def save_to_json(json_data, output_file):
    """
    Save the given data to a JSON file.
    @param json_data: The data to save.
    @param output_file: The path to the output file.
    @return: None
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as file:
        json.dump(json_data, file, indent=4)


def parse_cyclictest_verbose_data(file_path):
    """
    Reads and filters the file to extract CPU, Loop, and Latency (and optionally SMI count). Uses regex matching to
    parse the data. If SMI count is not present, it is set to -1.
    @param file_path: The path to the input file.
    @return: The filtered data as a DataFrame.  Columns: CPU, Tick, Latency, SMI Count (zero if not present).
    """
    pattern = re.compile(r'^\s*(\d+):\s*(\d+):\s*(\d+)(?:\s*(\d+))?\s*$')

    with open(file_path, 'r') as file:
        filtered_data = [
            [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4) or -1)]
            for line in file if (match := pattern.match(line))
        ]

    return pd.DataFrame(filtered_data, columns=['CPU', 'Tick', 'Latency', 'SMI Count'])


def parse_rtla_data(
        input_file='H:\\MA\\comparison_rtla_662126realtime11rtlts\\broadwellDockerStress-ng_run1\\rtla_output'):
    """
    Parse the given RTLA output file and return the data in a structured format. Uses regex matching the rtla output to parse the data.
    @param input_file: The path to the input file.
    @return: The parsed data.
    """
    num_threads = 4
    categories = ["IRQ", "THR", "USR"]
    data_line_pattern = re.compile(
        r'^(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+$')
    summary_line_pattern = re.compile(
        r'^(over|count|min|avg|max):\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+$')

    parsed_data = {
        f"thread{thread_num}": {
            category: {"hist": {}, "over": [], "count": [], "min": [], "avg": [], "max": []}
            for category in categories
        } for thread_num in range(num_threads)
    }

    with open(input_file, 'r') as file:
        if not file.readline():
            print(f"Input file is empty. File path: {input_file}")
            exit(1)
        for line in file:
            if data_line_match := data_line_pattern.match(line):
                index = int(data_line_match.group(1))

                for i in range(1, 5):
                    thread_num = i - 1
                    thread_string = f"thread{thread_num}"
                    keys = ['IRQ', 'THR', 'USR']

                    for j, key in enumerate(keys):
                        group_num = i * 3 + j - 1
                        parsed_data[thread_string][key]["hist"][index] = int(data_line_match.group(group_num))

            elif summary_line_match := summary_line_pattern.match(line):
                cat_type = summary_line_match.group(1)

                for i in range(1, 5):
                    thread_num = i - 1
                    thread_string = f"thread{thread_num}"
                    group_nums = {'IRQ': (i * 3) - 1, 'THR': i * 3, 'USR': (i * 3) + 1}

                    for key, group_num in group_nums.items():
                        parsed_data[thread_string][key][cat_type] = int(summary_line_match.group(group_num))
    enrich_hist_data(parsed_data)
    return parsed_data


def parse_cyclictest_json_data(input_file):
    """
    Parse the given cyclictest JSON output file and return the data in a structured format matching the rtla format.
    @param input_file: The path to the input file.
    @return: The parsed data.
    """
    with open(input_file, 'r') as file:
        data = json.load(file)

    reformed_data = {}
    for thread_id, thread_data in data["thread"].items():
        thread_key = "thread" + thread_id
        histogram_data = thread_data["histogram"]
        histogram_sum = sum(histogram_data.values())
        total_cycles = thread_data.get("cycles", 0)

        reformed_data[thread_key] = {
            "USR": {
                "hist": {},
                "over": max(0, total_cycles - histogram_sum),
                "count": total_cycles,
                "min": thread_data.get("min", 0),
                "avg": thread_data.get("avg", 0),
                "max": thread_data.get("max", 0)
            }
        }

        for latency, count in histogram_data.items():
            reformed_data[thread_key]["USR"]["hist"][latency] = count

    enrich_hist_data(reformed_data)
    return reformed_data


def get_percentile(size, percentile, data_set):
    """
    Get the percentile of the given aggregated data set.
    @param size: The size of all samples in the dataset.
    @param percentile: The percentile to get. Number between 0> and <100
    @param data_set: The data set to analyze.
    @return: The percentile of the data set.
    """
    if percentile > 100 or percentile < 0:
        print("Invalid quartile")
        return 0

    pos = (percentile / 100) * size
    cur_pos = 0
    for index, pair in enumerate(data_set.items()):
        cur_pos += int(pair[1])
        val = float(pair[0])
        if cur_pos >= pos:
            if size % 2 == 0:
                if cur_pos + 1 >= pos or index + 1 < len(data_set):
                    return (val + val) / 2
                else:
                    return val + float(data_set[index + 1][1])
            else:
                return val


def get_fliers(data_set, upper_limit, lower_limit):
    """
    Get the fliers in the given data set.
    @param upper_limit: The upper limit for a latency to be considered a flier.
    @param lower_limit: The lower limit for a latency to be considered a flier.
    @param data_set: The data set to analyze.
    @return: A tuple (set, count) of a set of fliers and the number of fliers in the data set.
    The number includes duplicates, the set does not.
    """
    fliers = set([])
    flierCount = 0
    for index, pair in enumerate(data_set.items()):
        if (int(pair[0]) > upper_limit or int(pair[0]) < lower_limit) and pair[1] > 0:
            fliers.add(pair[0])
            flierCount += pair[1]
    return fliers, flierCount


def enrich_hist_data(hist_data):
    """
    Create the statistics for the given histogram data. The Whiskers are calculated as the 1.5 * IQR from the Q1 and
    Q3 values and then set to the closest value in the data set inside these bounds.
    @param hist_data: The histogram data to analyze.
    @return: The enriched histogram data.
    """
    for thread_id in hist_data.keys():
        thread_data = hist_data[thread_id]
        for type in thread_data.keys():
            type_data = thread_data[type]
            hist = type_data['hist']
            type_data['HistMax'] = max((key for key, value in hist.items() if value != 0), key=int, default=None)
            type_data['HistSpan'] = int(type_data['HistMax']) - int(type_data['min'])
            type_data[('HistAvg')] = sum(int(latency) * count for latency, count in hist.items()) / sum(hist.values())
            type_data['Var'] = sum(
                count * (int(latency) - type_data['HistAvg']) ** 2 for latency, count in hist.items()) / sum(
                hist.values())
            type_data['StdDev'] = np.sqrt(type_data['Var'])
            type_data['Q1'] = get_percentile(int(type_data["count"]), 25, hist)
            type_data['Q2'] = get_percentile(int(type_data["count"]), 50, hist)
            type_data['Q3'] = get_percentile(int(type_data["count"]), 75, hist)
            type_data['P99'] = get_percentile(int(type_data["count"]), 99, hist)
            type_data['IQR'] = type_data['Q3'] - type_data['Q1']
            type_data['WhiskerLow'] = type_data['Q1'] - 1.5 * type_data['IQR']
            type_data['WhiskerHigh'] = type_data['Q3'] + 1.5 * type_data['IQR']
            type_data['WhiskerLowBounded'] = min((int(key) for key in hist if int(key) >= type_data['WhiskerLow']),
                                                 default=None)
            type_data['WhiskerHighBounded'] = max((int(key) for key in hist if int(key) <= type_data['WhiskerHigh']),
                                                  default=None)
            fliers, flier_count = get_fliers(hist, int(type_data['WhiskerHigh']), int(type_data['WhiskerLow']))
            type_data['Fliers'] = int(flier_count)
            type_data['RealFliers'] = int(flier_count) + int(type_data['over'])
            type_data['FlierPercentage'] = (type_data['RealFliers'] / sum(hist.values())) * 100
            type_data['FliersSet'] = list(fliers)
    return hist_data


def get_boxplot_data(hist_data, data_type, labels=None):
    """
    Get the data for a boxplot from the enriched histogram data.
    @param hist_data: The enriched histogram data.
    @param data_type: The data type to get the boxplot data for. Normally USR, THR or IRQ.
    @return: The data for the boxplot.
    """
    result = []
    for idx, thread_id in enumerate(hist_data.keys()):
        data_segment = hist_data[thread_id][data_type]
        result += [{
            'label': labels[idx] if labels else "Thread " + str(idx),
            'mean': float(data_segment['avg']),
            'med': float(data_segment['Q2']),
            'q1': float(data_segment['Q1']),
            'q3': float(data_segment['Q3']),
            'iqr': float(data_segment['IQR']),
            'whishi': float(data_segment['WhiskerHighBounded']),
            'whislo': float(data_segment['WhiskerLowBounded']),
            'fliers': data_segment["FliersSet"]
        }]
    return result


def gen_boxplot(boxplot_data, title=None, x_label=None, y_label="Latency (ns)",
                figure_size=(6, 6), palette="tab10", save_path=None, ylim=None, show_fliers=False, legend=False):
    """
    Generate a boxplot from the given data with Seaborn color palettes and custom labels.
    @param show_fliers: Boolean to show fliers in the boxplot.
    @param boxplot_data: The data to generate the boxplot from. Is an array of list with ['label', 'mean', 'iqr',
     'cilo', 'cihi', 'whishi', 'whislo', 'fliers', 'q1', 'med', 'q3']
    @param title: Title of the boxplot.
    @param x_label: Label for the x-axis.
    @param y_label: Label for the y-axis.
    @param figure_size: Size of the figure.
    @param palette: Name of the Seaborn color palette to use.
    @return: None
    """
    fig, axs = plt.subplots(figsize=figure_size)
    meanprops = {'marker': '^', 'markerfacecolor': 'red', 'markeredgecolor': 'red'}
    flierprops = dict(marker='o', markerfacecolor='b', markersize=1)
    box = axs.bxp(boxplot_data, showmeans=True, meanline=False, showfliers=show_fliers, patch_artist=True,
                  meanprops=meanprops, flierprops=flierprops)

    colors = sns.color_palette(palette, n_colors=len(boxplot_data))
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    if title:
        axs.set_title(title)
    if x_label:
        axs.set_xlabel(x_label)
    axs.set_ylabel(y_label)

    if ylim:
        axs.set(ylim=ylim)

    if legend:
        labels = []
        handles = []
        for i, (patch, data) in enumerate(zip(box['boxes'], boxplot_data)):
            label = f"{data['label']} (Mean: {data['mean']:.0f}, Median: {data['med']:.0f}, IQR: {data['iqr']:.0f})"
            labels.append(label)
            handles.append(patch)

        axs.legend(handles, labels, loc='upper right')

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=900)
    else:
        plt.show()
    plt.close()


def gen_histogram(hist_data, category='USR', save_path=None, title=None, figure_size=(10, 6), limit_x_to_p99=False,
                  limit_x=None):
    """
    Generate a histogram from the given data.
    @param hist_data: Dictionary containing histogram data
    @param category: Category of data to plot
    @param save_path: Path to save the plot image
    @param title: Title of the plot
    @param figure_size: Size of the figure
    @param limit_x_to_p99: Boolean to limit x-axis to P99
    @param limit_x: Custom limit for x-axis
    """
    print("Starting histogram generation...")

    plt.figure(figsize=figure_size)

    all_latencies = set()
    thread_histories = {}
    max_99 = 0
    min_latency = float('inf')
    max_latency = float('-inf')

    for thread_name, thread_data in hist_data.items():
        print(f"Processing thread: {thread_name}")
        hist = thread_data[category]['hist']
        thread_histories[thread_name] = {int(k): v for k, v in hist.items() if v != 0}
        max_99 = max(max_99, thread_data[category]['P99'])
        min_latency = min(min_latency, thread_data[category]['min'])
        max_latency = max(max_latency, max(thread_histories[thread_name].keys(), default=min_latency))
        all_latencies.update(thread_histories[thread_name].keys())

    if not all_latencies:
        print(f"No valid data found for category '{category}'")
        return

    print("Accumulating latencies and counts...")
    latencies = np.array(sorted(all_latencies))

    bottoms = np.zeros(len(latencies))
    colors = sns.color_palette("tab10", len(hist_data))

    print("Creating bar plots for each thread...")
    for thread, color in zip(hist_data.keys(), colors):
        thread_counts = np.array([thread_histories[thread].get(latency, 0) for latency in latencies])
        plt.bar(latencies, thread_counts, bottom=bottoms, color=color, label=thread, align='center')
        bottoms += thread_counts

    plt.xlabel('Latency (ns)')
    plt.ylabel('Count')

    if limit_x and limit_x_to_p99:
        print("Both limit_x and limit_x_to_p99 are set. limit_x_to_p99 will be ignored.")
    elif limit_x:
        plt.xlim(1000, limit_x)
    elif limit_x_to_p99:
        plt.xlim(0, round_to_next_x(max_99))

    if title:
        plt.title(title)

    plt.legend(title='Thread', loc='upper left')

    if save_path:
        print(f"Saving histogram to {save_path}.png and {save_path}.pdf")
        plt.savefig(f"{save_path}.png")
        plt.savefig(f"{save_path}.pdf")
    else:
        print("Displaying histogram...")
        plt.show()

    plt.close()
    print("Histogram generation completed.")


def aggregate_data(data):
    """
    Aggregate the given data by combining histograms and recalculating count, min, avg, and max for each category
    within each thread. This will lose some precision because we cannot recalculate the values for values outside
    the histogram. As some enriched values need to be recalculated, the enriched data will be recalled.
    @param data: The data to aggregate, structured as a dictionary with thread keys, each containing
    'USR', 'THR', and 'IRQ' sub-dictionaries.
    @return: The aggregated data.
    """
    aggregated_data = {"Aggregated": {}}

    for thread_key, categories in data.items():
        for category_key, details in categories.items():
            if category_key not in aggregated_data["Aggregated"]:
                aggregated_data["Aggregated"][category_key] = {
                    "hist": {},
                    "over": 0,
                    "count": 0,
                    "min": float('inf'),
                    "max": float('-inf'),
                    "avg": 0
                }

            for latency, count in details['hist'].items():
                if latency in aggregated_data["Aggregated"][category_key]["hist"]:
                    aggregated_data["Aggregated"][category_key]["hist"][latency] += count
                else:
                    aggregated_data["Aggregated"][category_key]["hist"][latency] = count

            aggregated_data["Aggregated"][category_key]["over"] += details.get("over", 0)
            aggregated_data["Aggregated"][category_key]["count"] += details["count"]
            aggregated_data["Aggregated"][category_key]["min"] = min(aggregated_data["Aggregated"][category_key]["min"],
                                                                     details["min"])
            aggregated_data["Aggregated"][category_key]["max"] = max(aggregated_data["Aggregated"][category_key]["max"],
                                                                     details["max"])

    for category in aggregated_data["Aggregated"].values():
        total_latency = sum(int(latency) * count for latency, count in category["hist"].items())
        category["avg"] = total_latency / category["count"] if category["count"] != 0 else 0
        category["hist"] = dict(
            sorted(category["hist"].items(), key=lambda item: int(item[0])))  # Sort numerically not alphabetically

    enrich_hist_data(aggregated_data)
    return aggregated_data


def gen_boxplot_from_hist(histogram_data, data_type, labels=None, title=None, x_label=None,
                          y_label="Latency (ns)", save_path=None, figure_size=(4, 6), palette="viridis",
                          align_size=True, y_limit=None, show_fliers=False, legend=False):
    """
    Generate a boxplot from the given histogram data. Wrapper function for get_boxplot_data and gen_boxplot.
    @param show_fliers: Boolean to show fliers in the boxplot.
    @param y_limit: The y-axis limits. Will be overridden by align_size.
    @param histogram_data: The histogram data to generate the boxplot from.
    @param data_type: The data type to get the boxplot data for. Normally USR, THR or IRQ.
    @param labels: The labels for the boxplot.
    @param title: Title of the boxplot
    @param x_label: Label for the x-axis
    @param y_label: Label for the y-axis
    @param save_path: Path to save the plot to.
    @return: None
    """
    bxp = get_boxplot_data(hist_data=histogram_data, data_type=data_type, labels=labels)
    y_limit = y_limit
    if align_size:
        min_value = 999999999
        max_value = -999999999

        for thread_key, thread_info in histogram_data.items():
            usr_data = thread_info.get("USR", {})
            min_v = int(usr_data["WhiskerLow"])
            max_v = int(usr_data["WhiskerHigh"])
            if min_v < min_value:
                min_value = min_v
            if max_v > max_value:
                max_value = max_v
        rounded_min_value = 0
        rounded_max_value = round_to_next_x(max_value)
        y_limit = (rounded_min_value, rounded_max_value)

    gen_boxplot(bxp, title=title, x_label=x_label, y_label=y_label, save_path=save_path, figure_size=figure_size,
                palette=palette, ylim=y_limit, show_fliers=show_fliers, legend=legend)


def round_to_next_x(number):
    """
    Round the given number to the next multiple of 10.
    @param number: The number to round.
    @return: The rounded number.
    """
    num_digits = math.floor(math.log10(number)) + 1
    return math.ceil(number / 10 ** (num_digits - 1)) * 10 ** (num_digits - 1)


def merge_data(*args, names=None):
    """
    Merge multiple dictionaries into a single dictionary, replacing their top-level keys with custom names.
    Each dictionary should have exactly one top-level key.
    @param args: Unspecified number of dictionaries to merge.
    @param names: List of new names for the top-level keys of each dictionary. Must match the number of data dictionaries.
    @return: A dictionary with each input dictionary nested under a uniquely renamed key.
    """
    if not names or len(names) != len(args):
        print("A list of names must be provided, and its length must match the number of data dictionaries.")
        return 1

    merged_data = {}
    for name, data_dict in zip(names, args):
        if len(data_dict) != 1:
            print("Each dictionary must have exactly one top-level key.")
            return 1
        original_key = next(iter(data_dict))
        merged_data[name] = data_dict[original_key]

    enrich_hist_data(merged_data)
    return merged_data


def get_var_name(var):
    """
    Get the name of a variable from the global scope.
    @param var: The variable to get the name of.
    @return: The name of the variable.
    """
    for name, value in globals().items():
        if value is var:
            return name


def gen_barplot(data, attribute, experiment_type, x_axis_label="Data Sample", y_axis_label="Latency (ns)",
                save_path=None, y_lim=None, shorten_labels=False, figsize=(10, 6), print_size=True):
    """
    Generate a bar plot for the specified attribute within a given experiment type across multiple top_level_keys.

    @param data: Histogram data to plot
    @param attribute: String specifying the attribute to plot (e.g., 'max', 'min', 'avg')
    @param experiment_type: String specifying the type of experiment ('IRQ', 'THR', 'USR')
    @param x_axis_label: Label for the x-axis
    @param y_axis_label: Label for the y-axis
    @param save_path: String specifying the path to save the plot
    @param y_lim: Tuple containing the lower and upper limits for the y-axis
    @param shorten_labels: Boolean indicating if the bar labels should be shortened to the first letter of each word
    @param figsize: Size of the figure (width, height)
    @param print_size: Boolean indicating if the bar values should be displayed on the bars

    @return: None, saves the plot to the specified path or displays it
    """
    values = []
    top_level_keys = []
    bar_names = []

    for thread, content in data.items():
        if experiment_type in content:
            values.append(content[experiment_type][attribute])
            top_level_keys.append(thread)
            if shorten_labels:
                parts = thread.split()
                shortened = []
                for part in parts:
                    if 'stress-ng' in part.lower():
                        shortened.append('N')
                    elif 'stress' in part.lower():
                        shortened.append('S')
                    else:
                        shortened.append(part[0])
                bar_names.append(''.join(shortened))
            else:
                bar_names.append(thread)

    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("viridis", n_colors=len(top_level_keys))
    bars = ax.bar(top_level_keys, values, color=colors)

    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    pos = np.arange(len(top_level_keys))
    ax.set_xticks(pos)
    ax.set_xticklabels(bar_names, rotation=90)

    if print_size:
        for bar in bars:
            yval = bar.get_height() / 2
            if isinstance(bar.get_height(), float):
                label = f'{bar.get_height():.2f}'
            else:
                label = f'{bar.get_height():d}'
            ax.text(bar.get_x() + bar.get_width() / 2, yval, label, ha='center', va='center', rotation=90, color='red')

    if y_lim:
        ax.set_ylim(y_lim)

    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}.pdf")
        plt.savefig(f"{save_path}.png", dpi=900)
    else:
        plt.show()
    plt.close()


def gen_tex_table(data, list_of_attributes, experiment_type, save_path):
    """
    TODO Needs to be finished
    @param data:
    @param list_of_attributes:
    @param experiment_type:
    @param save_path:
    @return:
    """
    header = 'Experiment & ' + ' & '.join([attr.capitalize() for attr in list_of_attributes]) + ' \\\\ \\hline'
    rows = [header]

    for experiment, experiment_data in data.items():
        if experiment_type in experiment_data:
            values = experiment_data[experiment_type]
            row = f"{experiment}"
            for attribute in list_of_attributes:
                if attribute in values:
                    row += f" & ${values[attribute]:.0f}$" if not values[
                        attribute].is_integer() else f" & ${int(values[attribute])}$"
                else:
                    row += " & -"
            row += " \\\\\\hline"
            rows.append(row)

    latex_table = '\\begin{tabular}{|' + 'c|' * (len(list_of_attributes) + 1) + '}\n\\hline\n'
    latex_table += '\n'.join(rows) + '\n\\end{tabular}'

    with open(save_path, 'w') as file:
        file.write(latex_table)

    return latex_table


def calculate_change(old_value, new_value):
    """
    Calculate the absolute and percentage change between two values.
    @param old_value: The old value.
    @param new_value: The new value.
    @return: A tuple containing the absolute and percentage change.
    """
    absolute_change = new_value - old_value
    if old_value != 0:  # Avoid division by zero
        percentage_change = (absolute_change / old_value) * 100
    else:
        percentage_change = float('inf')

    return absolute_change, percentage_change


def gen_timeseries(df, sm='max', save_path=None, title=None, ylim=None, palette='tab10'):
    """
    Plot the timeseries of the given data frame.
    @param df: The data frame to plot. Needs to contain 'CPU', 'Tick', and 'Latency' columns.
    @param sm: The smoothing method to use. Can be 'max', 'mean', or 'none'.
    @param save_path: The folder path to save the plot to.
    @param title: The title of the plot. TODO Not implemented yet. Currently toggle to turn title off.
    @param ylim: The y-axis limits.
    @param palette: The Seaborn color palette to use.
    """
    custom_colors = sns.color_palette(palette, n_colors=len(df['CPU'].unique()))
    plt.figure()

    for cpu, group in df.groupby('CPU'):
        window_size = max(len(group) // 100, 1)
        if sm == 'max':
            latency_values = group['Latency'].rolling(window=window_size, min_periods=1).max()
        elif sm == 'min':
            latency_values = group['Latency'].rolling(window=window_size, min_periods=1).min()
        elif sm == 'mean':
            latency_values = group['Latency'].rolling(window=window_size, min_periods=1).mean()
        else:
            latency_values = group['Latency']
        plt.plot(group['Tick'], latency_values, linestyle='-', label=f'CPU {cpu}', alpha=0.5,
                 color=custom_colors[cpu % len(custom_colors)])

    plt.xlabel('Loops')
    plt.ylabel('Latency (ns)')
    plt.legend(loc='upper right')
    plt.grid(True)
    if title:
        plt.title(title)
    if ylim:
        plt.ylim(ylim)

    plt.tight_layout()

    if save_path:
        plt.savefig(f'{save_path}.pdf')
        plt.savefig(f'{save_path}.png', dpi=900)
    else:
        plt.show()
    plt.close()


def plot_change(data, save_path=None, figsize=(12, 8), statistical_parameter="Q2"):
    """
    Plot the latency for each metric and annotate absolute and relative changes.
    Add a third legend for changes between native and dockerized variants.
    @param statistical_parameter: The statistical parameter to plot. Default is the median.
    @param save_path: The path to save the plot to.
    @param data: The dataset containing metrics for each experiment.
    """
    colors = sns.color_palette("tab10")
    cc = itertools.cycle(colors)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    change_legend_labels = []
    plot_lines = []
    parameters = []
    metrics_dict = {}

    for idx, (experiment, metrics) in enumerate(data.items()):
        x = ['IRQ', 'THR', 'USR']
        y = [metrics['IRQ'][statistical_parameter], metrics['THR'][statistical_parameter],
             metrics['USR'][statistical_parameter]]
        parameters.append(experiment)

        parts = experiment.split()
        if len(parts) >= 3:
            platform = parts[0]
            test_type = parts[2]
        elif len(parts) == 2:
            platform = parts[0]
            test_type = parts[1]
        else:
            platform = parts[0]
            test_type = "Unknown"

        key = f"{platform} {test_type}"

        marker = 'o'
        if "Host" in parts:
            variant = "Host"
            marker = '<'
        elif "Docker" in parts:
            variant = "Docker"
            marker = '>'
        else:
            variant = "Unknown"

        if key not in metrics_dict:
            metrics_dict[key] = {}
        metrics_dict[key][variant] = y

        c = next(cc)
        line, = ax.plot(x, y, marker=marker, linestyle='-', color=c, label=experiment)
        plot_lines.append(line)

        change_label = f"{experiment}: " + " - ".join(
            f"{x[i]} to {x[i + 1]} Δ={calculate_change(y[i], y[i + 1])[0]:.2f}, "
            f"{calculate_change(y[i], y[i + 1])[1]:.2f}%"
            for i in range(len(x) - 1)
        )  # + f" - {x[0]} to {x[2]} Δ={calculate_change(y[0], y[2])[0]:.2f}, {calculate_change(y[0], y[2])[1]:.2f}%"
        change_legend_labels.append(change_label)

    variant_change_labels = []
    for key, variants in metrics_dict.items():
        if "Host" in variants and "Docker" in variants:
            host_y = variants["Host"]
            docker_y = variants["Docker"]
            variant_change_label = f"{key}: " + " - ".join(
                f"{x[i]} Δ={calculate_change(host_y[i], docker_y[i])[0]:.2f}, "
                f"{calculate_change(host_y[i], docker_y[i])[1]:.2f}%"
                for i in range(len(x))
            )
            variant_change_labels.append(variant_change_label)

    # ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Median Latency (ns)', fontsize=12)
    ax.grid(True)

    legend1 = plt.legend(change_legend_labels, fontsize=10, loc='upper left')
    ax.add_artist(legend1)
    legend3 = plt.legend(labels=variant_change_labels, fontsize=10, loc='lower right', markerfirst=False,
                         title="Change between native and dockerized variants")
    ax.add_artist(legend3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def Calc_Confidence_Effect(aggdataset, system1Name, system2Name, irq_metric="USR", save_path=None):
    """
    Calculate the confidence interval and effect size between two systems for a given IRQ metric.
    @param aggdataset:  An aggregated dataset containing the data to analyze (enriched) data.
    @param system1Name: Name of the first subset of data to compare.
    @param system2Name: Name of the second subset of data to compare.
    @param irq_metric: The IRQ metric to analyze. IRQ, THR, or USR. Irq is the default.
    @param save_path: The path to save the plot to.
    @return: A dictionary containing the results of the analysis.
    """
    hist_dict1 = aggdataset[system1Name][irq_metric]["hist"]
    hist_dict2 = aggdataset[system2Name][irq_metric]["hist"]

    expanded_array1 = np.repeat(np.fromiter(hist_dict1.keys(), dtype=int), np.fromiter(hist_dict1.values(), dtype=int))
    expanded_array2 = np.repeat(np.fromiter(hist_dict2.keys(), dtype=int), np.fromiter(hist_dict2.values(), dtype=int))
    print(f"length of expanded_array1: {len(expanded_array1)} and expanded_array2: {len(expanded_array2)}")

    mean1, std1 = np.mean(expanded_array1), np.std(expanded_array1, ddof=1)
    mean2, std2 = np.mean(expanded_array2), np.std(expanded_array2, ddof=1)

    median1 = np.median(expanded_array1)
    median2 = np.median(expanded_array2)

    t_stat, p_value_t = stats.ttest_ind(expanded_array1, expanded_array2, equal_var=False, alternative='less')
    print(f"t test: {stats.ttest_ind(expanded_array1, expanded_array2, equal_var=False, alternative='less')}")

    u_stat, p_value_u = stats.mannwhitneyu(expanded_array1, expanded_array2, alternative='less')
    print(f"u test: {stats.mannwhitneyu(expanded_array1, expanded_array2, alternative='less')}")

    pooled_std = np.sqrt((std1 ** 2 + std2 ** 2) / 2)
    cohen_d = (mean2 - mean1) / pooled_std

    alpha = 0.05
    n1, n2 = len(expanded_array1), len(expanded_array2)
    se_diff = pooled_std * np.sqrt(1 / n1 + 1 / n2)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n1 + n2 - 2)
    ci_low = cohen_d - t_crit * se_diff
    ci_high = cohen_d + t_crit * se_diff

    difference_mean = mean2 - mean1
    increase_mean = (difference_mean / mean1) * 100

    difference_median = median2 - median1
    increase_median = (difference_median / median1) * 100

    levene_stat, p_value_levene = stats.levene(expanded_array1, expanded_array2)
    print(f"Levene's test for equal variances: {levene_stat}, p-value: {p_value_levene}")

    results = {
        'means': (mean1, mean2),
        'std_devs': (std1, std2),
        'medians': (median1, median2),
        't_test': (t_stat, p_value_t),
        'mann_whitney_u': (u_stat, p_value_u),
        'effect_size': cohen_d,
        'effect_size_ci': (ci_low, ci_high),
        'difference_mean': difference_mean,
        'increase_mean': increase_mean,
        'difference_median': difference_median,
        'increase_median': increase_median,
        'levene_test': (levene_stat, p_value_levene)
    }

    return results



########################################################################################################################
# User defined variables
# Set the base path to the directory containing the output files
# Load the RTLA and cyclictest data
########################################################################################################################

basepath = "H:\\MA\\Final\\comp"
output = f"{basepath}\\output"

########################################################################################################################
# Broadwell Data
########################################################################################################################
# RTLA Data
broadwellRtlaSubFolder = f"{basepath}\\comparison_f12_broadwell_rtla_662328realtime11rtlts"

broadwellRtlaStressHost = parse_rtla_data(
    f'{broadwellRtlaSubFolder}\\HostStress_run1\\rtla_output')
broadwellRtlaStressDocker = parse_rtla_data(
    f'{broadwellRtlaSubFolder}\\DockerStress_run1\\rtla_output')
broadwellRtlaStressNgHost = parse_rtla_data(
    f'{broadwellRtlaSubFolder}\\HostStress-ng_run1\\rtla_output')
broadwellRtlaStressNgDocker = parse_rtla_data(
    f'{broadwellRtlaSubFolder}\\DockerStress-ng_run1\\rtla_output')

# Cyclictest Data
broadwellCyclictestSubFolder = f"{basepath}\\comparison_f12_broadwell_cyclic_662328realtime11rtlts"
broadwellCyclicStressHost = parse_cyclictest_json_data(
    f'{broadwellCyclictestSubFolder}\\HostStress_run1\\output.json')
broadwellCyclicStressDocker = parse_cyclictest_json_data(
    f'{broadwellCyclictestSubFolder}\\DockerStress_run1\\output.json')
broadwellCyclicStressNgHost = parse_cyclictest_json_data(
    f'{broadwellCyclictestSubFolder}\\HostStress-ng_run1\\output.json')
broadwellCyclicStressNgDocker = parse_cyclictest_json_data(
    f'{broadwellCyclictestSubFolder}\\DockerStress-ng_run1\\output.json')
########################################################################################################################
# Haswell Data
########################################################################################################################
# RTLA Data
haswellRtlaSubFolder = f"{basepath}\\comparison_f12_haswell_rtla_662328realtime11rtlts"
haswellRtlaStressHost = parse_rtla_data(
    f'{haswellRtlaSubFolder}\\HostStress_run1\\rtla_output')
haswelllRtlaStressDocker = parse_rtla_data(
    f'{haswellRtlaSubFolder}\\DockerStress_run1\\rtla_output')
haswellRtlaStressNgHost = parse_rtla_data(
    f'{haswellRtlaSubFolder}\\HostStress-ng_run1\\rtla_output')
haswellRtlaStressNgDocker = parse_rtla_data(
    f'{haswellRtlaSubFolder}\\DockerStress-ng_run1\\rtla_output')

# Cyclictest Data
haswellCyclictestSubFolder = f"{basepath}\\comparison_f12_haswell_cyclic_662328realtime11rtlts"
haswellCyclicStressHost = parse_cyclictest_json_data(
    f'{haswellCyclictestSubFolder}\\HostStress_run1\\output.json')
haswellCyclicStressDocker = parse_cyclictest_json_data(
    f'{haswellCyclictestSubFolder}\\DockerStress_run1\\output.json')
haswellCyclicStressNgHost = parse_cyclictest_json_data(
    f'{haswellCyclictestSubFolder}\\HostStress-ng_run1\\output.json')
haswellCyclicStressNgDocker = parse_cyclictest_json_data(
    f'{haswellCyclictestSubFolder}\\DockerStress-ng_run1\\output.json')

# Aggregated Data#######################################################################################################
broadwellRtlaStressHostAggregated = aggregate_data(broadwellRtlaStressHost)
broadwellRtlaStressDockerAggregated = aggregate_data(broadwellRtlaStressDocker)
haswellRtlaStressHostAggregated = aggregate_data(haswellRtlaStressHost)
haswellRtlaStressDockerAggregated = aggregate_data(haswelllRtlaStressDocker)
broadwellRtlaStressNgHostAggregated = aggregate_data(broadwellRtlaStressNgHost)
broadwellRtlaStressNgDockerAggregated = aggregate_data(broadwellRtlaStressNgDocker)
haswellRtlaStressNgHostAggregated = aggregate_data(haswellRtlaStressNgHost)
haswellRtlaStressNgDockerAggregated = aggregate_data(haswellRtlaStressNgDocker)
broadwellCyclicStressHostAggregated = aggregate_data(broadwellCyclicStressHost)
broadwellCyclicStressDockerAggregated = aggregate_data(broadwellCyclicStressDocker)
haswellCyclicStressHostAggregated = aggregate_data(haswellCyclicStressHost)
haswellCyclicStressDockerAggregated = aggregate_data(haswellCyclicStressDocker)
broadwellCyclicStressNgHostAggregated = aggregate_data(broadwellCyclicStressNgHost)
broadwellCyclicStressNgDockerAggregated = aggregate_data(broadwellCyclicStressNgDocker)
haswellCyclicStressNgHostAggregated = aggregate_data(haswellCyclicStressNgHost)
haswellCyclicStressNgDockerAggregated = aggregate_data(haswellCyclicStressNgDocker)

########################################################################################################################
# RTLA
########################################################################################################################
# Stress
specific_out = output + f"\\rtlaStress"
os.makedirs(specific_out, exist_ok=True)


rtlaStressMerged = merge_data(broadwellRtlaStressHostAggregated, broadwellRtlaStressDockerAggregated,
                              haswellRtlaStressHostAggregated, haswellRtlaStressDockerAggregated,
                              names=["Broadwell Host", "Broadwell Docker", "Haswell Host", "Haswell Docker "])

for data in [broadwellRtlaStressHost, broadwellRtlaStressDocker, haswellRtlaStressHost, haswelllRtlaStressDocker]:
    name = get_var_name(data)
    save_to_json(data, f"{specific_out}\\{name}.json")
    for type in ["USR", "THR", "IRQ"]:
        gen_boxplot_from_hist(data, type, title=f"{name} {type} Boxplot", x_label="Threads",
                              y_label="Latency (ns)",
                              save_path=f"{specific_out}\\{name}_{type}_boxplot_stress.pdf", legend=False)

save_to_json(rtlaStressMerged, f"{specific_out}\\merged.json")
type = "THR"
gen_boxplot_from_hist(rtlaStressMerged, type,
                      y_label="Latency (ns)",
                      save_path=f"{specific_out}\\merged_{type}_boxplot_stress.pdf",
                      labels=["Broadwell Host", "Broadwell Docker", "Haswell Host", "Haswell Docker "],
                      figure_size=(6, 3), y_limit=(1000, 6000), align_size=False, legend=True)
type = "USR"
gen_boxplot_from_hist(rtlaStressMerged, type,
                      y_label="Latency (ns)",
                      save_path=f"{specific_out}\\merged_{type}_boxplot_stress.pdf",
                      labels=["Broadwell Host", "Broadwell Docker", "Haswell Host", "Haswell Docker "],
                      figure_size=(6, 4), y_limit=(1000, 9000), align_size=False, legend=True)

gen_boxplot_from_hist(rtlaStressMerged, "IRQ",
                      y_label="Latency (ns)",
                      save_path=f"{specific_out}\\merged_IRQ_boxplot_stress.pdf",
                      labels=["Broadwell Host", "Broadwell Docker", "Haswell Host", "Haswell Docker "],
                      figure_size=(6, 2), y_limit=(100, 1200), align_size=False, legend=True)

gen_tex_table(rtlaStressMerged, ["avg", "max", "over", "StdDev", "P99", "HistSpan"], "USR",
              f"{specific_out}\\stats.tex")
gen_tex_table(rtlaStressMerged, ["count", "over", "max", "P99", "StdDev"], "USR",
              f"{specific_out}\\statsNew.tex")


plot_change(rtlaStressMerged, save_path=f"{specific_out}\\median_latency_stress.pdf")
plot_change(rtlaStressMerged, save_path=f"{specific_out}\\average_latency_stress.pdf", statistical_parameter="avg")

# Stress-ng ############################################################################################################
specific_out = output + f"\\rtlaStressNg"
os.makedirs(specific_out, exist_ok=True)
for data in [broadwellRtlaStressNgHost, broadwellRtlaStressNgDocker, haswellRtlaStressNgHost,
             haswellRtlaStressNgDocker]:
    name = get_var_name(data)
    save_to_json(data, f"{specific_out}\\{name}.json")
    for type in ["USR", "THR", "IRQ"]:
        gen_boxplot_from_hist(data, type, title=f"{name} {type} Boxplot", x_label="Threads",
                              y_label="Latency (ns)",
                              save_path=f"{specific_out}\\{name}_{type}_boxplot_stress-ng.pdf", legend=False)

rtlStressNgMerged = merge_data(broadwellRtlaStressNgHostAggregated, broadwellRtlaStressNgDockerAggregated,
                               haswellRtlaStressNgHostAggregated, haswellRtlaStressNgDockerAggregated,
                               names=["Broadwell Host", "Broadwell Docker", "Haswell Host", "Haswell Docker "])
save_to_json(rtlStressNgMerged, f"{specific_out}\\merged.json")
type = "THR"
gen_boxplot_from_hist(rtlStressNgMerged, type,
                      y_label="Latency (ns)",
                      save_path=f"{specific_out}\\merged_{type}_boxplot_stress-ng.pdf",
                      labels=["Broadwell Host", "Broadwell Docker", "Haswell Host", "Haswell Docker "],
                      figure_size=(6, 3), y_limit=(1000, 6000), align_size=False, legend=True)
type = "USR"
gen_boxplot_from_hist(rtlStressNgMerged, type,
                      y_label="Latency (ns)",
                      save_path=f"{specific_out}\\merged_{type}_boxplot_stress-ng.pdf",
                      labels=["Broadwell Host", "Broadwell Docker", "Haswell Host", "Haswell Docker "],
                      figure_size=(6, 4), y_limit=(1000, 9000), align_size=False, legend=True)

gen_boxplot_from_hist(rtlStressNgMerged, "IRQ",
                      y_label="Latency (ns)",
                      save_path=f"{specific_out}\\merged_IRQ_boxplot_stress-ng.pdf",
                      labels=["Broadwell Host", "Broadwell Docker", "Haswell Host", "Haswell Docker "],
                      figure_size=(6, 2), y_limit=(100, 1200), align_size=False, legend=True)

gen_tex_table(rtlStressNgMerged, ["avg", "max", "over", "StdDev", "P99", "HistSpan"], "USR",
              f"{specific_out}\\stats.tex")
gen_tex_table(rtlStressNgMerged, ["count", "over", "max", "P99", "StdDev"], "USR",
              f"{specific_out}\\statsNew.tex")


plot_change(rtlStressNgMerged, save_path=f"{specific_out}\\median_latency_stress-ng.pdf")
plot_change(rtlStressNgMerged, save_path=f"{specific_out}\\average_latency_stress-ng.pdf", statistical_parameter="avg")

# # All RTLA
specific_out = output + f"\\rtlaAll"
os.makedirs(specific_out, exist_ok=True)
rtlaAll = merge_data(aggregate_data(broadwellRtlaStressHost), aggregate_data(broadwellRtlaStressDocker),
                     aggregate_data(haswellRtlaStressHost), aggregate_data(haswelllRtlaStressDocker),
                     aggregate_data(broadwellRtlaStressNgHost), aggregate_data(broadwellRtlaStressNgDocker),
                     aggregate_data(haswellRtlaStressNgHost), aggregate_data(haswellRtlaStressNgDocker),
                     names=["Broadwell Host Stress", "Broadwell Docker Stress", "Haswell Host Stress",
                            "Haswell Docker Stress", "Broadwell Host Stress-ng", "Broadwell Docker Stress-ng",
                            "Haswell Host Stress-ng",
                            "Haswell Docker Stress-ng"])

plot_change(rtlaAll, save_path=f"{specific_out}\\median_latency_all.pdf")

########################################################################################################################
# Cyclictest
########################################################################################################################
# Stress
specific_out = output + f"\\cyclictestStress"
os.makedirs(specific_out, exist_ok=True)
for data in [broadwellCyclicStressHost, broadwellCyclicStressDocker, haswellCyclicStressHost,
             haswellCyclicStressDocker]:
    name = get_var_name(data)
    save_to_json(data, f"{specific_out}\\{name}.json")
    for type in ["USR"]:
        gen_boxplot_from_hist(data, type, title=f"{name} {type} Boxplot", x_label="Threads",
                              y_label="Latency (ns)",
                              save_path=f"{specific_out}\\{name}_{type}_boxplot_stress.pdf", legend=False)



cylicStressMerged = merge_data(broadwellCyclicStressHostAggregated, broadwellCyclicStressDockerAggregated,
                               haswellCyclicStressHostAggregated, haswellCyclicStressDockerAggregated,
                               names=["Broadwell Host", "Broadwell Docker", "Haswell Host", "Haswell Docker "])
save_to_json(cylicStressMerged, f"{specific_out}\\merged.json")
for type in ["USR"]:
    gen_boxplot_from_hist(cylicStressMerged, type,
                          y_label="Latency (ns)",
                          save_path=f"{specific_out}\\merged_{type}_boxplot_stress.pdf",
                          labels=["Broadwell Host", "Broadwell Docker", "Haswell Host", "Haswell Docker "],
                          figure_size=(6, 5), y_limit=(1000, 8000), align_size=False, legend=False)

gen_tex_table(cylicStressMerged, ["avg", "max", "over", "StdDev", "P99", "HistSpan"], "USR",
              f"{specific_out}\\stats.tex")
gen_tex_table(cylicStressMerged, ["count", "over", "max", "P99", "StdDev"], "USR",
              f"{specific_out}\\statsNew.tex")

#
# # # Verbose output parsing - slow only uncomment if needed
# haswellHostVerbose = parse_cyclictest_verbose_data(f"{haswellCyclictestSubFolder}\\HostStress_run1\\output")
# broadwellHostVerbose = parse_cyclictest_verbose_data(f"{broadwellCyclictestSubFolder}\\HostStress_run1\\output")
# gen_timeseries(broadwellHostVerbose, sm="none", save_path=f"{specific_out}\\timeseries_Unfiltered_broadwellHost-stress",
#                ylim=(2000, 19000))
# gen_timeseries(broadwellHostVerbose, sm="max", save_path=f"{specific_out}\\timeseries_Max_broadwellHost-stress",
#                ylim=(2000, 19000))
# gen_timeseries(broadwellHostVerbose, sm="mean", save_path=f"{specific_out}\\timeseries_Mean_broadwellHost-stress",
#                ylim=(2000, 8000))
# gen_timeseries(broadwellHostVerbose, sm="mean", save_path=f"{specific_out}\\timeseries_Mean_broadwellHost-stress",
#                ylim=(2000, 8000))
# gen_timeseries(broadwellHostVerbose, sm="min", save_path=f"{specific_out}\\timeseries_Min_broadwellHost-stress",
#                ylim=(2000, 8000))
# gen_timeseries(haswellHostVerbose, sm="none", save_path=f"{specific_out}\\timeseries_Unfiltered_haswellHost-stress",
#                ylim=(2000, 19000))
# gen_timeseries(haswellHostVerbose, sm="max", save_path=f"{specific_out}\\timeseries_Max_haswellHost-stress",
#                ylim=(2000, 19000))
# gen_timeseries(haswellHostVerbose, sm="mean", save_path=f"{specific_out}\\timeseries_Mean_haswellHost-stress",
#                ylim=(2000, 8000))
# gen_timeseries(haswellHostVerbose, sm="min", save_path=f"{specific_out}\\timeseries_Min_haswellHost-stress",
#                ylim=(2000, 8000))
#
#
# broadwellDockerVerbose = parse_cyclictest_verbose_data(f"{broadwellCyclictestSubFolder}\\DockerStress_run1\\output")
# haswellDockerVerbose = parse_cyclictest_verbose_data(f"{haswellCyclictestSubFolder}\\DockerStress_run1\\output")
# gen_timeseries(broadwellDockerVerbose, sm="none", save_path=f"{specific_out}\\timeseries_Unfiltered_broadwellDocker-stress",
#                ylim=(2000, 19000))
# gen_timeseries(broadwellDockerVerbose, sm="max", save_path=f"{specific_out}\\timeseries_Max_broadwellDocker-stress",
#                ylim=(2000, 19000))
# gen_timeseries(broadwellDockerVerbose, sm="mean", save_path=f"{specific_out}\\timeseries_Mean_broadwellDocker-stress",
#                ylim=(2000, 8000))
# gen_timeseries(broadwellDockerVerbose, sm="min", save_path=f"{specific_out}\\timeseries_Min_broadwellDocker-stress",
#                ylim=(2000, 8000))
# gen_timeseries(haswellDockerVerbose, sm="none", save_path=f"{specific_out}\\timeseries_Unfiltered_haswellDocker-stress",
#                ylim=(2000, 19000))
# gen_timeseries(haswellDockerVerbose, sm="max", save_path=f"{specific_out}\\timeseries_Max_haswellDocker-stress",
#                ylim=(2000, 19000))
# gen_timeseries(haswellDockerVerbose, sm="mean", save_path=f"{specific_out}\\timeseries_Mean_haswellDocker-stress",
#                ylim=(2000, 8000))
# gen_timeseries(haswellDockerVerbose, sm="min", save_path=f"{specific_out}\\timeseries_Min_haswellDocker-stress",
#                ylim=(2000, 8000))

# Strss-ng ############################################################################################################
specific_out = output + f"\\cyclictestStressNg"
os.makedirs(specific_out, exist_ok=True)
for data in [broadwellCyclicStressNgHost, broadwellCyclicStressNgDocker, haswellCyclicStressNgHost,
             haswellCyclicStressNgDocker]:
    name = get_var_name(data)
    save_to_json(data, f"{specific_out}\\{name}.json")
    for type in ["USR"]:
        gen_boxplot_from_hist(data, type, title=f"{name} {type} Boxplot", x_label="Threads",
                              y_label="Latency (ns)",
                              save_path=f"{specific_out}\\{name}_{type}_boxplot_stress-ng.pdf", figure_size=(6, 6),
                              legend=False)



cyclicStressNgMerged = merge_data(broadwellCyclicStressNgHostAggregated, broadwellCyclicStressNgDockerAggregated,
                                  haswellCyclicStressNgHostAggregated, haswellCyclicStressNgDockerAggregated,
                                  names=["Broadwell Host", "Broadwell Docker", "Haswell Host", "Haswell Docker "])
save_to_json(cyclicStressNgMerged, f"{specific_out}\\merged.json")
type = "USR"
gen_boxplot_from_hist(cyclicStressNgMerged, type, x_label=None,
                      y_label="Latency (ns)",
                      save_path=f"{specific_out}\\merged_{type}_boxplot_stress-ng.pdf",
                      labels=["Broadwell Host", "Broadwell Docker", "Haswell Host", "Haswell Docker "],
                      figure_size=(6, 5), y_limit=(1000, 8000), align_size=False, legend=True)

gen_tex_table(cyclicStressNgMerged, ["avg", "max", "over", "StdDev", "P99", "HistSpan"], "USR",
              f"{specific_out}\\stats.tex")

gen_tex_table(cyclicStressNgMerged, ["count", "over", "max", "P99", "StdDev"], "USR",
              f"{specific_out}\\statsNew.tex")

# # # Verbose output parsing - slow only uncomment if needed
# broadwellHostVerbose = parse_cyclictest_verbose_data(f"{broadwellCyclictestSubFolder}\\HostStress-ng_run1\\output")
# haswellHostVerbose = parse_cyclictest_verbose_data(f"{haswellCyclictestSubFolder}\\HostStress-ng_run1\\output")
# gen_timeseries(broadwellHostVerbose, sm="none",
#                save_path=f"{specific_out}\\timeseries_Unfiltered_broadwellHost-stress-ng",
#                ylim=(1000, 110000))
# gen_timeseries(broadwellHostVerbose, sm="max", save_path=f"{specific_out}\\timeseries_Max_broadwellHost-stress-ng",
#                ylim=(1000, 110000))
# gen_timeseries(broadwellHostVerbose, sm="mean", save_path=f"{specific_out}\\timeseries_Mean_broadwellHost-stress-ng",
#                ylim=(1000, 8000))
# gen_timeseries(broadwellHostVerbose, sm="min", save_path=f"{specific_out}\\timeseries_min_broadwellHost-stress-ng",
#                ylim=(1000, 8000))
# gen_timeseries(haswellHostVerbose, sm="none", save_path=f"{specific_out}\\timeseries_Unfiltered_haswellHost-stress-ng",
#                ylim=(1000, 110000))
# gen_timeseries(haswellHostVerbose, sm="max", save_path=f"{specific_out}\\timeseries_Max_haswellHost-stress-ng",
#                ylim=(1000, 110000))
# gen_timeseries(haswellHostVerbose, sm="mean", save_path=f"{specific_out}\\timeseries_Mean_haswellHost-stress-ng",
#                ylim=(1000, 8000))
# gen_timeseries(haswellHostVerbose, sm="min", save_path=f"{specific_out}\\timeseries_min_haswellHost-stress-ng",
#                ylim=(1000, 8000))
#
# broadwellDockerVerbose = parse_cyclictest_verbose_data(f"{broadwellCyclictestSubFolder}\\DockerStress-ng_run1\\output")
# haswellDockerVerbose = parse_cyclictest_verbose_data(f"{haswellCyclictestSubFolder}\\DockerStress-ng_run1\\output")
# gen_timeseries(broadwellDockerVerbose, sm="none",
#                save_path=f"{specific_out}\\timeseries_Unfiltered_broadwellDocker-stress-ng",
#                ylim=(1000, 110000))
# gen_timeseries(broadwellDockerVerbose, sm="max", save_path=f"{specific_out}\\timeseries_Max_broadwellDocker-stress-ng",
#                ylim=(1000, 110000))
# gen_timeseries(broadwellDockerVerbose, sm="mean",
#                save_path=f"{specific_out}\\timeseries_Mean_broadwellDocker-stress-ng",
#                ylim=(1000, 8000))
# gen_timeseries(broadwellDockerVerbose, sm="min", save_path=f"{specific_out}\\timeseries_min_broadwellDocker-stress-ng",
#                ylim=(1000, 8000))
# gen_timeseries(haswellHostVerbose, sm="none", save_path=f"{specific_out}\\timeseries_Unfiltered_haswellHost-stress-ng",
#                ylim=(1000, 110000))
# gen_timeseries(haswellHostVerbose, sm="max", save_path=f"{specific_out}\\timeseries_Max_haswellHost-stress-ng",
#                ylim=(1000, 110000))
# gen_timeseries(haswellHostVerbose, sm="mean", save_path=f"{specific_out}\\timeseries_Mean_haswellHost-stress-ng",
#                ylim=(1000, 8000))
# gen_timeseries(haswellHostVerbose, sm="min", save_path=f"{specific_out}\\timeseries_min_haswellHost-stress-ng",
#                ylim=(1000, 8000))
########################################################################################################################
# All
########################################################################################################################


all_aggregated = merge_data(haswellCyclicStressHostAggregated, haswellRtlaStressHostAggregated,
                            broadwellCyclicStressHostAggregated, broadwellRtlaStressHostAggregated,
                            haswellCyclicStressDockerAggregated, haswellRtlaStressDockerAggregated,
                            broadwellCyclicStressDockerAggregated, broadwellRtlaStressDockerAggregated,
                            haswellCyclicStressNgHostAggregated, haswellRtlaStressNgHostAggregated,
                            broadwellCyclicStressNgHostAggregated, broadwellRtlaStressNgHostAggregated,
                            haswellCyclicStressNgDockerAggregated, haswellRtlaStressNgDockerAggregated,
                            broadwellCyclicStressNgDockerAggregated, broadwellRtlaStressNgDockerAggregated,
                            names=["Haswell Cyclictest Stress Host", "Haswell RTLA Stress Host",
                                   "Broadwell Cyclictest Stress Host", "Broadwell RTLA Stress Host",
                                   "Haswell Cyclictest Stress Docker", "Haswell RTLA Stress Docker",
                                   "Broadwell Cyclictest Stress Docker", "Broadwell RTLA Stress Docker",
                                   "Haswell Cyclictest Stress-ng Host", "Haswell RTLA Stress-ng Host",
                                   "Broadwell Cyclictest Stress-ng Host", "Broadwell RTLA Stress-ng Host",
                                   "Haswell Cyclictest Stress-ng Docker", "Haswell RTLA Stress-ng Docker",
                                   "Broadwell Cyclictest Stress-ng Docker", "Broadwell RTLA Stress-ng Docker"])

all_aggregated_sorted = merge_data(haswellCyclicStressHostAggregated, haswellCyclicStressDockerAggregated,
                                   broadwellCyclicStressHostAggregated, broadwellCyclicStressDockerAggregated,
                                    haswellRtlaStressHostAggregated, haswellRtlaStressDockerAggregated,
                                   broadwellRtlaStressHostAggregated, broadwellRtlaStressDockerAggregated,
                                    haswellCyclicStressNgHostAggregated, haswellCyclicStressNgDockerAggregated,
                                   broadwellCyclicStressNgHostAggregated, broadwellCyclicStressNgDockerAggregated,
                                    haswellRtlaStressNgHostAggregated, haswellRtlaStressNgDockerAggregated,
                                   broadwellRtlaStressNgHostAggregated, broadwellRtlaStressNgDockerAggregated,
                                   names=["Haswell Cyclictest Stress Host", "Haswell Cyclictest Stress Docker",
                                            "Broadwell Cyclictest Stress Host", "Broadwell Cyclictest Stress Docker",
                                            "Haswell RTLA Stress Host", "Haswell RTLA Stress Docker",
                                            "Broadwell RTLA Stress Host", "Broadwell RTLA Stress Docker",
                                            "Haswell Cyclictest Stress-ng Host", "Haswell Cyclictest Stress-ng Docker",
                                            "Broadwell Cyclictest Stress-ng Host", "Broadwell Cyclictest Stress-ng Docker",
                                            "Haswell RTLA Stress-ng Host", "Haswell RTLA Stress-ng Docker",
                                            "Broadwell RTLA Stress-ng Host", "Broadwell RTLA Stress-ng Docker"])


gen_boxplot_from_hist(all_aggregated_sorted, "USR", y_label="Latency (ns)", labels=["HCSH", "HCSD", "BCSH",
                                                                                    "BCSD","HRSH", "HRSD", "BRSH",
                                                                                    "BRSD",
                                                                                    "HCNH", "HCND", "BCNH", "BCND",
                                                                                    "HRNH", "HRND", "BRNH", "BRND"],
                      save_path=f"{output}\\all_USR_boxplot.pdf", align_size=False, legend=False, figure_size=(12, 6))


gen_tex_table(all_aggregated, ["count", "Fliers", "over"], "USR", f"{output}\\stats.tex")

gen_barplot(all_aggregated, "count", "USR", x_axis_label="Experiment", y_axis_label="Total Sample Count",
            save_path=f"{output}\\barplot_samplecount_all", shorten_labels=True, figsize=(10, 4))
gen_barplot(rtlStressNgMerged, "count", "USR", x_axis_label="Experiment", y_axis_label="Count",
            save_path=f"{output}\\barplot")
gen_barplot(haswellRtlaStressNgDocker, "count", "USR", x_axis_label="Thread", y_axis_label="Count",
            save_path=f"{output}\\barplot_haswellRtlaStressNgDocker")
gen_barplot(haswellRtlaStressNgHost, "count", "USR", x_axis_label="Thread", y_axis_label="Count",
            save_path=f"{output}\\barplot_haswellRtlaStressNgHost")
gen_barplot(broadwellRtlaStressNgDocker, "count", "USR", x_axis_label="Thread", y_axis_label="Count",
            save_path=f"{output}\\barplot_broadwellRtlaStressNgDocker")
gen_barplot(broadwellRtlaStressNgHost, "count", "USR", x_axis_label="Thread", y_axis_label="Count",
            save_path=f"{output}\\barplot_broadwellRtlaStressNgHost")

gen_barplot(all_aggregated, "FlierPercentage", "USR", x_axis_label="Experiment", y_axis_label="Outlier Percentage %",
            save_path=f"{output}\\barplot_flierP_all", shorten_labels=True, figsize=(10, 4))

# # Generate histograms for all data. Slow only uncomment if needed
histogram_folder = f"{output}\\histograms"
os.makedirs(histogram_folder, exist_ok=True)
for data in [broadwellRtlaStressHost, broadwellRtlaStressDocker, haswellRtlaStressHost, haswelllRtlaStressDocker,
             broadwellRtlaStressNgHost, broadwellRtlaStressNgDocker, haswellRtlaStressNgHost, haswellRtlaStressNgDocker,
             broadwellCyclicStressHost, broadwellCyclicStressDocker, haswellCyclicStressHost, haswellCyclicStressDocker,
             broadwellCyclicStressNgHost, broadwellCyclicStressNgDocker, haswellCyclicStressNgHost,
             haswellCyclicStressNgDocker]:
    name = get_var_name(data)

    gen_histogram(data, "USR", save_path=f"{histogram_folder}\\{name}_histogram_USR", limit_x=15000)
    gen_histogram(data, "USR", save_path=f"{histogram_folder}\\{name}_histogram_USR_no_lim")


# Boxplots with flier ##################################################################################################
boxplot_flier_folder = f"{output}\\boxplot_fliers"
os.makedirs(boxplot_flier_folder, exist_ok=True)

for test in [broadwellRtlaStressHost, broadwellRtlaStressDocker, haswellRtlaStressHost, haswelllRtlaStressDocker,
             haswellRtlaStressNgHost, haswellRtlaStressNgDocker]:
    for type in ["USR", "THR", "IRQ"]:
        name = get_var_name(test)
        gen_boxplot_from_hist(test, type, x_label="Threads", y_label="Latency (ns)", labels=[0, 1, 2, 3],
                              save_path=f"{boxplot_flier_folder}\\{name}_{type}_boxplot_flier", show_fliers=True,
                              align_size=False, legend=False)

# Significant difference ################################################################################################
significant_diff_folder = f"{output}\\significant_diff"
os.makedirs(significant_diff_folder, exist_ok=True)
names = ["Haswell Cyclictest Stress Host", "Haswell RTLA Stress Host",
         "Broadwell Cyclictest Stress Host", "Broadwell RTLA Stress Host",
         "Haswell Cyclictest Stress Docker", "Haswell RTLA Stress Docker",
         "Broadwell Cyclictest Stress Docker", "Broadwell RTLA Stress Docker",
         "Haswell Cyclictest Stress-ng Host", "Haswell RTLA Stress-ng Host",
         "Broadwell Cyclictest Stress-ng Host", "Broadwell RTLA Stress-ng Host",
         "Haswell Cyclictest Stress-ng Docker", "Haswell RTLA Stress-ng Docker",
         "Broadwell Cyclictest Stress-ng Docker", "Broadwell RTLA Stress-ng Docker"]

results = []
for host in names:
    if 'Host' in host:
        docker = host.replace('Host', 'Docker')
        if docker in names:
            save_path = f"{significant_diff_folder}\\{host.lower().replace(' ', '_')}"
            result = Calc_Confidence_Effect(all_aggregated, host, docker, irq_metric='USR', save_path=save_path)
            results.append((host, result))

latex_table = r"""
\begin{table}[h!]
\centering
\resizebox{\textwidth}{!}{\begin{tabular}{|c|c|c|c|c|c|c|c|c|}
\hline
Test set & t test (T;P) & Mann-Whitney (U;P) & Levene's test (T;P) & Cohen's d & Cohen's d Interval & Mean Diff & Mean Increase (\%) & Median Diff & Median Increase (\%) \\
\hline
"""

for host, result in results:
    t_test_t, t_test_p = result['t_test']
    mann_whitney_u, mann_whitney_p = result['mann_whitney_u']
    levene_t, levene_p = result['levene_test']
    cohen_d = result['effect_size']
    cohen_d_interval = result['effect_size_ci']
    mean_diff = result['difference_mean']
    mean_increase = result['increase_mean']
    median_diff = result['difference_median']
    median_increase = result['increase_median']

    cohen_d_interval_str = f"({cohen_d_interval[0]:.2f}, {cohen_d_interval[1]:.2f})"
    latex_table += f"{host.replace('Host', '')} & ({t_test_t:.2f}; {t_test_p:.2f}) & ({mann_whitney_u:.2f}; {mann_whitney_p:.2f}) & ({levene_t:.2f}; {levene_p:.2f}) & {cohen_d:.2f} & {cohen_d_interval_str} & {mean_diff:.2f} & {mean_increase:.2f} & {median_diff:.2f} & {median_increase:.2f} \\\\\n"
    latex_table += r"\hline" + "\n"

latex_table += r"""
\end{tabular}}
\caption{Results of Performance Tests}
\label{table:results}
\end{table}
"""

latex_file_path = os.path.join(significant_diff_folder, "performance_tests_results.tex")

with open(latex_file_path, "w") as latex_file:
    latex_file.write(latex_table)

print(results)
print("Done")
