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


def parse_stress_ng_log(file_path):
    pattern_stressor = re.compile(r'^stress-ng:\s+info:\s+\[\d+\]\s+(passed|skipped|failed):\s*(\d+)\s*(|:\s(.*))$')
    pattern_time = re.compile(r'^stress-ng:\s+info:\s+\[\d+\]\s+(successful|unsuccessful)\s+run\s+completed\s+in\s+(\d+)\s+hour,\s+?(\d+(?:\.\d+)?)\s+secs$')
    results = {
        "passed": {},
        "passed_count": 0,
        "passed_sum": 0,
        "skipped": {},
        "skipped_count": 0,
        "skipped_sum": 0,
        "failed": {},
        "failed_count": 0,
        "failed_sum": 0,
        "time": 0,
        "status": "unknown"
    }

    with open(file_path, 'r') as file:
        for line in file:
            if match := pattern_stressor.match(line):
                status, count, stressors, stressors_list = match.groups()
                if stressors_list:
                    stressors_dict = dict(re.findall(r"([\w-]+)\s+\((\d+)\)", stressors_list))
                    stressors_dict = {stressor: int(count) for stressor, count in stressors_dict.items()}
                    results[status] = stressors_dict
                    results[f"{status}_count"] = len(stressors_dict)
                    results[f"{status}_sum"] = sum(stressors_dict.values())
                else:
                    results[status] = {}
                    results[f"{status}_count"] = 0
                    results[f"{status}_sum"] = int(count)
            elif match := pattern_time.match(line):
                status, hours, seconds = match.groups()
                total_seconds = int(hours) * 3600 + float(seconds)
                results["status"] = status
                results["time"] = total_seconds

    return results

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
                figure_size=(6, 6), palette="tab10", save_path=None, ylim=None, show_fliers=True):
    """
    Generate a boxplot from the given data with Seaborn color palettes and custom labels.
    @param boxplot_data: The data to generate the boxplot from. Is an array of list with ['label', 'mean', 'iqr',
     'cilo', 'cihi', 'whishi', 'whislo', 'fliers', 'q1', 'med', 'q3']
    @param title: Title of the boxplot.
    @param x_label: Label for the x-axis.
    @param y_label: Label for the y-axis.
    @param figure_size: Size of the figure.
    @param palette: Name of the Seaborn color palette to use.
    @param save_path: Path to save the plot.
    @param ylim: Limit for the y-axis.
    @return: None
    """
    fig, axs = plt.subplots(figsize=figure_size)
    meanprops = {'marker': '^', 'markerfacecolor': 'black', 'markeredgecolor': 'black'}
    flierprops = dict(marker='o', markerfacecolor='b', markersize=2)
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
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
        save_path= save_path.replace("pdf", "png") # Cheap hack to save both formats
        plt.savefig(save_path, dpi=900)
        print(f"Saved boxplot to {save_path}")
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
                           y_limit=None, show_fliers=True):
    """
    Generate a boxplot from the given histogram data. Wrapper function for get_boxplot_data and gen_boxplot.
    @param show_fliers: Show the fliers in the boxplot.
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
    gen_boxplot(bxp, title=title, x_label=x_label, y_label=y_label, save_path=save_path, figure_size=figure_size,
                palette=palette, ylim=y_limit, show_fliers=show_fliers)


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
        plt.savefig(f"{save_path}.png")
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
        plt.savefig(f'{save_path}.png')
        print(f"Saved timeseries plot to {save_path}.pdf and {save_path}.png")
    else:
        plt.show()
    plt.close()


def plot_change(data, save_path=None, figsize=(12, 8), statistical_parameter="Q2"):
    """
    Plot the latency for each metric and annotate absolute changes.
    @param statistical_parameter: The statistical parameter to plot. Default is the median.
    @param save_path: The path to save the plot to.
    @param data: The dataset containing metrics for each experiment.
    """
    colors = sns.color_palette("tab10")
    cc = itertools.cycle(colors)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    change_legend_labels = []
    plot_lines = []
    metrics_dict = {'Stress': {}, 'Stress-ng': {}}

    for idx, (experiment, metrics) in enumerate(data.items()):
        x = ['IRQ', 'THR', 'USR']
        y = [metrics['IRQ'][statistical_parameter], metrics['THR'][statistical_parameter],
             metrics['USR'][statistical_parameter]]

        parts = experiment.split()
        platform = parts[1] if len(parts) > 1 else "Unknown"
        stress_type = "Stress-ng" if "Stress-ng" in experiment else "Stress"
        variant = "Docker" if "Docker" in experiment else "Podman" if "Podman" in experiment else "LXC" if "LXC" in experiment else "Unknown"

        key = f"{platform} {stress_type}"

        marker = 'o'
        if variant == "Docker":
            marker = '>'
        elif variant == "Podman":
            marker = '^'
        elif variant == "LXC":
            marker = 's'

        if key not in metrics_dict[stress_type]:
            metrics_dict[stress_type][key] = {}
        metrics_dict[stress_type][key][variant] = y

        c = next(cc)
        line, = ax.plot(x, y, marker=marker, linestyle='-', color=c, label=experiment)
        plot_lines.append(line)

    ax.set_ylabel('Median Latency (ns)', fontsize=12)
    ax.grid(True)

    print(change_legend_labels)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def gen_plot_stress_test_results(data_list, names=None, show_total=False, save_path=None, show_labels=True):
    """
    Generate a bar plot of the stress-ng log file.
    @param data_list: The list of data sets to plot.
    @param names: The names of the data sets.
    @param show_total: Show the total count of tests.
    @param save_path: The path to save the plot to.
    @param show_labels: Show the labels on the bars.
    @return: None
    """
    if names is None:
        names = [f"Dataset {i + 1}" for i in range(len(data_list))]

    total_counts = []
    passed_counts = []
    skipped_counts = []
    failed_counts = []

    for data in data_list:
        passed_count = data.get("passed_count", 0)
        skipped_count = data.get("skipped_count", 0)
        failed_count = data.get("failed_count", 0)

        total_count = passed_count + skipped_count + failed_count

        total_counts.append(total_count)
        passed_counts.append(passed_count)
        skipped_counts.append(skipped_count)
        failed_counts.append(failed_count)

    x = np.arange(len(data_list))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    if show_total:
        rects1 = ax.bar(x - 1.5 * width, total_counts, width, label='Total Count')
    rects2 = ax.bar(x - 0.5 * width, passed_counts, width, label='Passed Count')
    rects3 = ax.bar(x + 0.5 * width, skipped_counts, width, label='Skipped Count')
    rects4 = ax.bar(x + 1.5 * width, failed_counts, width, label='Failed Count')

    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', rotation=90, fontsize=6)

    if show_labels:
        if show_total:
            add_labels(rects1)
        add_labels(rects2)
        add_labels(rects3)
        add_labels(rects4)

    ax.set_xlabel('Datasets')
    ax.set_ylabel('Counts')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90)
    ax.legend()

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot of stress-ng to {save_path}.pdf")
    else:
        plt.show()


def compare_and_generate_latex_table(datasets, names, save_path):
    unique_stressors = set()
    for data in datasets:
        for category in ['passed', 'skipped', 'failed']:
            unique_stressors.update(data[category].keys())

    comparison_data = {stressor: [] for stressor in unique_stressors}
    for idx, data in enumerate(datasets):
        for stressor in unique_stressors:
            if stressor in data['passed']:
                comparison_data[stressor].append("✔")
            elif stressor in data['failed']:
                comparison_data[stressor].append("✘")
            elif stressor in data['skipped']:
                comparison_data[stressor].append("-")
            else:
                comparison_data[stressor].append(" ")

    stressors_with_changes = {
        stressor: comparison for stressor, comparison in comparison_data.items()
        if any(comparison[0] != c for c in comparison[1:])
    }

    df = pd.DataFrame(stressors_with_changes)
    df.insert(0, "Experiment", names)

    df = df.set_index('Experiment').transpose().reset_index()
    df.columns = ['Stressor'] + names

    latex_table = "\\begin{table}[h!]\n\\centering\n\\begin{longtable}{|l|" + "c|" * (
                len(df.columns) - 1) + "}\n\\hline\n"

    header = " & ".join(df.columns) + " \\\\ \\hline\n"
    latex_table += header

    for _, row in df.iterrows():
        row_data = " & ".join(row.astype(str)) + " \\\\ \\hline\n"
        latex_table += row_data

    latex_table += "\\end{longtable}\n\\caption{Comparison of Stressors Across Datasets}\n\\label{table:comparison}\n\\end{table}"

    with open(save_path, 'w', encoding='utf-8') as file:
        file.write(latex_table)

########################################################################################################################
# User defined variables
# Set the base path to the directory containing the output files
# Load the RTLA and Cyclictest data
########################################################################################################################

basepath = "H:\\MA\\Final\\isol"
output = f"{basepath}\\output"
os.makedirs(output, exist_ok=True)
log_output = f"{basepath}\\logs"
########################################################################################################################
# Broadwell Data
########################################################################################################################
# RTLA Data ############################################################################################################
rtlaSubFolder = f"{basepath}\\isol_f2_broadwell_rtla_662328realtime11rtlts"

# Docker Stress-ng Host
rtlaDockerStressNgHostLog = parse_stress_ng_log(f"{rtlaSubFolder}\\DockerStress-ngHost_run1\\stress-ng.log")
save_to_json(rtlaDockerStressNgHostLog, f"{log_output}\\rtla_DockerStressNgHost.json")
rtlaDockerStressNgHost = parse_rtla_data(f"{rtlaSubFolder}\\DockerStress-ngHost_run1\\rtla_output")

# Docker Stress-ng Extreme Docker
rtlaDockerStressNgExtremeDockerLog = parse_stress_ng_log(f"{rtlaSubFolder}\\DockerStress-ngExtremeDocker_run1\\stress-ng.log")
save_to_json(rtlaDockerStressNgExtremeDockerLog, f"{log_output}\\rtla_DockerStressNgExtremeDocker.json")
rtlaDockerStressNgExtremeDocker = parse_rtla_data(f"{rtlaSubFolder}\\DockerStress-ngExtremeDocker_run1\\rtla_output")

# Docker Stress-ng Limited Docker
rtlaDockerStressNgLimitedDockerLog = parse_stress_ng_log(f"{rtlaSubFolder}\\DockerStress-ngLimitedDocker_run1\\stress-ng.log")
save_to_json(rtlaDockerStressNgLimitedDockerLog, f"{log_output}\\rtla_DockerStressNgLimitedDocker.json")
rtlaDockerStressNgLimitedDocker = parse_rtla_data(f"{rtlaSubFolder}\\DockerStress-ngLimitedDocker_run1\\rtla_output")

# Docker Stress-ng Privileged Docker
rtlaDockerStressNgPrivilegedDockerLog = parse_stress_ng_log(f"{rtlaSubFolder}\\DockerStress-ngPrivilegedDocker_run1\\stress-ng.log")
save_to_json(rtlaDockerStressNgPrivilegedDockerLog, f"{log_output}\\rtla_DockerStressNgPrivilegedDocker.json")
rtlaDockerStressNgPrivilegedDocker = parse_rtla_data(f"{rtlaSubFolder}\\DockerStress-ngPrivilegedDocker_run1\\rtla_output")

# Docker Stress-ng Unprivileged Docker
rtlaDockerStressNgUnprivilegedDockerLog = parse_stress_ng_log(f"{rtlaSubFolder}\\DockerStress-ngUnprivilegedDocker_run1\\stress-ng.log")
save_to_json(rtlaDockerStressNgUnprivilegedDockerLog, f"{log_output}\\rtla_DockerStressNgUnprivilegedDocker.json")
rtlaDockerStressNgUnprivilegedDocker = parse_rtla_data(f"{rtlaSubFolder}\\DockerStress-ngUnprivilegedDocker_run1\\rtla_output")

########################################################################################################################

# Host Stress-ng Extreme Docker
rtlaHostStressNgExtremeDockerLog = parse_stress_ng_log(f"{rtlaSubFolder}\\HostStress-ngExtremeDocker_run1\\stress-ng.log")
save_to_json(rtlaHostStressNgExtremeDockerLog, f"{log_output}\\rtla_HostStressNgExtremeDocker.json")
rtlaHostStressNgExtremeDocker = parse_rtla_data(f"{rtlaSubFolder}\\HostStress-ngExtremeDocker_run1\\rtla_output")

# Host Stress-ng Host
rtlaHostStressNgHostLog = parse_stress_ng_log(f"{rtlaSubFolder}\\HostStress-ngHost_run1\\stress-ng.log")
save_to_json(rtlaHostStressNgHostLog, f"{log_output}\\rtla_HostStressNgHost.json")
rtlaHostStressNgHost = parse_rtla_data(f"{rtlaSubFolder}\\HostStress-ngHost_run1\\rtla_output")

# Host Stress-ng Limited Docker
rtlaHostStressNgLimitedDockerLog = parse_stress_ng_log(f"{rtlaSubFolder}\\HostStress-ngLimitedDocker_run1\\stress-ng.log")
save_to_json(rtlaHostStressNgLimitedDockerLog, f"{log_output}\\rtla_HostStressNgLimitedDocker.json")
rtlaHostStressNgLimitedDocker = parse_rtla_data(f"{rtlaSubFolder}\\HostStress-ngLimitedDocker_run1\\rtla_output")

# Host Stress-ng Privileged Docker
rtlaHostStressNgPrivilegedDockerLog = parse_stress_ng_log(f"{rtlaSubFolder}\\HostStress-ngPrivilegedDocker_run1\\stress-ng.log")
save_to_json(rtlaHostStressNgPrivilegedDockerLog, f"{log_output}\\rtla_HostStressNgPrivilegedDocker.json")
rtlaHostStressNgPrivilegedDocker = parse_rtla_data(f"{rtlaSubFolder}\\HostStress-ngPrivilegedDocker_run1\\rtla_output")

# Host Stress-ng Unprivileged Docker
rtlaHostStressNgUnprivilegedDockerLog = parse_stress_ng_log(f"{rtlaSubFolder}\\HostStress-ngUnprivilegedDocker_run1\\stress-ng.log")
save_to_json(rtlaHostStressNgUnprivilegedDockerLog, f"{log_output}\\rtla_HostStressNgUnprivilegedDocker.json")
rtlaHostStressNgUnprivilegedDocker = parse_rtla_data(f"{rtlaSubFolder}\\HostStress-ngUnprivilegedDocker_run1\\rtla_output")

########################################################################################################################

rtlaDockerStressNgHostAggregated = aggregate_data(rtlaDockerStressNgHost)
rtlaDockerStressNgPrivilegedDockerAggregated = aggregate_data(rtlaDockerStressNgPrivilegedDocker)
rtlaDockerStressNgUnprivilegedDockerAggregated = aggregate_data(rtlaDockerStressNgUnprivilegedDocker)
rtlaDockerStressNgLimitedDockerAggregated = aggregate_data(rtlaDockerStressNgLimitedDocker)
rtlaDockerStressNgExtremeDockerAggregated = aggregate_data(rtlaDockerStressNgExtremeDocker)

rtlaHostStressNgHostAggregated = aggregate_data(rtlaHostStressNgHost)
rtlaHostStressNgPrivilegedDockerAggregated = aggregate_data(rtlaHostStressNgPrivilegedDocker)
rtlaHostStressNgUnprivilegedDockerAggregated = aggregate_data(rtlaHostStressNgUnprivilegedDocker)
rtlaHostStressNgLimitedDockerAggregated = aggregate_data(rtlaHostStressNgLimitedDocker)
rtlaHostStressNgExtremeDockerAggregated = aggregate_data(rtlaHostStressNgExtremeDocker)

# Cyclictest Data ######################################################################################################
cyclictestSubFolder = f"{basepath}\\isol_f2_broadwell_cyclic_662328realtime11rtlts"

# Docker Stress-ng Host
cyclictestDockerStressNgHostLog = parse_stress_ng_log(f"{cyclictestSubFolder}\\DockerStress-ngHost_run1\\stress-ng.log")
save_to_json(cyclictestDockerStressNgHostLog, f"{log_output}\\cyclictest_HostStressNgHost.json")
cyclictestDockerStressNgHost = parse_cyclictest_json_data(f"{cyclictestSubFolder}\\DockerStress-ngHost_run1\\output.json")

# Docker Stress-ng Extreme Docker
cyclictestDockerStressNgExtremeDockerLog = parse_stress_ng_log(f"{cyclictestSubFolder}\\DockerStress-ngExtremeDocker_run1\\stress-ng.log")
save_to_json(cyclictestDockerStressNgExtremeDockerLog, f"{log_output}\\cyclictest_DockerStressNgExtremeDocker.json")
cyclictestDockerStressNgExtremeDocker = parse_cyclictest_json_data(f"{cyclictestSubFolder}\\DockerStress-ngExtremeDocker_run1\\output.json")

# Docker Stress-ng Limited Docker
cyclictestDockerStressNgLimitedDockerLog = parse_stress_ng_log(f"{cyclictestSubFolder}\\DockerStress-ngLimitedDocker_run1\\stress-ng.log")
save_to_json(cyclictestDockerStressNgLimitedDockerLog, f"{log_output}\\cyclictest_DockerStressNgLimitedDocker.json")
cyclictestDockerStressNgLimitedDocker = parse_cyclictest_json_data(f"{cyclictestSubFolder}\\DockerStress-ngLimitedDocker_run1\\output.json")

# Docker Stress-ng Privileged Docker
cyclictestDockerStressNgPrivilegedDockerLog = parse_stress_ng_log(f"{cyclictestSubFolder}\\DockerStress-ngPrivilegedDocker_run1\\stress-ng.log")
save_to_json(cyclictestDockerStressNgPrivilegedDockerLog, f"{log_output}\\cyclictest_DockerStressNgPrivilegedDocker.json")
cyclictestDockerStressNgPrivilegedDocker = parse_cyclictest_json_data(f"{cyclictestSubFolder}\\DockerStress-ngPrivilegedDocker_run1\\output.json")

# Docker Stress-ng Unprivileged Docker
cyclictestDockerStressNgUnprivilegedDockerLog = parse_stress_ng_log(f"{cyclictestSubFolder}\\DockerStress-ngUnprivilegedDocker_run1\\stress-ng.log")
save_to_json(cyclictestDockerStressNgUnprivilegedDockerLog, f"{log_output}\\cyclictest_DockerStressNgUnprivilegedDocker.json")
cyclictestDockerStressNgUnprivilegedDocker = parse_cyclictest_json_data(f"{cyclictestSubFolder}\\DockerStress-ngUnprivilegedDocker_run1\\output.json")


########################################################################################################################

# Host Stress-ng Extreme Docker
cyclictestHostStressNgExtremeDockerLog = parse_stress_ng_log(f"{cyclictestSubFolder}\\HostStress-ngExtremeDocker_run1\\stress-ng.log")
save_to_json(cyclictestHostStressNgExtremeDockerLog, f"{log_output}\\cyclictest_HostStressNgExtremeDocker.json")
cyclictestHostStressNgExtremeDocker = parse_cyclictest_json_data(f"{cyclictestSubFolder}\\HostStress-ngExtremeDocker_run1\\output.json")

# Host Stress-ng Host
cyclictestHostStressNgHostLog = parse_stress_ng_log(f"{cyclictestSubFolder}\\HostStress-ngHost_run1\\stress-ng.log")
save_to_json(cyclictestHostStressNgHostLog, f"{log_output}\\cyclictest_HostStressNgHost.json")
cyclictestHostStressNgHost = parse_cyclictest_json_data(f"{cyclictestSubFolder}\\HostStress-ngHost_run1\\output.json")

# Host Stress-ng Limited Docker
cyclictestHostStressNgLimitedDockerLog = parse_stress_ng_log(f"{cyclictestSubFolder}\\HostStress-ngLimitedDocker_run1\\stress-ng.log")
save_to_json(cyclictestHostStressNgLimitedDockerLog, f"{log_output}\\cyclictest_HostStressNgLimitedDocker.json")
cyclictestHostStressNgLimitedDocker = parse_cyclictest_json_data(f"{cyclictestSubFolder}\\HostStress-ngLimitedDocker_run1\\output.json")

# Host Stress-ng Privileged Docker
cyclictestHostStressNgPrivilegedDockerLog = parse_stress_ng_log(f"{cyclictestSubFolder}\\HostStress-ngPrivilegedDocker_run1\\stress-ng.log")
save_to_json(cyclictestHostStressNgPrivilegedDockerLog, f"{log_output}\\cyclictest_HostStressNgPrivilegedDocker.json")
cyclictestHostStressNgPrivilegedDocker = parse_cyclictest_json_data(f"{cyclictestSubFolder}\\HostStress-ngPrivilegedDocker_run1\\output.json")

# Host Stress-ng Unprivileged Docker
cyclictestHostStressNgUnprivilegedDockerLog = parse_stress_ng_log(f"{cyclictestSubFolder}\\HostStress-ngUnprivilegedDocker_run1\\stress-ng.log")
save_to_json(cyclictestHostStressNgUnprivilegedDockerLog, f"{log_output}\\cyclictest_HostStressNgUnprivilegedDocker.json")
cyclictestHostStressNgUnprivilegedDocker = parse_cyclictest_json_data(f"{cyclictestSubFolder}\\HostStress-ngUnprivilegedDocker_run1\\output.json")

########################################################################################################################
cyclictestHostStressNgHostAggregated = aggregate_data(cyclictestHostStressNgHost)
cyclictestHostStressNgPrivilegedDockerAggregated = aggregate_data(cyclictestHostStressNgPrivilegedDocker)
cyclictestHostStressNgUnprivilegedDockerAggregated = aggregate_data(cyclictestHostStressNgUnprivilegedDocker)
cyclictestHostStressNgLimitedDockerAggregated = aggregate_data(cyclictestHostStressNgLimitedDocker)
cyclictestHostStressNgExtremeDockerAggregated = aggregate_data(cyclictestHostStressNgExtremeDocker)

cyclictestDockerStressNgHostAggregated = aggregate_data(cyclictestDockerStressNgHost)
cyclictestDockerStressNgPrivilegedDockerAggregated = aggregate_data(cyclictestDockerStressNgPrivilegedDocker)
cyclictestDockerStressNgUnprivilegedDockerAggregated = aggregate_data(cyclictestDockerStressNgUnprivilegedDocker)
cyclictestDockerStressNgLimitedDockerAggregated = aggregate_data(cyclictestDockerStressNgLimitedDocker)
cyclictestDockerStressNgExtremeDockerAggregated = aggregate_data(cyclictestDockerStressNgExtremeDocker)

########################################################################################################################
# Merge Datasets
########################################################################################################################
# Stress Data (Cyclic and RTLA)
# Merged Cyclic and RTLA Data

cyclictestHost = merge_data(cyclictestHostStressNgHostAggregated, cyclictestHostStressNgPrivilegedDockerAggregated ,
                       cyclictestHostStressNgUnprivilegedDockerAggregated, cyclictestHostStressNgLimitedDockerAggregated,
                        cyclictestHostStressNgExtremeDockerAggregated,
                       names=["Host Cyclictest - Stress-ng Host" ,
                              "Host Cyclictest - Stress-ng Privileged Docker",
                              "Host Cyclictest - Stress-ng Unprivileged Docker",
                              "Host Cyclictest - Stress-ng Limited Docker",
                              "Host Cyclictest - Stress-ng Extreme Docker"])

cyclictestDocker = merge_data(cyclictestDockerStressNgHostAggregated,
                              cyclictestDockerStressNgPrivilegedDockerAggregated ,
                              cyclictestDockerStressNgUnprivilegedDockerAggregated,
                              cyclictestDockerStressNgLimitedDockerAggregated,
                              cyclictestDockerStressNgExtremeDockerAggregated,
                              names=["Docker Cyclictest - Stress-ng Host" ,
                                     "Docker Cyclictest - Stress-ng Privileged Docker",
                                     "Docker Cyclictest - Stress-ng Unprivileged Docker",
                                     "Docker Cyclictest - Stress-ng Limited Docker",
                                     "Docker Cyclictest - Stress-ng Extreme Docker"])

rtlaHost = merge_data(rtlaHostStressNgHostAggregated, rtlaHostStressNgPrivilegedDockerAggregated,
                      rtlaHostStressNgUnprivilegedDockerAggregated, rtlaHostStressNgLimitedDockerAggregated,
                      rtlaHostStressNgExtremeDockerAggregated,
                      names=["Host RTLA - Stress-ng Host",
                             "Host RTLA - Stress-ng Privileged Docker",
                             "Host RTLA - Stress-ng Unprivileged Docker",
                             "Host RTLA - Stress-ng Limited Docker",
                             "Host RTLA - Stress-ng Extreme Docker"])

rtlaDocker = merge_data(rtlaDockerStressNgHostAggregated, rtlaDockerStressNgPrivilegedDockerAggregated,
                        rtlaDockerStressNgUnprivilegedDockerAggregated, rtlaDockerStressNgLimitedDockerAggregated,
                        rtlaDockerStressNgExtremeDockerAggregated,
                        names=["Docker RTLA - Stress-ng Host",
                               "Docker RTLA - Stress-ng Privileged Docker",
                               "Docker RTLA - Stress-ng Unprivileged Docker",
                               "Docker RTLA - Stress-ng Limited Docker",
                               "Docker RTLA - Stress-ng Extreme Docker"])

allMerged = merge_data(
    cyclictestHostStressNgHostAggregated, cyclictestHostStressNgPrivilegedDockerAggregated,
    cyclictestHostStressNgUnprivilegedDockerAggregated, cyclictestHostStressNgLimitedDockerAggregated,
    cyclictestHostStressNgExtremeDockerAggregated, cyclictestDockerStressNgHostAggregated,
    cyclictestDockerStressNgPrivilegedDockerAggregated, cyclictestDockerStressNgUnprivilegedDockerAggregated,
    cyclictestDockerStressNgLimitedDockerAggregated, cyclictestDockerStressNgExtremeDockerAggregated,
    rtlaHostStressNgHostAggregated, rtlaHostStressNgPrivilegedDockerAggregated,
    rtlaHostStressNgUnprivilegedDockerAggregated, rtlaHostStressNgLimitedDockerAggregated,
    rtlaHostStressNgExtremeDockerAggregated, rtlaDockerStressNgHostAggregated,
    rtlaDockerStressNgPrivilegedDockerAggregated, rtlaDockerStressNgUnprivilegedDockerAggregated,
    rtlaDockerStressNgLimitedDockerAggregated, rtlaDockerStressNgExtremeDockerAggregated,
    names=[
        "Host Cyclictest - Stress-ng Host", "Host Cyclictest - Stress-ng Privileged Docker",
        "Host Cyclictest - Stress-ng Unprivileged Docker", "Host Cyclictest - Stress-ng Limited Docker",
        "Host Cyclictest - Stress-ng Extreme Docker", "Docker Cyclictest - Stress-ng Host",
        "Docker Cyclictest - Stress-ng Privileged Docker", "Docker Cyclictest - Stress-ng Unprivileged Docker",
        "Docker Cyclictest - Stress-ng Limited Docker", "Docker Cyclictest - Stress-ng Extreme Docker",
        "Host RTLA - Stress-ng Host", "Host RTLA - Stress-ng Privileged Docker",
        "Host RTLA - Stress-ng Unprivileged Docker", "Host RTLA - Stress-ng Limited Docker",
        "Host RTLA - Stress-ng Extreme Docker", "Docker RTLA - Stress-ng Host",
        "Docker RTLA - Stress-ng Privileged Docker", "Docker RTLA - Stress-ng Unprivileged Docker",
        "Docker RTLA - Stress-ng Limited Docker", "Docker RTLA - Stress-ng Extreme Docker"
    ]
)

allNormalCyclictest = merge_data(
    cyclictestHostStressNgHostAggregated,
    cyclictestHostStressNgPrivilegedDockerAggregated,
    cyclictestHostStressNgUnprivilegedDockerAggregated,
    cyclictestDockerStressNgHostAggregated,
    cyclictestDockerStressNgPrivilegedDockerAggregated,
    cyclictestDockerStressNgUnprivilegedDockerAggregated,
    names=[
        "Host Cyclictest - Stress-ng Host",
        "Host Cyclictest - Stress-ng Privileged Docker",
        "Host Cyclictest - Stress-ng Unprivileged Docker",
        "Docker Cyclictest - Stress-ng Host",
        "Docker Cyclictest - Stress-ng Privileged Docker",
        "Docker Cyclictest - Stress-ng Unprivileged Docker"
    ]
)

allNormalRtla = merge_data(
    rtlaHostStressNgHostAggregated,
    rtlaHostStressNgPrivilegedDockerAggregated,
    rtlaHostStressNgUnprivilegedDockerAggregated,
    rtlaDockerStressNgHostAggregated,
    rtlaDockerStressNgPrivilegedDockerAggregated,
    rtlaDockerStressNgUnprivilegedDockerAggregated,
    names=[
        "Host RTLA - Stress-ng Host",
        "Host RTLA - Stress-ng Privileged Docker",
        "Host RTLA - Stress-ng Unprivileged Docker",
        "Docker RTLA - Stress-ng Host",
        "Docker RTLA - Stress-ng Privileged Docker",
        "Docker RTLA - Stress-ng Unprivileged Docker"
    ]
)


########################################################################################################################
# Plotting
########################################################################################################################

# Latex Output #########################################################################################################

stress_ng_inputs = [rtlaDockerStressNgHostLog, rtlaDockerStressNgPrivilegedDockerLog,
                              rtlaDockerStressNgUnprivilegedDockerLog, rtlaDockerStressNgLimitedDockerLog,
                              rtlaDockerStressNgExtremeDockerLog, rtlaHostStressNgHostLog,
                              rtlaHostStressNgPrivilegedDockerLog, rtlaHostStressNgUnprivilegedDockerLog,
                              rtlaHostStressNgLimitedDockerLog, rtlaHostStressNgExtremeDockerLog,
                              cyclictestDockerStressNgHostLog, cyclictestDockerStressNgPrivilegedDockerLog,
                              cyclictestDockerStressNgUnprivilegedDockerLog, cyclictestDockerStressNgLimitedDockerLog,
                              cyclictestDockerStressNgExtremeDockerLog, cyclictestHostStressNgHostLog,
                              cyclictestHostStressNgPrivilegedDockerLog, cyclictestHostStressNgUnprivilegedDockerLog,
                              cyclictestHostStressNgLimitedDockerLog, cyclictestHostStressNgExtremeDockerLog]

rtla_stress_ng_inputs_complete = [rtlaDockerStressNgHostLog, rtlaDockerStressNgPrivilegedDockerLog,
                              rtlaDockerStressNgUnprivilegedDockerLog, rtlaDockerStressNgLimitedDockerLog,
                              rtlaDockerStressNgExtremeDockerLog, rtlaHostStressNgHostLog,
                              rtlaHostStressNgPrivilegedDockerLog, rtlaHostStressNgUnprivilegedDockerLog,
                              rtlaHostStressNgLimitedDockerLog, rtlaHostStressNgExtremeDockerLog]

cyclictest_stress_ng_inputs_all = [cyclictestDockerStressNgHostLog, cyclictestDockerStressNgPrivilegedDockerLog,
                              cyclictestDockerStressNgUnprivilegedDockerLog, cyclictestDockerStressNgLimitedDockerLog,
                              cyclictestDockerStressNgExtremeDockerLog, cyclictestHostStressNgHostLog,
                              cyclictestHostStressNgPrivilegedDockerLog, cyclictestHostStressNgUnprivilegedDockerLog,
                              cyclictestHostStressNgLimitedDockerLog, cyclictestHostStressNgExtremeDockerLog]

rtla_stress_ng_inputs = [rtlaDockerStressNgHostLog, rtlaDockerStressNgPrivilegedDockerLog,
                              rtlaDockerStressNgUnprivilegedDockerLog, rtlaDockerStressNgLimitedDockerLog]

rtla_stress_ng_inputs_extreme = [rtlaDockerStressNgLimitedDockerLog, rtlaDockerStressNgExtremeDockerLog]

rtla_stress_ng_inputs_all = [rtlaDockerStressNgHostLog, rtlaDockerStressNgPrivilegedDockerLog,
                              rtlaDockerStressNgUnprivilegedDockerLog, rtlaDockerStressNgLimitedDockerLog,
                             rtlaDockerStressNgExtremeDockerLog]

gen_tex_table(allMerged, ["count", "over", "max", "avg", "Q2", "StdDev"], "USR",
              f"{output}\\stats_overview.tex")

compare_and_generate_latex_table(rtla_stress_ng_inputs, ["RD-H", "RD-P", "RD-U", "RD-L"], f"{output}\\rtla_stress_ng_inputs_h_l.tex")
compare_and_generate_latex_table(rtla_stress_ng_inputs_extreme, ["RD-L", "RD-E"], f"{output}\\rtla_stress_ng_inputs_l_e.tex")

compare_and_generate_latex_table(rtla_stress_ng_inputs_all, ["RD-H", "RD-P", "RD-U", "RD-L", "RD-E"], f"{output}\\rtla_stress_ng_inputs_all.tex")
compare_and_generate_latex_table(rtla_stress_ng_inputs_complete, ["RD-H", "RD-P", "RD-U", "RD-L", "RD-E", "RH-H", "RH-P", "RH-U", "RH-L", "RH-E"], f"{output}\\stress_ng_inputs_allRtla.tex")
compare_and_generate_latex_table(cyclictest_stress_ng_inputs_all, ["CD-H", "CD-P", "CD-U", "CD-L", "CD-E", "CH-H", "CH-P", "CH-U", "CH-L", "CH-E"], f"{output}\\stress_ng_inputs_allCyclic.tex")

# Plot stress-ng information ###########################################################################################
gen_plot_stress_test_results(stress_ng_inputs,
                             names=["RD-H", "RD-P", "RD-U", "RD-L", "RD-E",
                                    "RH-H", "RH-P", "RH-U", "RH-L", "RH-E",
                                    "CD-H", "CD-P", "CD-U", "CD-L", "CD-E",
                                    "CH-H", "CH-P", "CH-U", "CH-L", "CH-E"],
                             save_path=f"{output}\\stress_test_results.pdf")

# Boxplots #############################################################################################################
boxplot_threads_level = f"{output}\\boxplots_threads_level"
os.makedirs(boxplot_threads_level, exist_ok=True)

for test in [rtlaDockerStressNgHost, rtlaDockerStressNgPrivilegedDocker, rtlaDockerStressNgUnprivilegedDocker,
             rtlaDockerStressNgLimitedDocker, rtlaDockerStressNgExtremeDocker, rtlaHostStressNgHost,
             rtlaHostStressNgPrivilegedDocker, rtlaHostStressNgUnprivilegedDocker, rtlaHostStressNgLimitedDocker,
             rtlaHostStressNgExtremeDocker, cyclictestDockerStressNgHost, cyclictestDockerStressNgPrivilegedDocker,
             cyclictestDockerStressNgUnprivilegedDocker, cyclictestDockerStressNgLimitedDocker,
             cyclictestDockerStressNgExtremeDocker, cyclictestHostStressNgHost, cyclictestHostStressNgPrivilegedDocker,
             cyclictestHostStressNgUnprivilegedDocker, cyclictestHostStressNgLimitedDocker,
             cyclictestHostStressNgExtremeDocker]:
    gen_boxplot_from_hist(test, "USR", x_label="Threads",
                          y_label="Latency (ns)",
                          save_path=f"{boxplot_threads_level}\\{get_var_name(test)}_USR_boxplot_stress-ng.pdf",
                          show_fliers=False)


gen_boxplot_from_hist(allNormalCyclictest, "USR", x_label="Experiments", y_label="Latency (ns)",
                      save_path=f"{output}\\allNonLimitedCyclictest_boxplot_stress-ng.pdf", figure_size=(6, 8),
                      labels=["HC-H", "HC-P", "HC-U", "DC-H", "DC-P", "DC-U"], show_fliers=False , y_limit=(0, 10000)
)

gen_boxplot_from_hist(allNormalRtla, "USR", x_label="Experiments", y_label="Latency (ns)",
                      save_path=f"{output}\\allNonLimitedRtla_boxplot_stress-ng.pdf", figure_size=(6, 8),
                      labels=["HR-H", "HR-P", "HR-U", "DR-H", "DR-P", "DR-U"], show_fliers=False, y_limit=(0, 10000)
)


boxplotRtlaThreadLevelOut = f"{output}\\boxplots_rtla_thread_level"
os.makedirs(boxplotRtlaThreadLevelOut, exist_ok=True)
for test in [rtlaDockerStressNgExtremeDocker, rtlaHostStressNgExtremeDocker, rtlaDockerStressNgLimitedDocker, rtlaHostStressNgLimitedDocker]:
    for type in ["USR", "THR", "IRQ"]:
        gen_boxplot_from_hist(test, type, save_path=f"{boxplotRtlaThreadLevelOut}\\{get_var_name(test)}_{type}_stress-ng_flier.pdf", figure_size=(3, 4), labels=[0,1,2,3], x_label="Threads")
        if type == "USR":
            y_limit = (0, 10000)
        elif type == "THR":
            y_limit = (0, 6000)
        elif type == "IRQ":
            y_limit = (0, 2000)
        gen_boxplot_from_hist(test, type, save_path=f"{boxplotRtlaThreadLevelOut}\\{get_var_name(test)}_{type}_stress-ng_flierless.pdf", show_fliers=False, figure_size=(2, 3), labels=[0,1,2,3], x_label="Threads", y_limit=y_limit)
        gen_tex_table(test, ["max", "avg", "Q2", "StdDev"], type, f"{boxplotRtlaThreadLevelOut}\\{get_var_name(test)}_{type}_stress-ng_table.tex")



#Histogram Data ########################################################################################################

# hist_output = f"{output}\\histograms"
# os.makedirs(hist_output, exist_ok=True)
#
# for test in [rtlaDockerStressNgHost, rtlaDockerStressNgPrivilegedDocker, rtlaDockerStressNgUnprivilegedDocker,
#              rtlaDockerStressNgLimitedDocker, rtlaDockerStressNgExtremeDocker, rtlaHostStressNgHost,
#              rtlaHostStressNgPrivilegedDocker, rtlaHostStressNgUnprivilegedDocker, rtlaHostStressNgLimitedDocker,
#              rtlaHostStressNgExtremeDocker, cyclictestDockerStressNgHost, cyclictestDockerStressNgPrivilegedDocker,
#              cyclictestDockerStressNgUnprivilegedDocker, cyclictestDockerStressNgLimitedDocker,
#              cyclictestDockerStressNgExtremeDocker, cyclictestHostStressNgHost, cyclictestHostStressNgPrivilegedDocker,
#              cyclictestHostStressNgUnprivilegedDocker, cyclictestHostStressNgLimitedDocker,
#              cyclictestHostStressNgExtremeDocker]:
#     gen_histogram(test, save_path=f"{hist_output}\\{get_var_name(test)}_hist", limit_x=15000)

#Verbose Data ##########################################################################################################

# timerseries_path = f"{output}\\timerseries"
# os.makedirs(timerseries_path, exist_ok=True)
# for test in ["DockerStress-ngHost_run1", "DockerStress-ngPrivilegedDocker_run1",
#              "DockerStress-ngUnprivilegedDocker_run1", "DockerStress-ngLimitedDocker_run1",
#              "DockerStress-ngExtremeDocker_run1", "HostStress-ngHost_run1",
#              "HostStress-ngPrivilegedDocker_run1", "HostStress-ngUnprivilegedDocker_run1",
#              "HostStress-ngLimitedDocker_run1", "HostStress-ngExtremeDocker_run1"]:
#     cyclictestVerboseOutput = parse_cyclictest_verbose_data(f"{cyclictestSubFolder}\\{test}\\output")
#     gen_timeseries(cyclictestVerboseOutput, save_path=f"{timerseries_path}\\{test}_none", sm='none', ylim=(0, 900000))
#     gen_timeseries(cyclictestVerboseOutput, save_path=f"{timerseries_path}\\{test}_max", sm='max', ylim=(0, 900000))
#     gen_timeseries(cyclictestVerboseOutput, save_path=f"{timerseries_path}\\{test}_min", sm='min', ylim=(0, 10000))
#     gen_timeseries(cyclictestVerboseOutput, save_path=f"{timerseries_path}\\{test}_mean", sm='mean', ylim=(0, 20000))

print("Done")
