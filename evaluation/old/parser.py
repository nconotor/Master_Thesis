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


def save_to_txt(data_str, output_file):
    """
    Save the given data string to a text file.
    @param data_str: The data string to save.
    @param output_file: The path to the output file.
    @return: None
    """
    with open(output_file, 'w') as file:
        file.write(data_str)


def save_to_json(data, output_file):
    """
    Save the given data to a JSON file.
    @param data: The data to save.
    @param output_file: The path to the output file.
    @return: None
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)


def parse_cyclictest_verbose_data(file_path):
    """
    Reads and filters the file to extract CPU, Loop, and Latency (and optionally SMI count). Uses regex matching to
    parse the data. If SMI count is not present, it is set to 0. @param file_path: The path to the input file.
    @return: The filtered data as a DataFrame.  Columns: CPU, Tick, Latency, SMI Count (zero if not present).
    """
    pattern = re.compile(r'^\s*(\d+):\s+(\d+):\s+(\d+)(?:\s+(\d+))?\s*$')

    with open(file_path, 'r') as file:
        filtered_data = [
            [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4) or 0)]
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
    :param size: The size of all samples in the dataset.
    :param percentile: The percentile to get. Number between 0> and <100
    :param data_set: The data set to analyze.
    :return: The percentile of the data set.
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


def get_fliers_count(data_set, limit):
    """
    Get the number of fliers in the given data set.
    :param data_set: The data set to analyze.
    :param limit: The limit to consider a value as a flier.
    :return: The number of fliers in the data set. TODO: This does not include lower fliers. There might be some after switiching to ns.
    """
    flier: int = 0
    for index, pair in enumerate(data_set.items()):
        if int(pair[0]) >= limit:
            flier += pair[1]
    return flier


def get_fliers(data_set, limit):
    """
    Get the fliers in the given data set.
    :param data_set: The data set to analyze.
    :param limit: The limit to consider a value as a flier.
    :return: The fliers in the data set as a set. Therefore, lacking duplicates. TODO: This does not include lower fliers. There might be some after switiching to ns.
    """
    fliers = set([])
    for index, pair in enumerate(data_set.items()):
        if int(pair[0]) >= limit:
            fliers.add(pair[0])
    return fliers


def enrich_hist_data(hist_data):
    """
    Create the statistics for the given histogram data. The Whiskers are calculated as the 1.5 * IRQ from the Q1 and
    Q3 values and then set to the closest value in the data set inside these bounds.
    :param hist_data: The histogram data to analyze.
    :return: The enriched histogram data.
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
            type_data['IRQ'] = type_data['Q3'] - type_data['Q1']
            type_data['WhiskerLow'] = type_data['Q1'] - 1.5 * type_data['IRQ']
            type_data['WhiskerHigh'] = type_data['Q3'] + 1.5 * type_data['IRQ']
            type_data['WhiskerLowBounded'] = min((int(key) for key in hist if int(key) >= type_data['WhiskerLow']),
                                                 default=None)
            type_data['WhiskerHighBounded'] = max((int(key) for key in hist if int(key) <= type_data['WhiskerHigh']),
                                                  default=None)
            type_data['Fliers'] = get_fliers_count(hist, float(type_data['WhiskerHigh']))
            type_data['RealFliers'] = get_fliers_count(hist, float(type_data['WhiskerHigh'])) + int(type_data['over'])
            type_data['FlierPercentage'] = (type_data['RealFliers'] / sum(hist.values())) * 100
            type_data['FliersSet'] = list(get_fliers(hist, type_data['WhiskerHigh']))
    return hist_data


def get_boxplot_data(hist_data, data_type, labels=None):
    """
    Get the data for a boxplot from the enriched histogram data.
    :param hist_data: The enriched histogram data.
    :param data_type: The data type to get the boxplot data for. Normally USR, THR or IRQ.
    :return: The data for the boxplot.
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
            'iqr': float(data_segment['IRQ']),
            'whishi': float(data_segment['WhiskerHighBounded']),
            'whislo': float(data_segment['WhiskerLowBounded']),
            'fliers': data_segment["FliersSet"]
        }]
    return result


def gen_boxplot(boxplot_data, title=None, x_label=None, y_label="Latency (ns)",
                figure_size=(6, 6), palette="viridis", save_path=None, ylim=None):
    """
    Generate a boxplot from the given data with Seaborn color palettes and custom labels.
    :param boxplot_data: The data to generate the boxplot from. Is an array of list with ['label', 'mean', 'iqr',
     'cilo', 'cihi', 'whishi', 'whislo', 'fliers', 'q1', 'med', 'q3']
    :param title: Title of the boxplot.
    :param x_label: Label for the x-axis.
    :param y_label: Label for the y-axis.
    :param figure_size: Size of the figure.
    :param palette: Name of the Seaborn color palette to use.
    :return: None
    """
    fig, axs = plt.subplots(figsize=figure_size)
    meanprops = {'marker': '^', 'markerfacecolor': 'red', 'markeredgecolor': 'red'}
    box = axs.bxp(boxplot_data, showmeans=True, meanline=False, showfliers=False, patch_artist=True,
                  meanprops=meanprops)

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
    else:
        plt.show()
    plt.close()


def gen_histogram(hist_data, category='USR', save_path=None, title=None, figure_size=(10, 6), limit_x_to_p99=True):
    """
    Generate a histogram from the given data.
    @param hist_data:
    @param category:
    @param save_path:
    @param title:
    @param figure_size:
    @param limit_x_to_p99:
    @return:
    """
    plt.figure(figsize=figure_size)

    all_latencies = set()
    max_99 = 0
    for thread_data in hist_data.values():
        all_latencies.update(thread_data[category]['hist'].keys())
        max_99 = max(max_99, thread_data[category]['P99'])
    all_latencies = sorted(all_latencies)

    bottoms = np.zeros(len(all_latencies))

    colors = sns.color_palette("viridis", len(hist_data.keys()))

    for thread, color in zip(hist_data, colors):
        thread_hist = hist_data[thread][category]['hist']

        counts = [thread_hist.get(latency, 0) for latency in all_latencies]

        plt.bar(all_latencies, counts, bottom=bottoms, color=color, label=thread, align='center')

        bottoms += counts

    plt.xlabel('Latency')
    plt.ylabel('Count')
    if limit_x_to_p99:
        plt.xlim(right=max_99)
    if title:
        plt.title(title)
    plt.legend(title='Thread', loc='upper right')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def aggregate_data(data):
    """
    Aggregate the given data by combining histograms and recalculating count, min, avg, and max for each category within each thread.
    :param data: The data to aggregate, structured as a dictionary with thread keys, each containing 'USR', 'THR', and 'IRQ' sub-dictionaries.
    :return: The aggregated data.
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

    return aggregated_data


def generate_boxplot_from_hist(histogram_data, data_type, labels=None, title=None, x_label=None,
                               y_label="Latency (ns)", save_path=None, figure_size=(4, 6), palette="viridis",
                               align_size=True):
    """
    Generate a boxplot from the given histogram data. Wrapper function for get_boxplot_data and gen_boxplot.
    :param histogram_data: The histogram data to generate the boxplot from.
    :param data_type: The data type to get the boxplot data for. Normally USR, THR or IRQ.
    :param labels: The labels for the boxplot.
    :param title: Title of the boxplot
    :param x_label: Label for the x-axis
    :param y_label: Label for the y-axis
    :param save_path: Path to save the plot to.
    :return: None
    """
    bxp = get_boxplot_data(hist_data=histogram_data, data_type=data_type, labels=labels)
    y_limit = None
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
                palette=palette, ylim=y_limit)


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
    :param args: Unspecified number of dictionaries to merge.
    :param names: List of new names for the top-level keys of each dictionary. Must match the number of data dictionaries.
    :return: A dictionary with each input dictionary nested under a uniquely renamed key.
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


def gen_aggregated_barplot(hist_data):
    """
    TODO INCOMPLETE DID NOT WORK AS INTENDED. Especially the x labels do not work as intended.
    Generate a bar plot from the aggregated data.
    @param hist_data:  The histogram data to generate the bar plot from.
    @return: None
    """
    data = {
        'Experiment': [],
        'Count': [],
        'Fliers': [],
        'Over': [],
    }

    for exp, metrics in hist_data.items():
        data['Experiment'].append("exp")
        data['Count'].append(metrics['USR']['count'])
        data['Fliers'].append(metrics['USR']['Fliers'])
        data['Over'].append(metrics['USR']['over'] if metrics['USR']['over'] > 0 else 1)

    df = pd.DataFrame(data)
    print(df)

    ax = df.plot(x='Experiment',
                 kind='bar',
                 stacked=False,
                 log=True,
                 title='Grouped Bar Graph with DataFrame')

    ax.set_ylabel('Count')
    ax.set_xlabel('Experiment')
    ax.legend(title="Attributes")
    plt.show()


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
            ax.text(bar.get_x() + bar.get_width()/2, yval, '%d' % int(yval), ha='center', va='center', rotation=90, color='red')

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
                    row += f" & ${values[attribute]:.2f}$" if not values[
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


def calc_change_between_latency_types(data, key_to_compare):
    """
    Calculate the change between the given latency types.
    @param data: The dictionary containing the latency data with 3 latency types (IRQ, THR, USR).
    @param key_to_compare: The key to compare between the latency types.
    @return: The changes between the latency types. Tuple containing the changes between IRQ and THR, THR and USR, and IRQ and USR.
    """
    irq_to_thr_changes = []
    thr_to_usr_changes = []
    irq_to_usr_changes = []

    for i in range(len(data)):
        thread_key = f"thread{i}"
        thread_data = data.get(thread_key, {})

        irq_latency = thread_data.get("IRQ", {}).get(key_to_compare)
        thr_latency = thread_data.get("THR", {}).get(key_to_compare)
        usr_latency = thread_data.get("USR", {}).get(key_to_compare)

        if irq_latency is not None and thr_latency is not None and thr_latency != 0:
            irq_to_thr_changes.append(calculate_change(irq_latency, thr_latency)[1])
        else:
            irq_to_thr_changes.append(None)

        if thr_latency is not None and usr_latency is not None and usr_latency != 0:
            thr_to_usr_changes.append(calculate_change(thr_latency, usr_latency)[1])
        else:
            thr_to_usr_changes.append(None)

        if irq_latency is not None and usr_latency is not None and usr_latency != 0:
            irq_to_usr_changes.append(calculate_change(irq_latency, usr_latency)[1])
        else:
            irq_to_usr_changes.append(None)
    return irq_to_thr_changes, thr_to_usr_changes, irq_to_usr_changes


def gen_timeseries(df, sm='max', save_path=None, title=None, ylim=None, palette='viridis'):
    """
    Plot the timeseries of the given data frame.
    """
    custom_colors = sns.color_palette(palette, n_colors=len(df['CPU'].unique()))

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
                 color=custom_colors[cpu % len(custom_colors)])

    if title:
        plt.title(f'{plot_name} CPU Latencies over Time'.title())
    plt.xlabel('Loop')
    plt.ylabel('Latency (ns)')
    plt.legend(loc='upper right')
    plt.grid(True)

    if ylim:
        plt.ylim(ylim)

    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}\\timeseries_{plot_name}.pdf')
    else:
        plt.show()
    plt.close()


def gen_timeseries_per_thread(df, sm='max', save_path=None, title=None, ylim=None):
    """
    Plot the timeseries of the given data frame.
    @param ylim: The y-axis limits. TODO not implemented yet.
    @param save_path: The folder path to save the plot to.
    @param title: The title of the plot. TODO Not implemented yet. Currently toggle to turn title off.
    @param df: The data frame to plot. Needs to contain 'CPU', 'Tick', and 'Latency' columns.
    @param sm: The smoothing method to use. Can be 'max', 'mean', or 'none'.
    @return: None
    """

    custom_colors = sns.color_palette()

    plt.figure()

    for cpu, group in df.groupby('CPU'):
        plt.figure(figsize=(10, 6))
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
        if title:
            plt.title(f'CPU {cpu} {plot_name} Latency over Time'.title())
        plt.xlabel('Loop')
        plt.ylabel(f'Latency (ns)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(f'{save_path}\\timeseries_{cpu}_{plot_name}.pdf')
        else:
            plt.show()
        plt.close()


def gen_latex_change_between_categories(dictionary, list_of_keys):
    """

    @param dictionary:
    @param list_of_keys:
    @return:
    """
    latex_output = "\\begin{tabular}{|c|" + "c|" * (len(list_of_keys) - 1) * 2 + "}\n"
    latex_output += "\\hline\n"
    header_row = "Metric "
    for i in range(len(list_of_keys) - 1):
        header_row += f"& Δ {list_of_keys[i]} to {list_of_keys[i + 1]} (abs) & Δ {list_of_keys[i]} to {list_of_keys[i + 1]} (%) "
    latex_output += header_row + "\\\\\n\\hline\n"

    categories = dictionary[next(iter(dictionary))].keys()

    for category in categories:
        category_data = dictionary[next(iter(dictionary))][category]
        latex_output += category + " "
        for i in range(len(list_of_keys) - 1):
            if list_of_keys[i] in category_data and list_of_keys[i + 1] in category_data:
                abs_change, perc_change = calculate_change(category_data[list_of_keys[i]],
                                                           category_data[list_of_keys[i + 1]])
                latex_output += f"& {abs_change:.2f} & {perc_change:.2f}% "
            else:
                latex_output += "& N/A & N/A "
        latex_output += "\\\\\n\\hline\n"

    latex_output += "\\end{tabular}"
    return latex_output


def gen_latex_change_between_key(data, category, list_of_experiments):
    latex_output = "\\begin{tabular}{|c|" + "c|" * (len(list_of_experiments) - 1) * 2 + "}\n"
    latex_output += "\\hline\n"
    header_row = "Metric "
    for i in range(len(list_of_experiments) - 1):
        header_row += f"& Δ {list_of_experiments[i]} to {list_of_experiments[i + 1]} (abs) & Δ {list_of_experiments[i]} to {list_of_experiments[i + 1]} (%) "
    latex_output += header_row + "\\\\\n\\hline\n"

    metrics = data[list_of_experiments[0]][category].keys()

    for metric in metrics:
        latex_output += metric + " "
        for i in range(len(list_of_experiments) - 1):
            old_value = data[list_of_experiments[i]][category][metric]
            new_value = data[list_of_experiments[i + 1]][category][metric]
            abs_change, perc_change = calculate_change(old_value, new_value)
            latex_output += f"& {abs_change:.2f} & {perc_change:.2f}% "
        latex_output += "\\\\\n\\hline\n"

    latex_output += "\\end{tabular}"
    return latex_output


########################################################################################################################
# User defined variables
# Set the base path to the directory containing the output files
# Load the RTLA and cyclictest data
########################################################################################################################

basepath = "H:\\MA\\Final\\comp"
output = f"{basepath}\\output"

