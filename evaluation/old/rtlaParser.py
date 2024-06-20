import re
import matplotlib.pyplot as plt
import json
import numpy as np
import os
import seaborn as sns
import pandas as pd


def save_to_json(data, output_file):
    """
    Save the given data to a JSON file.
    @param data: The data to save.
    @param output_file: The path to the output file.
    @return:
    """
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)


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
    :return: The number of fliers in the data set.
    """
    cur_pos = 0
    flier: int = 0
    for index, pair in enumerate(data_set.items()):
        cur_pos += int(pair[1])
        if cur_pos >= limit:
            flier += cur_pos
    return flier


def enrich_hist_data(hist_data):
    """
    Create the statistics for the given histogram data.
    :param hist_data: The histogram data to analyze.
    :return: The enriched histogram data.
    """
    for thread_id in hist_data.keys():
        thread_data = hist_data[thread_id]
        for type in thread_data.keys():
            type_data = thread_data[type]
            hist = type_data['hist']
            type_data['Q1'] = get_percentile(int(type_data["count"]), 25, hist)
            type_data['Q2'] = get_percentile(int(type_data["count"]), 50, hist)
            type_data['Q3'] = get_percentile(int(type_data["count"]), 75, hist)
            type_data['IRQ'] = type_data['Q3'] - type_data['Q1']
            type_data['WhiskerLow'] = type_data['Q1'] - 1.5 * type_data['IRQ']
            type_data['WhiskerHigh'] = type_data['Q3'] + 1.5 * type_data['IRQ']
            type_data['Fliers'] = get_fliers_count(hist, float(type_data['WhiskerHigh']))
            type_data['RealFliers'] = get_fliers_count(hist, float(type_data['WhiskerHigh'])) + int(type_data['over'])
    return hist_data


def get_boxplot_data(hist_data, data_type, labels=None):
    """
    Get the data for a boxplot from the enriched histogram data.
    :param hist_data: The enriched histogram data.
    :param data_type: The data type to get the boxplot data for. Normally USR, THR or IRQ.
    :return: The data for the boxplot.
    """
    result = []
    for idx,thread_id in enumerate(hist_data.keys()):
        data = hist_data[thread_id][data_type]
        result += [{
            'label': labels[idx] if labels else thread_id,
            'mean': float(data['avg']),
            'med': float(data['Q2']),
            'q1': float(data['Q1']),
            'q3': float(data['Q3']),
            'iqr': float(data['IRQ']),
            'whishi': float(data['WhiskerHigh']),
            'whislo': float(data['WhiskerLow']),
            'fliers': []
        }]
    return result


def gen_boxplot(boxplot_data, title="Boxplot", x_label="Threads", y_label="Latency (μs)",
                figure_size=(6, 6), palette="viridis"):
    """
    Generate a boxplot from the given data with Seaborn color palettes and custom labels.
    :param boxplot_data: The data to generate the boxplot from. Is an array of list with ['label', 'mean', 'iqr', 'cilo', 'cihi', 'whishi', 'whislo', 'fliers', 'q1', 'med', 'q3']
    :param title: Title of the boxplot.
    :param x_label: Label for the x-axis.
    :param y_label: Label for the y-axis.
    :param figure_size: Size of the figure.
    :param palette: Name of the Seaborn color palette to use.
    :return: None
    """
    fig, axs = plt.subplots(figsize=figure_size)
    meanprops = {'marker':'^', 'markerfacecolor':'red', 'markeredgecolor':'red'}
    box = axs.bxp(boxplot_data, showmeans=True, meanline=False, showfliers=False, patch_artist=True,meanprops=meanprops)

    colors = sns.color_palette(palette, n_colors=len(boxplot_data))
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    axs.set_title(title)
    axs.set_xlabel(x_label)
    axs.set_ylabel(y_label)
    plt.show()


def gen_histogram(index_count_paris):
    """
    Generate a histogram from the given data.
    :param index_count_paris: dictionary with the index as key and the count as value.
    :return: None
    """
    fig, axs = plt.subplots(figsize=(6, 6))
    axs.hist(index_count_paris.keys(), weights=index_count_paris.values(), bins=range(50))
    plt.show()


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
            raise ValueError("Each dictionary must have exactly one top-level key.")

        # Extract the single key in the dictionary
        original_key = next(iter(data_dict))
        # Replace the original top-level key with the custom name
        merged_data[name] = data_dict[original_key]

    return merged_data


basepath = "H:\\MA\\comparison_b1_broadwell_rtla_662328realtime11rtlts"
dataDockerNg = parse_rtla_data(f'{basepath}\\DockerStress-ng_run1\\rtla_output')
dataHostNg = parse_rtla_data(f'{basepath}\\HostStress-ng_run1\\rtla_output')
dataDocker = parse_rtla_data(f'{basepath}\\DockerStress_run1\\rtla_output')
dataHost = parse_rtla_data(f'{basepath}\\HostStress_run1\\rtla_output')
save_to_json(dataDockerNg, f'{basepath}\\output\\dataDockerNg.json')

cyclictest_data = parse_cyclictest_json_data(
    f'H:\\MA\\comparison_b1_broadwell_cyclic_662328realtime11rtlts\\DockerStress-ng_run1\\output.json')

# print(cyclictest_data)
# latency_types = ['IRQ', 'THR', 'USR']
aggr1 = aggregate_data(dataDockerNg)
aggr2 = aggregate_data(dataHostNg)

enrich_hist_data(aggr1)
enrich_hist_data(aggr2)

merged = merge_data(aggr1, aggr2, names=['DockerNg', 'HostNg'])
save_to_json(merged, f'{basepath}\\output\\dataMerged.json')
bp = get_boxplot_data(merged, 'USR', labels=['DockerNg', 'HostNg'])
gen_boxplot(bp)

# save_to_json(aggr, f'{basepath}\\output\\dataDockerNgAggr.json')
# enrich_hist_data(cyclictest_data)
# box = get_boxplot_data(cyclictest_data, 'USR')
# gen_boxplot(box)
# gen_histogram(dataDockerNg["thread0"]["THR"]["hist"])
