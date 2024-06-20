import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import seaborn as sns

def split_camel_case(input_string):
    words = []
    start_index = 0
    for i, char in enumerate(input_string):
        if char.isupper() and i > 0:
            words.append(input_string[start_index:i])
            start_index = i
    words.append(input_string[start_index:])
    return ' '.join(words)


def save_to_json(data, output_file):
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)


def save_to_txt(data_str, output_file):
    with open(output_file, 'w') as file:
        file.write(data_str)


def parse_folder_name(folder_name):
    match = re.match(r'(.+)_run(\d+)', folder_name)
    if match:
        return match.group(1), int(match.group(2))
    return None, None


def compute_statistics(histogram):
    sum_values = []
    for latency, count in histogram.items():
        sum_values.extend([int(latency)] * count)
    average = np.mean(sum_values)
    variance = np.var(sum_values)
    p99 = np.percentile(sum_values, 99)
    p999 = np.percentile(sum_values, 99.9)
    keys, values = zip(*histogram.items())
    intKeys = [int(key) for key in histogram.keys()]
    span = max(intKeys) - min(intKeys)
    return average, variance, span, p99, p999


def process_folders(base_path=None):
    if base_path is None:
        base_path = os.getcwd()

    data = {}
    for folder_name in os.listdir(base_path):
        full_path = os.path.join(base_path, folder_name)
        if os.path.isdir(full_path):
            name, run_number = parse_folder_name(folder_name)
            if name and run_number is not None:
                file_path = os.path.join(full_path, 'output.json')
                if os.path.exists(file_path):
                    with open(file_path, 'r') as file:
                        file_data = json.load(file)
                        thread_data = file_data.get('thread', {})
                        for thread_id, thread_info in thread_data.items():
                            histogram = thread_info.get('histogram', {})
                            average, variance, span, p99, p999 = compute_statistics(histogram)
                            thread_info['average'] = average
                            thread_info['variance'] = variance
                            thread_info['span'] = span
                            thread_info['p99'] = p99
                            thread_info['p999'] = p999

                        if name not in data:
                            data[name] = {}
                        data[name][run_number] = thread_data
                else:
                    print(f"output.json does not exist. Skipping.")
            else:
                print(f"Folder name {folder_name} could not be parsed. Skipping.")
        else:
            print(f"Path {full_path} is not a directory. Skipping.")

    return data


def aggregate_data(data):
    aggr_data = {}

    for name, runs in data.items():
        all_values = []
        combined_histogram = {}

        for run_number, threads in runs.items():
            for thread_id, thread_info in threads.items():
                histogram = thread_info['histogram']
                for latency, count in histogram.items():
                    all_values.extend([int(latency)] * count)
                    latency_int = int(latency)
                    if latency_int in combined_histogram:
                        combined_histogram[latency_int] += count
                    else:
                        combined_histogram[latency_int] = count

        overall_average = np.mean(all_values) if all_values else 0
        overall_variance = np.var(all_values) if all_values else 0
        p99 = np.percentile(all_values, 99)
        p999 = np.percentile(all_values, 99.9)
        int_keys = [int(key) for key in combined_histogram.keys()]
        max_val = max(int_keys)
        span = max_val - min(int_keys)

        sorted_combined_histogram = OrderedDict(sorted(combined_histogram.items()))
        aggr_data[name] = {
            'average_latency': overall_average,
            'variance': overall_variance,
            'combined_histogram': sorted_combined_histogram,
            'span': span,
            'max': max_val,
            'p99': p99,
            'p999': p999
        }

    return aggr_data


def generate_scatter_plot_for_categories(data, data_type='avg', legend=False):
    fig, ax = plt.subplots()
    y_labels = []
    spans = {}
    y_position = 0

    catSize = 0
    for category, runs in data.items():
        for _ in runs.keys():
            catSize += 1

    palette = sns.husl_palette(catSize)

    for category, runs in data.items():
        latencies = []

        for runidx, (run, threads) in enumerate(runs.items()):
            for thread_id, thread_info in threads.items():
                latency = thread_info[data_type]
                latencies.append(latency)
                ax.scatter(latency, y_position, color=palette[runidx], marker='.', s=50)

        if latencies:
            overall_avg = sum(latencies) / len(latencies)
            span = max(latencies) - min(latencies)
            spans[category] = span
            ax.scatter(overall_avg, y_position, color='black', marker='v')
            ax.text(overall_avg, y_position + 0.1, f' {overall_avg:.2f}', verticalalignment='center',
                    horizontalalignment='center')

        y_labels.append(split_camel_case(category).title())
        y_position += 1

    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Average Latency (μs)' if data_type == 'avg' else 'Maximum Latency (μs)')

    if legend:
        for category, span in spans.items():
            ax.plot([], [], ' ', label=f"{split_camel_case(category)} span: {span:.2f}".title())
        ax.legend()

    plt.tight_layout()
    filename = f"scatter_plot_categories_{data_type}_labels.pdf"
    plt.savefig(filename)
    plt.close()
    print(f"Scatter plot saved as {filename}")


def generate_latex_table(data):
    latex_str = "\\begin{table}[H]\n\\centering\n"
    latex_str += "\\begin{tabularx}{\\textwidth}{|c|Y|Y|Y|Y|Y|}\n\\hline\n"
    latex_str += "\\rowcolor{lightgray} Run  & Avg & Max & Variance & Span &  $99.9\\%$ Quantile \\\\ \\hline\n"
    labelName = []
    for name, runs in data.items():
        labelName.extend(name.capitalize())
        for run_number, threads in sorted(runs.items()):
            run_stats = {
                'avg': [],
                'max': [],
                'variance': [],
                'span': [],
                'p999': []
            }

            for thread_id, thread_info in threads.items():
                run_stats['avg'].append(thread_info['average'])
                run_stats['max'].append(thread_info['max'])
                run_stats['variance'].append(thread_info['variance'])
                run_stats['span'].append(thread_info['span'])
                run_stats['p999'].append(thread_info['p999'])

            row = f"{split_camel_case(name).title()} {run_number} & "
            row += f"${min(run_stats['avg']):.2f}-{max(run_stats['avg']):.2f}$ $({np.mean(run_stats['avg']):.2f})$ & "
            row += f"${min(run_stats['max'])} - {max(run_stats['max'])}$ $({np.mean(run_stats['max']):.2f})$ & "
            row += f"${min(run_stats['variance']):.2f} - {max(run_stats['variance']):.2f}$ $({np.mean(run_stats['variance']):.2f})$ & "
            row += f"${min(run_stats['span'])} - {max(run_stats['span'])}$ $({np.mean(run_stats['span']):.2f})$ & "
            row += (
                f"${min(run_stats['p999']):.2f} - {max(run_stats['p999']):.2f}$ $({np.mean(run_stats['p999']):.2f})$ "
                f"\\\\\n")
            latex_str += row
        latex_str += "\\hline \n"

    latex_str += "\\end{tabularx}\n"
    latex_str += "\\caption{Your caption here.}\n"
    latex_str += "\\label{tab:" + str().join(labelName) + "}\n"
    latex_str += "\\end{table}"

    return latex_str


def generate_latex_table_aggregated(data):
    latex_str = "\\begin{table}[H]\n\\centering\n"
    latex_str += "\\begin{tabularx}{\\textwidth}{|c|Y|Y|Y|Y|Y|}\n\\hline\n"
    latex_str += "\\rowcolor{lightgray} Category & Avg & Max & Variance & Span & $99.9\\%$ Quantile \\\\ \\hline\n"

    for name, stats in data.items():
        row = f"{split_camel_case(name).title()} & "
        row += f"{stats['average_latency']:.2f} & "
        row += f"{stats['max']:.2f} & "
        row += f"{stats['variance']:.2f} & "
        row += f"{stats['span']} & "
        row += f"{stats['p999']:.2f} \\\\\\hline\n"
        latex_str += row

    latex_str += "\\end{tabularx}\n"
    latex_str += "\\caption{Aggregated Data Statistics.}\n"
    latex_str += "\\label{tab:aggregated_data}\n"
    latex_str += "\\end{table}\n"

    return latex_str


def create_histogram_for_category(input_data, category, compare_with_other=None, x_limit=None, log_scale=False,
                                  xSize=10, ySize=6, automaticSize=True):
    if not automaticSize:
        plt.figure(figsize=(xSize, ySize))

    if category in input_data:
        category_data = input_data[category]['combined_histogram']
        latencies, counts = zip(*category_data.items())
        plt.bar(latencies, counts, color='red', alpha=0.7, label=category.title())
    else:
        print(f"Category '{category}' not found in the aggregated data.")
        return

    if compare_with_other is not None and compare_with_other in input_data:
        host_data = input_data[compare_with_other]['combined_histogram']
        host_latencies, host_counts = zip(*host_data.items())
        plt.bar(host_latencies, host_counts, color='blue', alpha=0.3, label=compare_with_other.title())

    plt.xlabel('Latency (μs)')
    plt.ylabel('Samples')
    plt.legend()

    if x_limit:
        plt.xlim([0, x_limit])

    if log_scale:
        plt.yscale('log')

    plt.tight_layout()
    plt.savefig(f"hist_{category}v{compare_with_other}.pdf")


categorized_data = process_folders()
save_to_json(categorized_data, 'categorized_data.json')
aggregated_data = aggregate_data(categorized_data)
save_to_json(aggregated_data, 'aggregated_data.json')

generate_scatter_plot_for_categories(categorized_data, "max")
generate_scatter_plot_for_categories(categorized_data, "avg")

latex_table_cat = generate_latex_table(categorized_data)
save_to_txt(latex_table_cat, 'cat_table.txt')

latex_table_aggr = generate_latex_table_aggregated(aggregated_data)
save_to_txt(latex_table_aggr, 'aggr_table.txt')

create_histogram_for_category(aggregated_data, 'broadwellHostStress', compare_with_other="broadwellDockerStress", x_limit=40, log_scale=True)
create_histogram_for_category(aggregated_data, 'broadwellDockerStress-ng', compare_with_other="broadwellHostStress-ng", x_limit=40, log_scale=True)
