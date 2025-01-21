import os
import json
from pathlib import Path
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def load_evaluation_files(data_dir, log_dir):
    evaluation_files = list(Path(data_dir).glob("*.json"))
    available_files = []
    
    for file_path in evaluation_files:
        log_exists = False
        with open(file_path, "r") as file:
            data = json.load(file)
            test_data = data["test"]
            log_file_path = os.path.join(log_dir, f"{file_path.stem}.log")
            if os.path.exists(log_file_path):
                log_exists = True
        if log_exists:
            available_files.append(file_path)
    
    return available_files

def load_log_file(log_file_path):
    with open(log_file_path, "r") as f:
        log_data = json.load(f)
    return log_data

def analyze_predictions(evaluation_files, log_dir):
    correct_predictions = []
    incorrect_predictions = []

    for file_path in evaluation_files:
        with open(file_path, "r") as file:
            data = json.load(file)
            test_data = data["test"]
            
            if len(test_data) > 1:
                print(f"More than one test data: {file_path}")

            for test_item in test_data[:1]:
                log_file_path = os.path.join(log_dir, f"{file_path.stem}.log")
                if os.path.exists(log_file_path):
                    log_data = load_log_file(log_file_path)
                    if log_data["error_flag"]:
                        print("ERROR FLAG SET!")
                    if log_data["comparison_flag"]:
                        correct_predictions.append({
                            "input": test_item["input"],
                            "true_output": test_item["output"]
                        })
                        #print(log_file_path)
                    else:
                        incorrect_predictions.append({
                            "input": test_item["input"],
                            "true_output": test_item["output"]
                        })
                        #print(log_file_path)

    return correct_predictions, incorrect_predictions

def collect_statistics(predictions):
    input_sizes = []
    output_sizes = []
    input_colors = Counter()
    output_colors = Counter()
    unique_color_counts = []

    for item in predictions:
        input_grid = np.array(item["input"])
        output_grid = np.array(item["true_output"])

        input_sizes.append(input_grid.shape)
        output_sizes.append(output_grid.shape)

        input_colors.update(input_grid.flatten())
        output_colors.update(output_grid.flatten())

        unique_color_count = len(set(input_grid.flatten()))
        unique_color_counts.append(unique_color_count)

    return {
        "input_sizes": Counter(input_sizes),
        "output_sizes": Counter(output_sizes),
        "input_colors": input_colors,
        "output_colors": output_colors,
        "unique_color_counts": Counter(unique_color_counts)
    }

from matplotlib.colors import LinearSegmentedColormap

def plot_statistics(statistics_true, statistics_false, total_true, total_false):
    # First window: non-heatmap plots
    fig1, axes1 = plt.subplots(2, 2, figsize=(20, 10))

    # Input color distributions
    true_input_colors = {k: v / 1000 for k, v in statistics_true["input_colors"].items()}
    false_input_colors = {k: v / 1000 for k, v in statistics_false["input_colors"].items()}
    axes1[0, 0].bar(false_input_colors.keys(), false_input_colors.values(), alpha=0.6, color='red', label='False')
    axes1[0, 0].bar(true_input_colors.keys(), true_input_colors.values(), alpha=0.6, color='green', label='True')
    axes1[0, 0].set_title('Input Color Distribution')
    axes1[0, 0].set_xlabel('Color')
    axes1[0, 0].set_ylabel('Count (k)')
    axes1[0, 0].legend()
    axes1[0, 0].set_xticks(range(0, 10))
    axes1[0, 0].set_xticklabels(range(0, 10))

    # Output color distributions
    true_output_colors = {k: v / 1000 for k, v in statistics_true["output_colors"].items()}
    false_output_colors = {k: v / 1000 for k, v in statistics_false["output_colors"].items()}
    axes1[0, 1].bar(false_output_colors.keys(), false_output_colors.values(), alpha=0.6, color='red', label='False')
    axes1[0, 1].bar(true_output_colors.keys(), true_output_colors.values(), alpha=0.6, color='green', label='True')
    axes1[0, 1].set_title('Output Color Distribution')
    axes1[0, 1].set_xlabel('Color')
    axes1[0, 1].set_ylabel('Count (k)')
    all_output_colors = list(true_output_colors.values()) + list(false_output_colors.values())
    axes1[0, 1].set_yticks(range(0, int(max(all_output_colors)) + 5, 5))
    axes1[0, 1].legend()
    axes1[0, 1].set_xticks(range(0, 10))
    axes1[0, 1].set_xticklabels(range(0, 10))

    # Unique color count distributions
    color_range = range(1, 11)
    true_color_counts = [statistics_true["unique_color_counts"].get(i, 0) for i in color_range]
    false_color_counts = [statistics_false["unique_color_counts"].get(i, 0) for i in color_range]
    axes1[1, 0].bar(color_range, false_color_counts, alpha=0.6, color='red', label='False')
    axes1[1, 0].bar(color_range, true_color_counts, alpha=0.6, color='green', label='True')
    axes1[1, 0].set_title('Evaluation Results by Unique Color Count')
    axes1[1, 0].set_xlabel('Unique Color Count')
    axes1[1, 0].set_ylabel('Number of Evaluations')
    axes1[1, 0].set_yticks(range(0, max(true_color_counts + false_color_counts) + 5, 5))
    axes1[1, 0].legend()
    axes1[1, 0].set_xticks(range(1, 11))
    axes1[1, 0].set_xticklabels(range(1, 11))

    # Total counts and percentages of correct and incorrect evaluations
    total = total_true + total_false
    true_percentage = (total_true / total * 100) if total > 0 else 0
    false_percentage = (total_false / total * 100) if total > 0 else 0
    labels = ['True', 'False']
    percentages = [true_percentage, false_percentage]
    counts = [total_true, total_false]
    axes1[1, 1].bar(labels, counts, alpha=0.6, color=['green', 'red'])
    axes1[1, 1].set_title('Total Counts and Percentages of Evaluations')
    axes1[1, 1].set_ylabel('Number of Evaluations')
    axes1[1, 1].set_yticks(range(0, max(counts) + 25, 25))
    for i, (count, percentage) in enumerate(zip(counts, percentages)):
        axes1[1, 1].text(i, count, f'{count} ({percentage:.1f}%)', ha='center', va='bottom')

    plt.tight_layout(pad=2.0)

    # Second window: heatmap plots
    fig2, axes2 = plt.subplots(2, 2, figsize=(20, 10))

    # Create custom colormaps
    true_cmap = LinearSegmentedColormap.from_list('true_cmap', ['white', 'green'])
    false_cmap = LinearSegmentedColormap.from_list('false_cmap', ['white', 'red'])

    # Input grid sizes heatmap (True)
    true_keys = statistics_true["input_sizes"].keys()
    false_keys = statistics_false["input_sizes"].keys()

    if true_keys and false_keys:
        max_rows = max(max(key[0] for key in true_keys), max(key[0] for key in false_keys))
        max_cols = max(max(key[1] for key in true_keys), max(key[1] for key in false_keys))
    elif true_keys:
        max_rows = max(key[0] for key in true_keys)
        max_cols = max(key[1] for key in true_keys)
    elif false_keys:
        max_rows = max(key[0] for key in false_keys)
        max_cols = max(key[1] for key in false_keys)
    else:
        max_rows, max_cols = 0, 0  # or some default value
        
    input_size_matrix_true = np.zeros((max_rows, max_cols))
    for key, value in statistics_true["input_sizes"].items():
        input_size_matrix_true[key[0]-1, key[1]-1] += value
    input_size_annotations_true = np.where(input_size_matrix_true != 0, input_size_matrix_true.astype(int), "")
    sns.heatmap(input_size_matrix_true, ax=axes2[0, 0], annot=input_size_annotations_true, fmt="", cmap=true_cmap, cbar=False, annot_kws={"size": 8}, linewidths=.5, linecolor='gray', square=True)
    axes2[0, 0].set_title('Input Grid Sizes (True)')
    axes2[0, 0].set_xlabel('Column')
    axes2[0, 0].set_ylabel('Row')
    axes2[0, 0].set_facecolor('white')
    axes2[0, 0].set_xticks([4, 9, 14, 19, 24])
    axes2[0, 0].set_xticklabels([5, 10, 15, 20, 25])
    axes2[0, 0].set_yticks([4, 9, 14, 19, 24])
    axes2[0, 0].set_yticklabels([5, 10, 15, 20, 25])

    # Input grid sizes heatmap (False)
    input_size_matrix_false = np.zeros((max_rows, max_cols))
    for key, value in statistics_false["input_sizes"].items():
        input_size_matrix_false[key[0]-1, key[1]-1] += value
    input_size_annotations_false = np.where(input_size_matrix_false != 0, input_size_matrix_false.astype(int), "")
    sns.heatmap(input_size_matrix_false, ax=axes2[0, 1], annot=input_size_annotations_false, fmt="", cmap=false_cmap, cbar=False, annot_kws={"size": 8}, linewidths=.5, linecolor='gray', square=True)
    axes2[0, 1].set_title('Input Grid Sizes (False)')
    axes2[0, 1].set_xlabel('Column')
    axes2[0, 1].set_ylabel('Row')
    axes2[0, 1].set_facecolor('white')
    axes2[0, 1].set_xticks([4, 9, 14, 19, 24])
    axes2[0, 1].set_xticklabels([5, 10, 15, 20, 25])
    axes2[0, 1].set_yticks([4, 9, 14, 19, 24])
    axes2[0, 1].set_yticklabels([5, 10, 15, 20, 25])

    # Output grid sizes heatmap (True)
    output_size_matrix_true = np.zeros((max_rows, max_cols))
    for key, value in statistics_true["output_sizes"].items():
        output_size_matrix_true[key[0]-1, key[1]-1] += value
    output_size_annotations_true = np.where(output_size_matrix_true != 0, output_size_matrix_true.astype(int), "")
    sns.heatmap(output_size_matrix_true, ax=axes2[1, 0], annot=output_size_annotations_true, fmt="", cmap=true_cmap, cbar=False, annot_kws={"size": 8}, linewidths=.5, linecolor='gray', square=True)
    axes2[1, 0].set_title('Output Grid Sizes (True)')
    axes2[1, 0].set_xlabel('Column')
    axes2[1, 0].set_ylabel('Row')
    axes2[1, 0].set_facecolor('white')
    axes2[1, 0].set_xticks([4, 9, 14, 19, 24])
    axes2[1, 0].set_xticklabels([5, 10, 15, 20, 25])
    axes2[1, 0].set_yticks([4, 9, 14, 19, 24])
    axes2[1, 0].set_yticklabels([5, 10, 15, 20, 25])

    # Output grid sizes heatmap (False)
    output_size_matrix_false = np.zeros((max_rows, max_cols))
    for key, value in statistics_false["output_sizes"].items():
        output_size_matrix_false[key[0]-1, key[1]-1] += value
    output_size_annotations_false = np.where(output_size_matrix_false != 0, output_size_matrix_false.astype(int), "")
    sns.heatmap(output_size_matrix_false, ax=axes2[1, 1], annot=output_size_annotations_false, fmt="", cmap=false_cmap, cbar=False, annot_kws={"size": 8}, linewidths=.5, linecolor='gray', square=True)
    axes2[1, 1].set_title('Output Grid Sizes (False)')
    axes2[1, 1].set_xlabel('Column')
    axes2[1, 1].set_ylabel('Row')
    axes2[1, 1].set_facecolor('white')
    axes2[1, 1].set_xticks([4, 9, 14, 19, 24])
    axes2[1, 1].set_xticklabels([5, 10, 15, 20, 25])
    axes2[1, 1].set_yticks([4, 9, 14, 19, 24])
    axes2[1, 1].set_yticklabels([5, 10, 15, 20, 25])

    plt.tight_layout(pad=2.0)

    # Display both windows at the same time
    plt.show()

def main(llmClientName=""):
    data_dir = "../data/evaluation"
    log_dir = "./log_"+llmClientName if llmClientName else "./log"
    os.makedirs(log_dir, exist_ok=True)

    evaluation_files = load_evaluation_files(data_dir, log_dir)
    print("Evaluation files:", len(evaluation_files))
    
    correct_predictions, incorrect_predictions = analyze_predictions(evaluation_files, log_dir)
    
    statistics_true = collect_statistics(correct_predictions)
    statistics_false = collect_statistics(incorrect_predictions)

    total_true = len(correct_predictions)
    total_false = len(incorrect_predictions)

    """
    print("True - Input Sizes:", statistics_true["input_sizes"])
    print("True - Output Sizes:", statistics_true["output_sizes"])
    print("True - Input Colors:", statistics_true["input_colors"])
    print("True - Output Colors:", statistics_true["output_colors"])
    
    print("False - Input Sizes:", statistics_false["input_sizes"])
    print("False - Output Sizes:", statistics_false["output_sizes"])
    print("False - Input Colors:", statistics_false["input_colors"])
    print("False - Output Colors:", statistics_false["output_colors"])
    """

    plot_statistics(statistics_true, statistics_false, total_true, total_false)

if __name__ == "__main__":
    llmClientName = "deepseek"  # None for Anthropic Claude 3.5 Sonnet or "openai" for GPT-4o
    main(llmClientName)
