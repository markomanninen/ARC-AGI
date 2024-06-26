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

    return correct_predictions

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

def plot_true_results(predictions):
    total = len(predictions)
    cols = int(np.ceil(np.sqrt(total)))
    rows = int(np.ceil(total / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    axes = axes.flatten()  # Flatten in case of a single row

    for ax, pred in zip(axes, predictions):
        true_output_grid = np.array(pred["true_output"])
        
        # Create the heatmap for the true output without annotations and add a frame border
        sns.heatmap(true_output_grid, ax=ax, cbar=False, annot=False, cmap='Blues', square=True, linewidths=.5, linecolor='black')
        ax.set_xticks([])
        ax.set_yticks([])

    # Remove unused axes
    for i in range(total, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout(pad=2.0)
    plt.show()

def main(llmClientName=""):
    data_dir = "../data/evaluation"
    log_dir = "./log_" + llmClientName if llmClientName else "./log"
    os.makedirs(log_dir, exist_ok=True)

    evaluation_files = load_evaluation_files(data_dir, log_dir)
    print("Evaluation files:", len(evaluation_files))
    
    correct_predictions = analyze_predictions(evaluation_files, log_dir)
    
    statistics_true = collect_statistics(correct_predictions)

    print("True - Input Sizes:", statistics_true["input_sizes"])
    print("True - Output Sizes:", statistics_true["output_sizes"])
    print("True - Input Colors:", statistics_true["input_colors"])
    print("True - Output Colors:", statistics_true["output_colors"])

    plot_true_results(correct_predictions)

if __name__ == "__main__":
    llmClientName = None  # None for Anthropic Claude 3.5 Sonnet or "openai" for GPT-4
    main(llmClientName)
