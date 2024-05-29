# Plotting Log Files for pretraining


import json
import argparse
import matplotlib.pyplot as plt
from itertools import cycle

def extract_data_from_log(file_path):
    print(f"Processing file: {file_path}")  
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]

    start_last_run_index = None
    for i, entry in enumerate(reversed(data)):
        if entry["epoch"] == 0:
            start_last_run_index = len(data) - 1 - i
            break

    if start_last_run_index is not None:
        data = data[start_last_run_index:]

    return data

def plot_data(train_data, val_data, labels):
    plt.figure(figsize=(10, 5))
    label_colors = {'Animals': 'red', 'Non-animals': 'green', 'Mixed': 'blue'}

    for (data, val, label) in zip(train_data, val_data, labels):
        color = label_colors.get(label, 'black')  # Default to black if label not found
        epochs = [entry["epoch"] for entry in data]
        train_loss = [entry["train_loss"] for entry in data]
        val_epochs = [entry["epoch"] for entry in val]
        val_loss = [entry["val_loss"] for entry in val]

        plt.plot(epochs, train_loss, label=f'Train Loss - {label}', color=color)
        plt.plot(val_epochs, val_loss, label=f'Validation Loss - {label}', linestyle='--', color=color)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Pretraining Loss per Epoch')
    plt.tight_layout()

    output_file = 'pretraining_loss_plot.png'
    plt.savefig(output_file)
    print(f"Plots saved to {output_file}")

def main(train_log_files, val_log_files, labels):
    train_data = [extract_data_from_log(f) for f in train_log_files]
    val_data = [extract_data_from_log(f) for f in val_log_files]
    plot_data(train_data, val_data, labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training and validation loss from log files.')
    parser.add_argument('--train_log_files', nargs='+', help='Paths to the training log files', required=True)
    parser.add_argument('--val_log_files', nargs='+', help='Paths to the validation log files', required=True)
    parser.add_argument('--labels', nargs='+', help='Labels for the different runs', required=True)

    args = parser.parse_args()

    if len(args.train_log_files) != len(args.val_log_files) or len(args.train_log_files) != len(args.labels):
        raise ValueError("The number of training log files, validation log files, and labels must be the same")

    # make sure there's no duplicate in file paths
    if len(set(args.train_log_files)) != len(args.train_log_files):
        raise ValueError("Duplicate training log files detected.")
    if len(set(args.val_log_files)) != len(args.val_log_files):
        raise ValueError("Duplicate validation log files detected.")

    main(args.train_log_files, args.val_log_files, args.labels)






    










