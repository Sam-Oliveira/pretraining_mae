import json
import matplotlib.pyplot as plt

def extract_data_from_log(file_path):
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

def plot_loss(train_data, labels):
    colors = {'Finetuned': 'red', 'Finetuned-half': 'green', 'Resnet-50': 'blue', 'MAE Supervised': 'orange'}
    plt.figure(figsize=(10, 5))
    for data, label in zip(train_data, labels):
        epochs = [entry["epoch"] for entry in data]
        train_loss = [entry["train_loss"] for entry in data]
        test_loss = [entry["test_loss"] for entry in data]
        color = colors[label]

        plt.plot(epochs, train_loss, label=f'Train Loss - {label}', color=color)
        plt.plot(epochs, test_loss, label=f'Validation Loss - {label}', linestyle='--', color=color)
    
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.title('Loss per Epoch', fontsize=20)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig('final_res_loss_plot.pdf')
    plt.close()

def plot_accuracy(train_data, labels):
    colors = {'Finetuned': 'red', 'Finetuned-half': 'green', 'Resnet-50': 'blue', 'MAE Supervised': 'orange'}
    plt.figure(figsize=(10, 5))
    for data, label in zip(train_data, labels):
        epochs = [entry["epoch"] for entry in data]
        test_accuracy = [entry["test_accuracy"] if label != 'Resnet-50' else entry["test_accuracy"] * 100 for entry in data]
        color = colors[label]

        plt.plot(epochs, test_accuracy, label=f'Accuracy - {label}', color=color)
    
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    plt.title('Accuracy per Epoch', fontsize=20)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig('final_res_accuracy_plot.pdf')
    plt.close()

def plot_iou(train_data, labels):
    colors = {'Finetuned': 'red', 'Finetuned-half': 'green', 'Resnet-50': 'blue', 'MAE Supervised': 'orange'}
    plt.figure(figsize=(10, 5))
    for data, label in zip(train_data, labels):
        epochs = [entry["epoch"] for entry in data]
        test_iou = [entry["test_IOU"] for entry in data]
        color = colors[label]

        plt.plot(epochs, test_iou, label=f'IOU - {label}', color=color)
    
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('IOU', fontsize=18)
    plt.title('IOU per Epoch', fontsize=20)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig('final_res_iou_plot.pdf')
    plt.close()

def main(log_files, labels):
    train_data = [extract_data_from_log(log_file) for log_file in log_files]
    plot_loss(train_data, labels)
    plot_accuracy(train_data, labels)
    plot_iou(train_data, labels)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot training and validation metrics from log files.')
    parser.add_argument('--log_files', nargs='+', required=True, help='Paths to the log files')
    parser.add_argument('--labels', nargs='+', required=True, help='Labels for the plot')
    args = parser.parse_args()

    if len(args.log_files) != len(args.labels):
        raise ValueError("The number of log files and labels must match.")

    main(args.log_files, args.labels)


# import json
# import matplotlib.pyplot as plt

# def extract_data_from_log(file_path):
#     with open(file_path, 'r') as file:
#         data = [json.loads(line) for line in file]

#     start_last_run_index = None
#     for i, entry in enumerate(reversed(data)):
#         if entry["epoch"] == 0:
#             start_last_run_index = len(data) - 1 - i
#             break

#     if start_last_run_index is not None:
#         data = data[start_last_run_index:]

#     return data

# def plot_loss(train_data, labels):
#     colors = {'Finetuned': 'red', 'Finetuned-half': 'green', 'Resnet-50': 'blue'}
#     plt.figure(figsize=(10, 5))
#     for data, label in zip(train_data, labels):
#         epochs = [entry["epoch"] for entry in data]
#         train_loss = [entry["train_loss"] for entry in data]
#         test_loss = [entry["test_loss"] for entry in data]
#         color = colors[label]

#         plt.plot(epochs, train_loss, label=f'Train Loss - {label}', color=color)
#         plt.plot(epochs, test_loss, label=f'Validation Loss - {label}', linestyle='--', color=color)
    
#     plt.xlabel('Epochs', fontsize=18)
#     plt.ylabel('Loss', fontsize=18)
#     plt.title('Loss per Epoch', fontsize=20)
#     plt.legend(fontsize=14)
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.tight_layout()
#     plt.savefig('new_res_loss_plot.pdf')
#     plt.close()

# def plot_accuracy(train_data, labels):
#     colors = {'Finetuned': 'red', 'Finetuned-half': 'green', 'Resnet-50': 'blue'}
#     plt.figure(figsize=(10, 5))
#     for data, label in zip(train_data, labels):
#         epochs = [entry["epoch"] for entry in data]
#         if label == 'Resnet-50':
#             test_accuracy = [entry["test_accuracy"] * 100 for entry in data]
#         else:
#             test_accuracy = [entry["test_accuracy"] for entry in data]
#         color = colors[label]

#         plt.plot(epochs, test_accuracy, label=f'Accuracy - {label}', color=color)
    
#     plt.xlabel('Epochs', fontsize=18)
#     plt.ylabel('Accuracy', fontsize=18)
#     plt.title('Accuracy per Epoch', fontsize=20)
#     plt.legend(fontsize=14)
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.tight_layout()
#     plt.savefig('new_res_accuracy_plot.pdf')
#     plt.close()

# def plot_iou(train_data, labels):
#     colors = {'Finetuned': 'red', 'Finetuned-half': 'green', 'Resnet-50': 'blue'}
#     plt.figure(figsize=(10, 5))
#     for data, label in zip(train_data, labels):
#         epochs = [entry["epoch"] for entry in data]
#         test_iou = [entry["test_IOU"] for entry in data]
#         color = colors[label]

#         plt.plot(epochs, test_iou, label=f'IOU - {label}', color=color)
    
#     plt.xlabel('Epochs', fontsize=18)
#     plt.ylabel('IOU', fontsize=18)
#     plt.title('IOU per Epoch', fontsize=20)
#     plt.legend(fontsize=14)
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.tight_layout()
#     plt.savefig('new_res_iou_plot.pdf')
#     plt.close()

# def main(log_files, labels):
#     train_data = [extract_data_from_log(log_file) for log_file in log_files]
#     plot_loss(train_data, labels)
#     plot_accuracy(train_data, labels)
#     plot_iou(train_data, labels)

# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser(description='Plot training and validation metrics from log files.')
#     parser.add_argument('--log_files', nargs='+', required=True, help='Paths to the log files')
#     parser.add_argument('--labels', nargs='+', required=True, help='Labels for the plot')
#     args = parser.parse_args()

#     if len(args.log_files) != len(args.labels):
#         raise ValueError("The number of log files and labels must match.")

#     main(args.log_files, args.labels)
