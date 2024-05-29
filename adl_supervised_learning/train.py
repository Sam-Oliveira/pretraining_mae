# Import libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys

# Check if cude is available
print(f'Cuda available: {torch.cuda.is_available()}')

# Assuming the utils/ and models/ directories are in the path
sys.path.append('./utils')
sys.path.append('./models')

print('Importing dataset.py and resnet.py')

from dataset import *  # Make sure this class is defined to handle the splits
from resnet import ResNetSegmentation   # Make sure the ResNet model is defined for segmentation

# Define a custom IoU Metric for validating the model.
def IoUMetric(pred, gt, softmax=False):

    pred=torch.argmax(pred,dim=1,keepdim=True)

    pred = torch.cat([ (pred == 0)], dim=1)
    #Add channel dimension to targets that was removed when computing CE loss in line 136
    #gt=gt[:,None,:,:]

    gt = torch.cat([ (gt == 0)], dim=1)

    intersection = torch.logical_and(gt,pred)
    union = torch.logical_or(gt,pred)

    # Compute the sum over all the dimensions except for the batch dimension.
    iou = (intersection.sum(dim=(1, 2, 3)) + 0.001) / (union.sum(dim=(1, 2, 3)) + 0.001)
    
    # Compute the mean over the batch dimension.
    return iou.mean()


def validate_model(model, dataloader, device, criterion):

    print('Validation started')

    # Set model to evaluation
    model.eval()

    # Initialise variables
    val_running_loss = 0.0
    correct = 0
    total = 0

    # Iterate over datalader
    for images, masks in dataloader:
        # Put images and mask on device
        images, masks = images.to(device), masks.to(device)
        masks = masks.squeeze(1)  # Removes the channel dimension, changing shape from [batch_size, 1, H, W] to [batch_size, H, W]
        masks = masks.long()      # Convert to long integers for compatibility with nn.CrossEntropyLoss

        # Forward pass
        outputs = model(images) # logits

        # Track IoU / inputs are logits and must be transformed
        val_IoU = IoUMetric(outputs, masks.unsqueeze(1), softmax=True)

        # Predicted mask
        _, predicted = torch.max(outputs, 1)

        # Compute error with cross-entropy
        val_running_loss += criterion(outputs, masks).item() * images.size(0)

        # Coompute correctly predicted pixels
        correct += (predicted == masks).sum().item()
        total += masks.numel()
    
    # Validation loss and accuracy
    val_loss = val_running_loss / len(dataloader.dataset)
    val_accuracy = correct / total

    return val_loss, val_accuracy, val_IoU

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, project_path):

    # Determine the number of parameters in the model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Define the log directory and file
    log_dir = os.path.join(project_path, 'logs')
    log_file_path = os.path.join(log_dir, 'training_log.txt')

    # Open the log file
    with open(log_file_path, 'a', buffering=1) as log_file:

        # Iterate over num_epochs
        for epoch in range(num_epochs):
            # Set model to training mode
            model.train()

            # Initialise running loss
            running_loss = 0.0

            # Iterate through training loader
            for images, masks in train_loader:

                # Put images and mask on device
                images, masks = images.to(device), masks.to(device)
                masks = masks.squeeze(1)  # Removes the channel dimension, changing shape from [batch_size, 1, H, W] to [batch_size, H, W]
                masks = masks.long()      # Convert to long integers for compatibility with nn.CrossEntropyLoss

                # Empty gradient
                optimizer.zero_grad()

                # Forward pass
                outputs = model(images) # logits

                # Track IoU / inputs are logits and must be transformed
                IoU = IoUMetric(outputs, masks.unsqueeze(1), softmax=True)

                # Loss, backward pass and optimizer step
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                # Compute running loss
                running_loss += loss.item() * images.size(0)

            # Print epoch loss
            epoch_loss = running_loss / len(train_loader.dataset)
            val_loss, val_acc, val_IoU = validate_model(model, val_loader, device, criterion)
            print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, Val Acc: {(100 * val_acc):.4f}, IoU: {IoU:.4f}, Val IoU: {val_IoU:.4f}')
            
            # Save model
            print(f'Save model for epoch {epoch}')
            torch.save(model.state_dict(), os.path.join(project_path, f'model_checkpoints/model_checkpoint_epoch{epoch}.pth'))

            # Log the metrics
            log_file.write(f'{{"epoch": {epoch}, "train_loss": {epoch_loss}, "test_loss": {val_loss}, "test_accuracy": {val_acc}, "train_IOU": {IoU}, "test_IOU": {val_IoU}, "n_parameters": {n_parameters}}}\n')

    print('Finished Training')

print('File is about to call the main()')


def main():
    # Configuration and hyperparameters
    root_dir = '/cs/student/projects3/COMP0197/grp3/adl_groupwork/adl_supervised_learning/'
    project_path = root_dir
    batch_size = 32 #32
    num_epochs = 100
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Call image and mask transform from dataset.py
    image_transform = create_image_transform()
    mask_transform = create_mask_transform()

    # Load datasets
    train_dataset = OxfordPetsDataset(root_dir=root_dir, split='train', image_transform=image_transform, mask_transform=mask_transform, seed=42)
    val_dataset = OxfordPetsDataset(root_dir=root_dir, split='val', image_transform=image_transform, mask_transform=mask_transform, seed=42)
    test_dataset = OxfordPetsDataset(root_dir=root_dir, split='test', image_transform=image_transform, mask_transform=mask_transform, seed=42)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss, and optimizer
    model = ResNetSegmentation(n_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    print('Training start')
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, project_path)

    # Load the last model for evaluation on the test data
    last_model = ResNetSegmentation(n_classes=3)
    last_model.load_state_dict(torch.load(os.path.join(project_path, f'model_checkpoints/model_checkpoint_epoch{num_epochs-1}.pth')))
    last_model.to(device)

    # Evaluate on test data
    test_loss, test_accuracy, test_IoU = validate_model(last_model, test_loader, device, criterion)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    # Log the test metrics
    log_dir = os.path.join(project_path, 'logs')
    test_log_file_path = os.path.join(log_dir, 'test_log.txt')

    with open(test_log_file_path, 'a') as test_log_file:
        test_log_file.write(f'{{"test_loss": {test_loss}, "test_accuracy": {(test_accuracy * 100):.4f}, "test_IOU": {test_IoU}}}\n')

if __name__ == '__main__':
    main()
