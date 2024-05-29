# Import torch
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision import transforms
from torch.utils.data import Dataset

# Import fundamentals
import os
import numpy as np
import random
from PIL import Image

def create_image_transform(custom_transform=None):
    """
    Create a composite transform consisting of standard transformations and an optional custom transformation.
    
    Args:
    custom_transform (callable, optional): An additional transform to be applied after the standard transformations.
    
    Returns:
    torchvision.transforms.Compose: A composite transform.
    """
    default_transforms = [
        transforms.Resize((224, 224)),  # Resize all images to 224x224
        transforms.ToTensor()           # Convert images to tensor format
    ]
    
    # If there is an additional transform provided by the user, append it to the list
    if custom_transform is not None:
        default_transforms.append(custom_transform)
    
    # Compose all the transforms into a single pipeline
    return transforms.Compose(default_transforms)

# def debug_transform(x):
#     print("Unique values post-transform:", x.unique())
#     return x

def create_mask_transform():
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.NEAREST),
        transforms.ToTensor(),
        lambda x: torch.clamp(x * 255 - 1, min=0, max=2)#debug_transform  # Add this to check the distribution of values
    ])

class OxfordPetsDataset(Dataset):
    def __init__(self, root_dir, split, image_transform=None, mask_transform=None, seed=None):
        """
        Initializes the dataset.

        Args:
        root_dir: Directory with all the dataset images and masks.
        split: One of 'train', 'val', or 'test' to select the data split.
        transform: Optional custom transform to be applied on a sample.
        """

        # Set seed
        if seed is not None:
            self.set_seed(seed)

        # Initialise variables
        self.root_dir = root_dir
        self.split = split
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        # Set up paths
        images_path = os.path.join(root_dir, 'OxfordPet/images')
        masks_path = os.path.join(root_dir, 'OxfordPet/annotations/trimaps')
        
        # Split into training, validation and test data set
        all_images = [img for img in os.listdir(images_path) if img.endswith('.jpg')]
        random.shuffle(all_images)
        num_images = len(all_images)
        train_end = int(0.7 * num_images)
        val_end = train_end + int(0.15 * num_images)
        
        if split == 'train':
            image_filenames = all_images[:train_end]
        elif split == 'val':
            image_filenames = all_images[train_end:val_end]
        elif split == 'test':
            image_filenames = all_images[val_end:]
        else:
            raise ValueError("Unknown split: {}. Expected 'train', 'val', or 'test'.".format(split))

        # Get images and masks for respective data set
        self.images = [os.path.join(images_path, fname) for fname in image_filenames]
        self.masks = [os.path.join(masks_path, fname.replace('.jpg', '.png')) for fname in image_filenames]

    def __len__(self):
        return len(self.images)
    
    def set_seed(self, seed):
        random.seed(seed)
        #torch.manual_seed(seed) # comment out if
        #torch.cuda.manual_seed(seed)
        #torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False

    def __getitem__(self, idx):
        # Get image and mask paths
        image_path = self.images[idx]

        # Open image as RGB and mask
        mask_path = self.masks[idx]
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Apply transformations to the image
        if self.image_transform:
            image = self.image_transform(image)
        
        # Apply mask_transform to the mask
        if self.mask_transform:
            mask = self.mask_transform(mask)


        #####################
        # Apply the same transformation to both image and mask
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        
        if random.random() > 0.5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)

        return image, mask
