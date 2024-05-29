# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import random # added this to call the seed
import PIL

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

from PIL import Image

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN # Judy: need to change these (not using imagenet)
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class OxfordPetsDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, seed=None):
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


        self.root_dir = root_dir
        self.split = split
        # Combine the default resize transformation with the conversion to tensor
        # Users can pass additional transformations which will be applied after resizing and converting to tensor
        default_transforms = [transforms.Resize((224, 224)), transforms.ToTensor()]
        if transform is not None:
            default_transforms.append(transform)
        self.transform = transforms.Compose(default_transforms)
        
        # Split into training, validation and test data set
        all_images = [img for img in os.listdir(os.path.join(root_dir, 'images')) if img.endswith('.jpg')]
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

        self.images = [os.path.join(root_dir, 'images', fname) for fname in image_filenames]
        self.masks = [os.path.join(root_dir, 'annotations/trimaps', fname.replace('.jpg', '.png')) for fname in image_filenames]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform is not None:
            state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(state)
            mask = self.transform(mask)

        mask = self.preprocess_segmentation(mask)

        return image, mask

    def preprocess_segmentation(self, mask):
        thresholds = [0.0, 0.004, 0.008, 0.2]

        processed_mask = torch.zeros_like(mask)

        for i in range(len(thresholds) - 1):
            lower_bound = thresholds[i]
            upper_bound = thresholds[i + 1]
            processed_mask[(mask >= lower_bound) & (mask < upper_bound)] = i

        return processed_mask

    def set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed) # comment out if
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False