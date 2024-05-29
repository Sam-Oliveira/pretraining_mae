import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
import time

import models_mae_ft
from util.lars import LARS

from util.datasets import OxfordPetsDataset
from util.pos_embed import interpolate_pos_embed
from util.misc import load_model
import util.misc as misc
from util.crop import RandomResizedCrop

from matplotlib import pyplot as plt

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning', add_help = False)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                    help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='ftmae_vit_base_patch16_dec512d8b', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.01, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # finetuning parameters
    parser.add_argument('--finetune', default='./output_dir_judy/checkpoint-149.pth',
                        help='finetune from checkpoint')


    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    return parser

def main(args):
    root_dir = '/cs/student/projects3/COMP0197/grp3/adl_groupwork/OxfordPet'
    dataset_train = OxfordPetsDataset(root_dir=root_dir, split='train')
    dataset_val = OxfordPetsDataset(root_dir=root_dir, split='val')

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    model = models_mae_ft.__dict__[args.model]()
    checkpoint_model = torch.load('judy_ft_model.pth', map_location='cpu')
    msg = model.load_state_dict(checkpoint_model, strict=False)

    for i, data in enumerate(data_loader_train, 0):
        if i == 1:
            break
        images, masks = data[0], data[1]
        output = model(images, masks)
        output = model.unpatchify(output)
        print(output[0])

        # plt.imshow('ft/mask.png', masks[0], cmap= 'gray')
        # plt.imsave('ft/output.png', output[0], cmap='gray')
    


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)