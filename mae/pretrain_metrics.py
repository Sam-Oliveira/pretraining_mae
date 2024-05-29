import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.pos_embed import interpolate_pos_embed

import models_mae

from engine_pretrain import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('MAE model checkpoints', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    parser.add_argument('--model', default='checkpoint-140.pth')
    parser.add_argument('--model_base', default='mae_vit_base_patch16_dec512d8b')
    parser.add_argument('--experiment', default="")
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--data_path', default='./coco_subset/val/')
    parser.add_argument('--which_subset', default='animals')
    parser.add_argument('--dataset_type', default = 'val/')

    parser.add_argument('--input_dir', default = 'output_dir_judyAnimals/')
    parser.add_argument('--output_dir', default='output_dir_pretrain/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    return parser


def evaluate(data_loader, model, device,epoch):

    with torch.no_grad():
        metric_logger = misc.MetricLogger(delimiter="  ")
        header = 'Val:'

        for batch in metric_logger.log_every(data_loader, 10, header):
            images = batch[0]
            target = batch[-1]
            
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            loss, pred, mask = model(images)


        # batch_size = args.batch_size
        metric_logger.update(loss=loss.item())
        # metric_logger.meters['IOU'].update(IOU.item(), n=batch_size)

        metric_logger.synchronize_between_processes()
        # print('* IOU {top1.global_avg:.3f}  loss {losses.global_avg:.3f}'
        #     .format(top1=metric_logger.IOU,  losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation - do we have to transform the validation set??
    transform_test = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_test = datasets.ImageFolder(os.path.join(args.data_path, args.which_subset), transform=transform_test)

    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    with open(os.path.join(args.output_dir + args.dataset_type, f"log_{args.which_subset}.txt"), mode="w", encoding="utf-8") as f:
        f.write("")
    for epoch in range(200):
        model_name = f"checkpoint-{epoch}.pth"

        # load model checkpoint into class instance
        model_path = args.input_dir + model_name

        # if checkpoint found
        if os.path.isfile(model_path):
            model = models_mae.__dict__[args.model_base]() 
            
            print(f"Found valid model {model_name}")
            checkpoint = torch.load(model_path, map_location='cpu')
            checkpoint_model = checkpoint['model']

            # interpolate position embedding
            interpolate_pos_embed(model, checkpoint_model)

            # load model
            model.load_state_dict(checkpoint_model, strict=False)
            model.to(device)

            test_stats = evaluate(data_loader_test, model, device, 0)
            # print(f"Metrics of the network on the {len(dataset_test)} test images:")
            print(f"Loss: {test_stats['loss']:.3f}")

            log_stats = {**{f'val_{k}': v for k, v in test_stats.items()},
                                'epoch': epoch}
            if args.output_dir and misc.is_main_process():
                with open(os.path.join(args.output_dir + args.dataset_type, f"log_{args.which_subset}.txt"), 
                            mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

    exit(0)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        path = args.output_dir + args.dataset_type
        Path(path).mkdir(parents=True, exist_ok=True)
    main(args)
