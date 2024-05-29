
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import json

from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import timm

assert timm.__version__ == "0.3.2" # version check

import util.misc as misc
from util.pos_embed import interpolate_pos_embed

import models_mae_finetune as models_mae
from engine_finetune import metrics


#from pet_dataset import OxfordPetsDataset
from util.datasets import OxfordPetsDataset


def get_args_parser():
    parser = argparse.ArgumentParser('MAE linear probing for image classification', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    parser.add_argument('--model', default='checkpoint-99.pth')
    parser.add_argument('--model_base', default='mae_vit_base_patch16_dec512d8b')
    parser.add_argument('--experiment', default="finetuning_supervised_mae/")
    parser.add_argument('--input_dir', default='output_dir_supervised/')

    parser.add_argument('--output_dir', default='output_dir_finetunedtest/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    return parser


def evaluate(data_loader, model, device,epoch):
    plotted = False

    with torch.no_grad():
        # CHANGED
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        metric_logger = misc.MetricLogger(delimiter="  ")
        header = 'Test:'

        # switch to evaluation mode
        model.eval()

        for batch in metric_logger.log_every(data_loader, 10, header):
            images = batch[0]
            target = batch[-1]
            
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():

                output = model(images)

                # OUTPUT[0] SEEMS TO BE A SCALAR FOR SOME REASON? MAYBE LOSS??!
                # OUTPUT[1] IS [BATCH_SIZE,1,196,768], NEED TO RESHAPE TO [BATCH_SIZE,3,224,224]
                output=torch.reshape(output[1],(images.shape[0],3,224,224))
                
                # This is needed, just simply selecting the value from the targets
                target = target[:,0,:,:].long()

                loss = criterion(output, target)

                # SOME CODE TO PLOT SOME RESULTS    
                if epoch%5==0 and not plotted:
                    print("plotting")
                    num_images=5
                    plt.figure()
                    fig, axs = plt.subplots(num_images, 2, figsize=(10, 30))
                    for i in range(num_images):
                        image = target[i]
                        image=image[None,:,:].to(torch.float).to('cpu')

                        prediction=output[i]
                        # get argmax in first ("class probs") dimension!
                        _,prediction=torch.max(prediction,dim=0,keepdim=True)
                        prediction = prediction.to(torch.float).to('cpu')

                        
                        axs[i, 0].imshow(image.reshape(224,224,1)/2.0, cmap='gray')
                        axs[i, 0].set_title('True Segmentation')

                        
                        axs[i, 1].imshow(prediction.reshape(224,224,1)/2.0, cmap='gray')
                        axs[i, 1].set_title('Prediction')
                    plt.tight_layout()
                    plt.savefig(args.output_dir + args.experiment + '/test_set.png'.format(f=epoch))
                    plotted = True


        IOU, acc = metrics(output,target)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['IOU'].update(IOU.item(), n=batch_size)
        metric_logger.meters['acc'].update(acc.item(), n=batch_size)

        metric_logger.synchronize_between_processes()
        print('* IOU {top1.global_avg:.3f}  loss {losses.global_avg:.3f}   accuracy {accuracy.global_avg:.3f}'
            .format(top1=metric_logger.IOU,  losses=metric_logger.loss, accuracy=metric_logger.acc))

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

    root_dir = '../OxfordPet'
    dataset_test = OxfordPetsDataset(root_dir=root_dir, split='test', seed=42)

    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, 
        sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    model = models_mae.__dict__[args.model_base]()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    # load model checkpoint into class instance
    #model_path = args.input_dir + args.experiment + args.model
    model_path = args.input_dir + args.model
    checkpoint = torch.load(model_path, map_location='cpu')
    checkpoint_model = checkpoint['model']

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # load model
    model.load_state_dict(checkpoint_model, strict=False)
    model.to(device)

    test_stats = evaluate(data_loader_test, model, device, 0)
    test_stats
    print(f"Metrics of the network on the {len(dataset_test)} test images:")
    print(f"Loss: {test_stats['loss']:.3f}\tIOU: {test_stats['IOU']:.3f}\tAccuracy: {test_stats['acc']:.3f}")

    with open(os.path.join(args.output_dir + args.experiment, f"log.txt"), 
                            mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(test_stats) + "\n")
    exit(0)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir + args.experiment).mkdir(parents=True, exist_ok=True)
    main(args)
