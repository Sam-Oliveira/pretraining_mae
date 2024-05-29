# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional
import matplotlib.pyplot as plt
import torch
from PIL import Image
from timm.data import Mixup
from timm.utils import accuracy
from torchvision import transforms
import util.misc as misc
import util.lr_sched as lr_sched
import numpy as np

# This file has the main training and evaluatin "boilerplate" code. Responsible for training over one epoch,
# and then evaluate function is called at the end of each epoch, on the validation set.
# Outputs some example segmentation masks on both the training and validation sets.

# Function to run 
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                outputs = model(samples)

                # OUTPUT[1] IS [BATCH_SIZE,1,196,768], NEED TO RESHAPE TO [BATCH_SIZE,3,224,224]
                outputs=torch.reshape(outputs[1],(samples.shape[0],3,224,224))
                
                # Simply selecting the value from the targets
                targets = targets[:,0,:,:].long()

                loss = criterion(outputs, targets)
        else:
            outputs = model(samples)

            # OUTPUT[1] IS [BATCH_SIZE,1,196,768], NEED TO RESHAPE TO [BATCH_SIZE,3,224,224]
            outputs=torch.reshape(outputs[1],(samples.shape[0],3,224,224))
            
            # Simply selecting the value from the targets
            targets = targets[:,0,:,:].long()

            loss = criterion(outputs, targets)
        

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # Code to plot example segmentation results on training set    
    if epoch%5==0:
        num_images=5
        plt.figure()
        fig, axs = plt.subplots(num_images, 2, figsize=(15, 10))
        for i in range(num_images):
            image = targets[i]
            image=image[None,:,:].to(torch.float).to('cpu')

            prediction=outputs[i]
            # get argmax in first ("class probs") dimension!
            _,prediction=torch.max(prediction,dim=0,keepdim=True)
            prediction = prediction.to(torch.float).to('cpu')

            
            axs[i, 0].imshow(image.reshape(224,224,1)/2.0, cmap='gray')
            axs[i, 0].set_title('True Segmentation')

            axs[i, 1].imshow(prediction.reshape(224,224,1)/2.0, cmap='gray')
            axs[i, 1].set_title('Prediction')
        plt.tight_layout()
        plt.savefig('{output}/Epoch_{f}.png'.format(output=args.output_dir, f=epoch))
        plt.show()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# Function to calculate loss/iou/accuracy on validation dataset
@torch.no_grad()
def evaluate(data_loader, model, device, epoch, args):
    # Calculates mean over pixels
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
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():

                output = model(images)

                # OUTPUT[1] IS [BATCH_SIZE,1,196,768], NEED TO RESHAPE TO [BATCH_SIZE,3,224,224]
                output=torch.reshape(output[1],(images.shape[0],3,224,224))
                
                # This is needed, just simply selecting the value from the targets
                target = target[:,0,:,:].long()

                loss = criterion(output, target)
        else:
            output = model(images)

            # OUTPUT[1] IS [BATCH_SIZE,1,196,768], NEED TO RESHAPE TO [BATCH_SIZE,3,224,224]
            output=torch.reshape(output[1],(images.shape[0],3,224,224))
            
            # This is needed, just simply selecting the value from the targets
            target = target[:,0,:,:].long()

            loss = criterion(output, target)


    # SOME CODE TO PLOT SOME validation set RESULTS    
    if epoch%5==0:
        num_images=5
        plt.figure()
        fig, axs = plt.subplots(num_images, 2, figsize=(15, 10))
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
        plt.savefig('{output}/segmentation/Epoch_{f}.png'.format(output=args.output_dir, f=epoch))
        plt.show()

    IOU,accuracy=metrics(output,target)

    batch_size = images.shape[0]
    metric_logger.update(loss=loss.item())
    metric_logger.meters['IOU'].update(IOU.item(), n=batch_size)
    metric_logger.meters['accuracy'].update(accuracy.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* IOU {iou.global_avg:.3f}  Accuracy {acc.global_avg:.1f}% loss {losses.global_avg:.3f} '
          .format(iou=metric_logger.IOU, acc=metric_logger.accuracy,losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def metrics(pred, gt):

    pred=torch.argmax(pred,dim=1,keepdim=True)
    
    # Compute accuracy
    gt_accuracy=gt[:,None,:,:]
    truth_tensor=torch.eq(pred,gt_accuracy)
    accuracy=100*torch.count_nonzero(truth_tensor.long())/torch.numel(pred)



    # Compute IOU

    pred = torch.cat([ (pred == 0)], dim=1)
    #Add channel dimension to targets that was removed when computing CE loss in line 136
    gt=gt[:,None,:,:]

    gt = torch.cat([ (gt == 0)], dim=1)


    intersection = torch.logical_and(gt,pred)
    union = torch.logical_or(gt,pred)

    # Compute the sum over all the dimensions except for the batch dimension.
    iou = (intersection.sum(dim=(1, 2, 3)) + 0.001) / (union.sum(dim=(1, 2, 3)) + 0.001)
    
    # Compute the mean over the batch dimension for the iou.
    return iou.mean(),accuracy