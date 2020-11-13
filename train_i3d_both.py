import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="2"
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='./hmdb51_dataset')
parser.add_argument('--split', type=str, default='1', help='1 or 2 or 3')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--save_loc', type=str, default='fine_tuned_model')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--max_steps', type=int, default=64e3)
parser.add_argument('--use_wandb',type=str, default='False')
parser.add_argument('--model_save',type=str, default='True')
parser.add_argument('--num_classes', type=int, default=8)

args = parser.parse_args()


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, transforms
import videotransforms
import wandb
import pdb


import numpy as np

from pytorch_i3d import I3D

from hmdb51_dataset import HMDB51_both as Dataset


def run(args):
    # setup dataset
    device = torch.device("cuda")
    if args.use_wandb == 'True':
        wandb.init(project='i3d-pytorch')
    train_split= os.path.join(args.root,'splits_8/train')
    val_split= os.path.join(args.root,'splits_8/val')
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset= Dataset(train_split, args.split, args.root, train_transforms, num_classes=args.num_classes)
    val_dataset = Dataset(val_split, args.split, args.root, test_transforms, num_classes=args.num_classes)

    dataloader= torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_dataloader= torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)     

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    
    # setup the model
    i3d = I3D('models/rgb_imagenet.pt', 'models/flow_imagenet.pt', num_classes=args.num_classes)

    #i3d.load_state_dict(torch.load('/ssd/models/000920.pt'))
    i3d.cuda()

    i3d= nn.DataParallel(i3d)

    lr = args.lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])


    num_steps_per_update = 4 # accum gradient
    steps = 0
    # train it
    while steps < args.max_steps:#for epoch in range(num_epochs):
        print('Step {}/{}'.format(steps, args.max_steps))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                i3d.train()
            else:
                i3d.eval()   # Set model to evaluate mode
            tot_loss = 0.0 
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()
            
            # Iterate over data.
            for data in dataloaders[phase]:
                num_iter += 1
                # get the inputs
                inputs_rgb, inputs_flow, labels = data
                inputs_rgb, inputs_flow, labels = inputs_rgb.to(device), inputs_flow.to(device), target.to(labels)
                t = inputs_rgb.size(2)

                per_frame_logits = i3d(inputs_rgb, inputs_flow)
                # upsample to input size
                per_frame_logits = F.interpolate(per_frame_logits, t, mode='linear')

                # compute localization loss
                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                tot_loc_loss += loc_loss.data

                # compute classification loss (with max-pooling along time B x C x T)
                cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
                tot_cls_loss += cls_loss.data

                loss = (0.5*loc_loss + 0.5*cls_loss)/num_steps_per_update
                tot_loss += loss.data
                loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_sched.step()
                    if args.model_save == 'True':
                        if steps % 20 == 0:
                            print('(model saving) {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/(10*num_steps_per_update), tot_cls_loss/(10*num_steps_per_update), tot_loss/10))
                            # save model
                            torch.save(i3d.module.state_dict(), args.save_loc + str(steps).zfill(6)+'.pt')
                            tot_loss = tot_loc_loss = tot_cls_loss = 0.
         
                
            if phase == 'val':
                print('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/num_iter, tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter))
                if args.use_wandb == 'True':
                    log_dict = {'Loc Loss': tot_loc_loss/num_iter, 'Cls Loss': tot_cls_loss/num_iter, 'Tot Loss': (tot_loss*num_steps_per_update)/num_iter}
                    wandb.log(log_dict)
                
    


if __name__ == '__main__':
    # need to add argparse
    run(args)
