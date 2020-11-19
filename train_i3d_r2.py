import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="4"
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, default='rgb', help='rgb or flow')
parser.add_argument('--root', type=str, default='./hmdb51_dataset')
parser.add_argument('--split', type=str, default='1', help='1 or 2 or 3')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--save_loc', type=str, default='fine_tuned_model')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--max_steps', type=int, default=64e3)
parser.add_argument('--use_wandb',type=str, default='False')
parser.add_argument('--model_save',type=str, default='False')
parser.add_argument('--num_classes', type=int, default=8)
parser.add_argument('--max_epoch', type=int, default=200)
parser.add_argument('--load_model', type=str, default='False')
parser.add_argument('--load_loc', type=str, default='000171.pt')

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

from models.pytorch_i3d_r2 import Inception_ResNetv2

from hmdb51_dataset import HMDB51 as Dataset


def run(args):
    device = torch.device("cuda")
    if args.use_wandb == 'True':
        wandb.init(project='i3d-pytorch')
    # setup dataset
    train_split= os.path.join(args.root,'splits_{}/train'.format(args.num_classes))
    val_split= os.path.join(args.root,'splits_{}/val'.format(args.num_classes))
    train_transforms = transforms.Compose([videotransforms.RandomCrop(299),
                                           videotransforms.RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(299)])

    dataset = Dataset(train_split, args.split, args.root, args.mode, train_transforms, args.num_classes, 299)
    val_dataset = Dataset(val_split, args.split, args.root, args.mode, test_transforms, args.num_classes, 299)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)    

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    epoch = 0
    
    # setup the model
    if args.mode == 'flow':
        i3d = Inception_ResNetv2(2, args.num_classes)
    else:
        i3d = Inception_ResNetv2(3, args.num_classes)
    
    if args.load_model == 'True':
        path = os.path.join('./', args.save_loc,'class_{}'.format(args.num_classes), args.mode, args.load_loc)
        epoch = int(args.load_loc[:-3])
        
    
    #i3d.load_state_dict(torch.load('/ssd/models/000920.pt'))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    lr = args.lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])


    num_steps_per_update = 4 # accum gradient
    steps = 0
    
    # train it
    while steps < args.max_steps and epoch < args.max_epoch:
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                i3d.train()
            else:
                i3d.eval()  # Set model to evaluate mode
                
            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()
            
            # Iterate over data.
            for data in dataloaders[phase]:
                num_iter += 1
                # get the inputs
                inputs, labels = data  #input: B x Channel x T x H x W , label: B x Class x T
                inputs, labels = inputs.to(device), labels.to(device)
                t = inputs.size(2)

                per_frame_logits = i3d(inputs)
                # upsample to input size
                # per_frame_logits = F.interpolate(per_frame_logits, t, mode='linear')

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
                    if steps % 10 == 0:
                        print('phase: {}  step: {}  Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, steps, tot_loc_loss/(10*num_steps_per_update), tot_cls_loss/(10*num_steps_per_update), tot_loss/10))
                        tot_loss = tot_loc_loss = tot_cls_loss = 0.
            
                        
            if phase == 'val':
                epoch += 1
                print('='*30)
                print('phase: {}  epoch: {}  Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, epoch, tot_loc_loss/num_iter, tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter))
                print('='*30)
                if args.use_wandb == 'True':
                    log_dict = {'Loc Loss': tot_loc_loss/num_iter, 'Cls Loss': tot_cls_loss/num_iter, 'Tot Loss': (tot_loss*num_steps_per_update)/num_iter}
                    wandb.log(log_dict)
                if args.model_save == 'True':
                    torch.save(i3d.module.state_dict(), os.path.join(args.save_loc,'class_{}'.format(args.num_classes), args.mode,  str(epoch).zfill(6)+'.pt'))
                    

                               
if __name__ == '__main__':
    # need to add argparse
    run(args)
