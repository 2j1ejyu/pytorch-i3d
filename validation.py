import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="3"
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, default='both', help='rgb or flow or both')
parser.add_argument('--root', type=str, default='./hmdb51_dataset')
parser.add_argument('--split', type=str, default='1', help='1 or 2 or 3')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--use_wandb',type=str, default='True')
parser.add_argument('--num_classes', type=int, default=8)
parser.add_argument('--max_epoch', type=int, default=200)
parser.add_argument('--load_loc', type=str, default='fine_tuned_model')

args = parser.parse_args()


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms
import wandb
import pdb
import pickle

import numpy as np

from pytorch_i3d import InceptionI3d

from hmdb51_dataset import HMDB51_both as Dataset


def main(args):
    device = torch.device("cuda")
    
    test_path = os.path.join(args.root,'splits_{}/val'.format(args.num_classes))
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    test_dataset = Dataset(test_path, args.split, args.root, test_transforms, args.num_classes)
    test_loader= torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    path_flow = os.path.join(args.load_loc,'class_{}'.format(args.num_classes), 'flow')
    models_flow = os.listdir(path_flow)
    models_flow.sort()
    
    path_rgb = os.path.join(args.load_loc,'class_{}'.format(args.num_classes), 'rgb')
    models_rgb = os.listdir(path_rgb)
    models_rgb.sort()
    
    if args.use_wandb == 'True':
        wandb.init(project='i3d-pytorch')
        
    for epoch in range(1,args.max_epoch+1): 
        model_flow = InceptionI3d(args.num_classes, in_channels=2)
        model_flow.load_state_dict(torch.load(os.path.join(path_flow, models_flow[epoch])))
        
        model_rgb = InceptionI3d(args.num_classes, in_channels=3)
        model_rgb.load_state_dict(torch.load(os.path.join(path_rgb, models_rgb[epoch])))
        
        test(model_flow, model_rgb, device, test_loader, epoch)

            
            
def test(model_flow, model_rgb, device, test_loader, epoch):

    correct = 0
    model_flow.cuda()
    model_rgb.cuda()
    model_flow.eval()
    model_rgb.eval()

    for data in test_loader:
        inputs_rgb, inputs_flow, target = data
        inputs_rgb, inputs_flow, target = inputs_rgb.to(device), inputs_flow.to(device), target.to(device)
        
        t = inputs_rgb.size(2)
    
        logits_flow = model_flow(inputs_flow)
        logits_rgb = model_rgb(inputs_rgb)
        logits = (logits_rgb + logits_flow)/2
        logits = F.interpolate(logits, t, mode='linear')
        
        pred = logits.max(2)[0].max(1)[1]
        target = target.max(2)[0].max(1)[1]
        
        correct += pred.eq(target).sum().item()


    accuracy = correct/float(len(test_loader))
    
    print("epoch: {}   accuracy: {}".format(epoch, accuracy))
    
    if args.use_wandb == 'True':
        log_dict = {'Accuracy': accuracy}
        wandb.log(log_dict)

if __name__ == '__main__':
    # need to add argparse
    main(args)
