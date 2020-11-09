import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="2"
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='./hmdb51_dataset')
parser.add_argument('--split', type=str, default='1', help='1 or 2 or 3')
parser.add_argument('--use_wandb',type=str, default='False')
parser.add_argument('--num_classes', type=int, default=8)

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

from pytorch_i3d import I3D

from hmdb51_dataset import HMDB51_both as Dataset


def main(args):
    ################################ setting ######################################
    test_path = os.path.join(args.root,'splits_8/val')
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    test_dataset = Dataset(test_path, args.split, args.root, test_transforms, num_classes=args.num_classes)
    test_loader= torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, pin_memory=True)
    
    ################################ attack ######################################
    path = "./fine_tuned_model"
    epsilons = [0,0.1,0.5,1,2,4,8]
    #models = os.listdir(path)
    #models.sort()
    models = ['fine_tuned_model008240.pt']
    device = torch.device("cuda")
    for eps in epsilons:
        if args.use_wandb == 'True':
            wb = wandb.init(project='i3d-pytorch', reinit=True)
            with wb:    
                for model_ in models:
                    if '.pt' not in model_:
                        continue
                    #if int(model_[-9:-3])%200 != 0:
                    #    continue
                    model = I3D('', '', num_classes=args.num_classes, pretrain = False)
                    model.load_state_dict(torch.load(os.path.join(path,model_)))
                    test(model, device, test_loader, eps, model_)
        else:
            for model_ in models:
                if '.pt' not in model_:
                    continue
                #if int(model_[-9:-3])%200 != 0:
                #    continue
                model = I3D('', '', num_classes=args.num_classes, pretrain = False)
                model.load_state_dict(torch.load(os.path.join(path,model_)))
                test(model, device, test_loader, eps, model_)
   
            
            
def test( model, device, test_loader, epsilon, model_name):
    
    correct_adv = 0
    correct = 0
    wrong_count = 0
    adv_examples = []
    model.cuda()
    model.eval()

    for data in test_loader:
        inputs_rgb, inputs_flow, target = data
        inputs_rgb, inputs_flow, target = inputs_rgb.to(device), inputs_flow.to(device), target.to(device)
        
        # attack only rgb
        inputs_rgb.requires_grad = True
        t = inputs_rgb.size(2)
    
        per_frame_logits = model(inputs_rgb, inputs_flow)
        # upsample to input size
        per_frame_logits = F.interpolate(per_frame_logits, t, mode='linear')
        
        init_pred = per_frame_logits.max(2)[0].max(1)[1]
        target_ = target.max(2)[0].max(1)[1]
        
        # if wrong, continue
        if init_pred.item() == target_.item():
            correct += 1

        loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, target)

        cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(target, dim=2)[0])

        loss = (0.5*loc_loss + 0.5*cls_loss)

        model.zero_grad()

        loss.backward()

        data_grad = inputs_rgb.grad.data

        perturbed_rgb = fgsm_attack_f(inputs_rgb, epsilon, data_grad, device, 0)

        per_frame_logits = model(perturbed_rgb, inputs_flow)

        final_pred = per_frame_logits.max(2)[0].max(1)[1]
        
        if final_pred.item() == target_.item():
            correct_adv += 1
            
        elif init_pred.item() == target_.item():
            # for visualization
            if wrong_count < 5:
                adv_ex = perturbed_rgb.squeeze().detach().cpu().numpy()
                ex = inputs_rgb.squeeze().detach().cpu().numpy()
                adv_examples.append( (ex, adv_ex) )
                wrong_count += 1

    accuracy = correct/float(len(test_loader))
    accuracy_adv = correct_adv/float(len(test_loader))
    
    print("Epsilon: {}   Accuracy = {}   Accuracy_adv = {}".format(epsilon, accuracy, accuracy_adv))
    
    if args.use_wandb == 'True':
        log_dict = {'Accuracy': accuracy, 'Accuracy_adv': accuracy_adv, 'epsilon': epsilon}
        wandb.log(log_dict)
    
    save_data = {'epsilon':epsilon, 'accuracy_org': accuracy, 'accuracy_adv': accuracy_adv, 'examples': adv_examples}
    path_name = os.path.join('./adv_data','first_frame',str(epsilon)+'_'+model_name[-9:-3]+'.pickle')
    with open(path_name,'wb') as fw:
        pickle.dump(save_data,fw)
            
            
def fgsm_attack_f(image, epsilon, data_grad, device, index):
    perturbed_image = torch.zeros(image.shape, device=device)
    perturbed_image[:,:,index,:,:] = epsilon*(data_grad[:,:,index,:,:].sign())
    perturbed_image += image
    perturbed_image = torch.clamp(perturbed_image, -1, 1)

    return perturbed_image

def fgsm_attack_all(image, epsilon, data_grad, device):
    
    perturbed_image = image + epsilon*(data_grad.sign())
    perturbed_image = torch.clamp(perturbed_image, -1, 1)

    return perturbed_image

    

if __name__ == '__main__':
    # need to add argparse
    main(args)
