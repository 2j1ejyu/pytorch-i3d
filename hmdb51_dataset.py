import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import pickle
import csv
import h5py
import random
import os
import os.path

import cv2

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))


def load_rgb_frames(image_dir, vid, start, num):
  frames = []
  for i in range(start, start+num):
    img = cv2.imread(os.path.join(image_dir, 'rgb', vid, 'frame'+str(i).zfill(6)+'.jpg'))[:, :, [2, 1, 0]]
    w,h,c = img.shape
    if w < 226 or h < 226:
        d = 226.-min(w,h)
        sc = 1+d/min(w,h)
        img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
    img = (img/255.)*2 - 1
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)

def load_flow_frames(image_dir, vid, start, num):
  frames = []
  for i in range(start, start+num):
    imgx = cv2.imread(os.path.join(image_dir, 'flow/x', vid, 'frame'+str(i).zfill(6)+'.jpg'), cv2.IMREAD_GRAYSCALE)
    imgy = cv2.imread(os.path.join(image_dir, 'flow/y', vid, 'frame'+str(i).zfill(6)+'.jpg'), cv2.IMREAD_GRAYSCALE)
    
    w,h = imgx.shape
    if w < 224 or h < 224:
        d = 224.-min(w,h)
        sc = 1+d/min(w,h)
        imgx = cv2.resize(imgx,dsize=(0,0),fx=sc,fy=sc)
        imgy = cv2.resize(imgy,dsize=(0,0),fx=sc,fy=sc)
        
    imgx = (imgx/255.)*2 - 1
    imgy = (imgy/255.)*2 - 1
    img = np.asarray([imgx, imgy]).transpose([1,2,0])    # (2,w,h) -> (w,h,2)
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)


def make_dataset(split_folder, split, root, mode, num_classes=51):
    split_data = os.path.join(split_folder,mode, 'split'+split, 'data.pickle')
    dataset = []
    with open(split_data, 'rb') as f:
        data = pickle.load(f)
    
    for vid in data.keys():
        num_frames = data[vid]['num_frame']
            
        if num_frames < 66:
            continue

        label = np.zeros((num_classes,num_frames), np.float32)

        ann = data[vid]['target']
        label[ann] = 1 # binary classification
        dataset.append((vid, label, num_frames))
    
    return dataset


def make_dataset_both(split_folder, split, root, num_classes=51):
    split_data_rgb = os.path.join(split_folder,'rgb', 'split'+split, 'data.pickle')
    split_data_flow = os.path.join(split_folder,'flow', 'split'+split, 'data.pickle')
    dataset = []
    with open(split_data_rgb, 'rb') as f:
        data_rgb = pickle.load(f)
    with open(split_data_flow, 'rb') as f:
        data_flow = pickle.load(f)
    
    for vid in data_rgb.keys():
        nf_flow = data_flow[vid]['num_frame']
        nf_rgb = data_rgb[vid]['num_frame']
        min_num_frames = min(nf_flow, nf_rgb)
        
        if min_num_frames < 66:
            continue
    
        label = np.zeros((num_classes,min_num_frames), np.float32)

        ann = data_rgb[vid]['target']
        label[ann] = 1 # binary classification
        dataset.append((vid, label, min_num_frames))
    
    return dataset


class HMDB51(data_utl.Dataset):

    def __init__(self, split_folder, split, root, mode, transforms=None, num_classes=51):
        self.data = make_dataset(split_folder, split, root, mode, num_classes)
        self.transforms = transforms
        self.mode = mode
        self.root = root

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, nf = self.data[index]
        start_f = random.randint(1,nf-65)

        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, vid, start_f, 64)
        else :
            imgs = load_flow_frames(self.root, vid, start_f, 64)

        label = label[:, start_f:start_f+64]

        imgs = self.transforms(imgs)

        return video_to_tensor(imgs), torch.from_numpy(label)

    def __len__(self):
        return len(self.data)


class HMDB51_both(data_utl.Dataset):

    def __init__(self, split_folder, split, root, transforms=None, num_classes=51):
        self.data = make_dataset_both(split_folder, split, root, num_classes)
        self.transforms = transforms
        self.root = root

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, nf = self.data[index]
        start_f = random.randint(1,nf-65)

        imgs_rgb = load_rgb_frames(self.root, vid, start_f, 64)
        imgs_flow = load_flow_frames(self.root, vid, start_f, 64)

        label = label[:, start_f:start_f+64]

        imgs_rgb = self.transforms(imgs_rgb)
        imgs_flow = self.transforms(imgs_flow)

        return video_to_tensor(imgs_rgb), video_to_tensor(imgs_flow), torch.from_numpy(label)

    def __len__(self):
        return len(self.data)
