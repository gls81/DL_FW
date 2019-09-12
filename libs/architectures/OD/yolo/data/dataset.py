# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 21:32:22 2019

@author: Gary
"""
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import functools
from PIL import Image
import math

#We need a set of definitions for usful stuuf in a set of files like load PIL etc in

class TrainingDataset(Dataset): 
    def __init__(self, config, height=224, width=224, transform=None, evaluation=False, flip=False, multi_labels=False):
        self.image_locations = config.image_locations
        self.gt_bbox

    def __getitem__(self, index):
        
        with open(self.image_locations[index], 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')
        
                                
        if self.gt_bbox and not self.lmToBbox:   
            bbox ,label = self.__getbbox__(index)
        elif self.gt_landmarks and self.lmToBbox:
            bbox ,_ = self.__getbbox__(index)
            bbox ,label = self.__getLmAsBox__(index,bbox)
        
        img, scale, size, bbox = self.transforms(img, bbox=bbox)
 
        torch.from_numpy(bbox)
        torch.from_numpy(label)
        
        return img, bbox, label, scale, index, size
        
    def __len__(self):
        return len(self.image_locations)