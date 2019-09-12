# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 15:37:21 2019

@author: Gary
"""
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import functools
from PIL import Image
import math

class videoData(Dataset):
    def __init__(self, info, crop_method=None, spatial_transform=None,temporal_transform=None,sample_duration=5, external_bbox=None):
        
        frames = info.video_frames_data
        images = info.image_locations
        self.data = self.get_dataset(frames,images)
        if external_bbox is not None:
            self.bbox_crop = self.get_bbox_Crop(frames,external_bbox)
        self.crop_method = crop_method  
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = self.get_default_video_loader()
        self.labels = np.asarray(info.palsy_level_data, dtype=None, order=None)
        self.label = np.asarray(self.labels[:,2], dtype='int32', order=None)
        
    def __getitem__(self, index):
        paths = self.data[index]

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(len(paths))
        clip = self.loader(paths, frame_indices)
        if self.crop_method is not None:
            clip = [self.crop_method(img, self.bbox_crop[index]) for img in clip]
            
#        for img in clip:
#            img.show()
            
        if self.spatial_transform is not None:
            clip = [self.spatial_transform(img) for img in clip]
        
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.label[index]
        
        return clip, target
        
    def __len__(self):
        return len(self.data)
    
    def get_dataset(self, frames,images):
        dataset = []
        
        frames_index = np.unique(frames)
        
       
        for frame in frames_index:
            index =  frames == frame
            filtered_list = [i for (i, v) in zip(images, index) if v]
            dataset.append(filtered_list)
        
        return dataset
    
    
    
    def get_bbox_Crop(self, frames,box):
        dataset = []
        crop = []
        
        frames_index = np.unique(frames)
        
       
        for frame in frames_index:
            index =  frames == frame
            filtered_list = [i for (i, v) in zip(box, index) if v]
            dataset.append(filtered_list)
            
        for boxes in dataset:
            x1 = math.inf  
            x2 = -math.inf
            y1 = math.inf
            y2 = -math.inf
            for box in boxes:
                box = np.reshape(box, (-1, 2))
                tmp_x1 = int(min(box[:,0]))
                if  tmp_x1 < x1:
                    x1 = tmp_x1
                tmp_x2 = int(max(box[:,0]))
                if  tmp_x2 > x2:
                    x2 = tmp_x2
                tmp_y1 = int(min(box[:,1]))
                if  tmp_y1 < y1:
                    y1 = tmp_y1
                tmp_y2  = int(max(box[:,1]))    
                if  tmp_y2 > y2:
                    y2 = tmp_y2
            crop.append([x1,x2,y1,y2])
            
        return crop
    
    def pil_loader(self, path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')


    def accimage_loader(path):
        try:
            return accimage.Image(path)
        except IOError:
            # Potentially a decoding problem, fall back to PIL.Image
            return pil_loader(path)


    def get_default_image_loader(self):
        from torchvision import get_image_backend
        if get_image_backend() == 'accimage':
            import accimage
            return accimage_loader
        else:
            return self.pil_loader


    def video_loader(video_dir_path, frame_indices, image_loader):
        video = []
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
            if os.path.exists(image_path):
                video.append(image_loader(image_path))
            else:
                return video

        return video


    def palsy_video_loader(self, image_paths, frame_indices, image_loader):
        video = []
        for i in frame_indices:
            image_path = image_paths[i]
            if os.path.exists(image_path):
                video.append(image_loader(image_path))
            else:
                return video

        return video

    def get_default_video_loader(self):
        image_loader = self.get_default_image_loader()
        return functools.partial(self.palsy_video_loader, image_loader=image_loader)