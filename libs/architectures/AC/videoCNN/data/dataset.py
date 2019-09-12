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
from libs.visualise import visualizeResults

class videoData(Dataset):
    def __init__(self, info, labels, crop_method=None, spatial_transform=None,temporal_transform=None, external_bbox=None):
        
        frames = info.labels['VIDEO']['INDEX']
        self.data = info.image_locations

        self.label = []
        self.label_names = []
        self.label_legend = []
        self.label_range = []
        
        if info.protocols:
            self.protocol_indexs = info.protocol_dic
        
        for label in info.labels:
            if label in labels:
                self.label_names.append(label)
                self.label_legend.append(info.labels_legend[label]['LEGEND']) 
                self.label_range.append(info.labels_legend[label]['RANGE']) 
                self.label.append(info.labels[label]['LABEL'])
        
        if len(self.label) == 1:
            self.labels = self.label[0]      
        else:
            self.labels = np.transpose(np.stack(self.label))
        
        if external_bbox is not None:
            self.bbox_crop = self.get_bbox_Crop(frames,external_bbox)
        self.crop_method = crop_method  
        self.spatial_transform = spatial_transform        
        self.temporal_transform = temporal_transform
        self.loader = self.get_default_video_loader()
        
        
    def __filterIndex__(self, index, protocol, purpose):
        """
            Takes a index valuex the size of the data set and filters.
            Useage incldes taking validation protocols to remove test data from training data etc.
    
        """    
        bool_filter = self.protocol_indexs[protocol][purpose][index]
        data = np.asarray(self.data)
        data = data[bool_filter]
        self.data = data.tolist()
        labels = np.asarray(self.labels)
        labels = labels[bool_filter]
        self.labels = labels.tolist()
        bbox_crop = np.asarray(self.bbox_crop)
        bbox_crop = bbox_crop[bool_filter]
        self.bbox_crop = bbox_crop.tolist()
        
        return 
        
    def __getitem__(self, index):
        paths = self.data[index]
        #print(index)
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(len(paths))
        clip = self.loader(paths, frame_indices)
        #if self.crop_method is not None:
            #clip = [self.crop_method(img, self.bbox_crop[index]) for img in clip]
        clip = self.crop_method(clip, self.bbox_crop[index])
        clip = self.spatial_transform(clip)
#        for i, img in enumerate(clip):
#            vs = visualizeResults()
#            save_name = 'D:/tmp/' + str(index) + '_' + str(i) + '.jpg'
#            vs.save_image(img,save_name)
                 
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        target = self.labels[index]
        
        return clip, target
        
    def __len__(self):
        return len(self.data)
      
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
                box = np.reshape(box[0], (-1, 2))
                #box = box[43:44,:]
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