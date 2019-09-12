# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 11:31:12 2018

@author: Gary
"""

from torch.utils.data.dataset import Dataset
#from torchvision import transforms
import pandas as pd 
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from skimage import io
from libs.dataset import utils
import random 
import cv2
import functools
from scipy import signal
import os

#A base class 
class CustomDataset(Dataset):
    def __init__(self, dataset_info, height=224, width=224, transform=None, evaluation=False, flip=False, multi_labels=False):
        """
        Args:
            config (string): easyDict with dataset details
            height (int): image height
            width (int): image width
            transform: pytorch transforms for transforms and tensor conversion
            bbox_mode is for various options to construct bounding boxes where 0 is use the normal object bbox
        """
        self.dataset_info = dataset_info
        #Images transformation parameters
        self.height = height
        self.width = width
        self.transform = transform
        self.evaluation = evaluation
        self.flip = flip
        self.multi_labels = multi_labels
        self.use_lm_box_scaling = True
        
  
    def __getitem__(self, index):
        return self.__getSkiimage__(index)
    
    def __getbbox__(self,index):  
        if self.gt_bbox:
            if  self.gt_bbox_data is None:
                self.__loadBbox__()
            
            #Get a the unscaled bbox
            boxes = self.gt_bbox_data[index].split(';')[:-1]
            bbox = np.zeros((len(boxes), 4),dtype=int)
            label = np.zeros((len(boxes),),dtype=int)
            for i, j in enumerate(boxes):
                row = np.fromstring(j, dtype=int, sep=" ")
                bbox[i,:] = row[0:4]
                label[i] = row[4]
            
            return bbox ,label
        else:
            print('No Bbox Data')
            return
    
       
    def __getExtraLabel__(self,index, ind=0):
        #Get extra labels
        if self.extra_gt_labels:
            if  self.gt_extra_data[ind] is None:
                self.__loadExtra__(ind)
            
            label = self.gt_extra_data[ind][index]
                
            return label
        else: 
            return
    
 
    def __getVideoLabel__(self, index):
        if self.video_frames:
            if  self.video_frames_data is None:
                self.__loadVideo__()
                
            return self.video_frames_data[index]
        else: 
            return
        

    def __getMultiLabels__(self,index):
        #Get multilabels generally used for mulit-task or label learning 
        if self.multi_labels:
            labels = self.multi_lab_data[index].split(';')[:-1]
            multi_labels = np.zeros((len(labels),self.num_multi_lab),dtype=int)
            for i, j in enumerate(labels):
                row = np.fromstring(j, dtype=int, sep=" ")
                multi_labels[i,:] = row[0:self.num_multi_lab]
    
            return multi_labels
        else: 
            return
    
    def __getLM__(self,index):
        #Get multilabels generally used for mulit-task or label learning 
        if self.landmarks:
            if  self.gt_landmark_data is None:
                self.__loadLandmarks__()
                
            lms = self.landmark_data[index].split(';')[:-1]
            lm = np.zeros((len(lms), self.landmark_number*2),dtype=int)
            vis = np.zeros((len(lms), self.landmark_number),dtype=int)
            for i, j in enumerate(lms):
                items = j.split(' ')
                for k in range(0,self.landmark_number*2,2):
                    lm[i,k] = int(float(items[k]))
                    lm[i,k+1] = int(float(items[k+1]))
                    if lm[i,k] !=0 and lm[i,k+1] !=0:
                        vis[i,int(k/2)] = 1
            return lm, vis
        else:
            print('No Landmark Data')
            return
        
    def __getLmAsBox__(self,index, bbox):
        #Transform a sset of XY landmarks to a set of bbox for each visible landmark (this was initially done in the ECCV work)
        scale_factors = [20,40,60]
        if self.use_lm_box_scaling:
            scale_level = scale_factors[random.randint(0, 3)]
        else:
            scale_level = 20
        
        lm,vis = self.__getLM__(index)
        scale_heights = ((bbox[:,2] - bbox[:,0]) /100 *scale_level) / 2
        scale_widths =  ((bbox[:,3] - bbox[:,1]) /100 *scale_level) / 2
        bbox = np.zeros((lm.shape[0]*self.landmark_number, 4),dtype=int)
        label = np.zeros((lm.shape[0]*self.landmark_number,),dtype=int)
        label_key = np.arange(self.landmark_number)
        vis = vis.reshape(lm.shape[0]*self.landmark_number)
        keep = vis != 0
        for i in range(0,lm.shape[0]):
            index_val = i*self.landmark_number
            start = index_val
            end = index_val + self.landmark_number
            bbox[start:end,1] = lm[i,::2] - scale_widths[i]
            bbox[start:end,3] = lm[i,::2] + scale_widths[i]
            bbox[start:end,0] = lm[i,1::2] - scale_heights[i]
            bbox[start:end,2] = lm[i,1::2] + scale_heights[i]
            label[start:end] = label_key

        bbox = bbox[keep,:]
        label = label[keep]
 
                    
        return bbox, label
    
    def __getLandmarksAsBox__(self, index, landmarks=None, visibility=None):     
        
        if landmarks is None:
            landmarks, visibility = self.__getLM__(index)
        
        #Need to call the get landmarks function here
        bbox = np.zeros((len(landmarks), 4),dtype=int)
        label = np.zeros((len(landmarks),),dtype=int)

        for i in range(landmarks.shape[0]):
            #Reshape inot N by 2 columns of X an Y landmarks
            tmp_lm =  landmarks[i].reshape(self.landmark_number,2)
            bbox[i,3] = np.max(tmp_lm[:,0])
            bbox[i,1] = np.min(tmp_lm[:,0])
            bbox[i,2] = np.max(tmp_lm[:,1])
            bbox[i,0] = np.min(tmp_lm[:,1])
                    
        return bbox, label
    
    def __parseMultiLabels__(self,config):
        self.multi_lab_path = self.base_path + self.subset_name + "/gt_multi/" 
        data = pd.read_csv(self.base_path  + 'dataloader/' +  self.subset_name + "_multi_file.txt", delimiter='\t', header=None)
        self.multi_lab_data = np.asarray(data.iloc[:, 1])
        self.num_multi_lab = config['DATA']['EXTRA_LABELS_NUM']
        self.names_multi_lab = config['DATA']['EXTRA_LABELS_NAMES']
        self.types_multi_lab = config['DATA']['EXTRA_LABELS_TYPE']
        self.num_classes_mulit_lab = config['DATA']['EXTRA_LABELS_TOTAL_CLASSES']                    
    
        return
    

    def __getVideoIndex__(self):
        
        data = pd.read_csv(self.base_path  + 'dataloader/' +  self.subset_name + "_video_file.txt", delimiter='\t', header=None)
        
        return np.asarray(data.iloc[:, 1])
    
    
    def __len__(self):
        return len(self.dataset_info.image_locations)
    
    
    def __getPILimage__(self, index):
        #Get orignal image
        img = Image.open(self.__getImagePath__(index))
        img = img.convert('RGB')
        
        return img
    
    def __getSkiimage__(self, index):
        img = io.imread(self.__getImagePath__(index))
         #Check 3 channels 
        if(len(img.shape)<3):
              w, h = img.shape
              ret = np.empty((w,h,3), dtype=np.uint8)
              ret[:,:,2] = ret[:,:,1] = ret[:,:,0] = img
              img = ret
        
        return img
    
    def __getImagePath__(self, index):
        return self.dataset_info.image_locations[index]
    
     
    def __inverse_normalize__(self,img):
        if self.caffe_pretrain:
            img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
            return img[::-1, :, :]
        # approximate un-normalize for visualize
        return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


    def caffe_normalize(self,img):
        """
        return appr -125-125 BGR
        """
        img = img[[2, 1, 0], :, :]  # RGB-BGR
        img = img * 255
        mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
        img = (img - mean).astype(np.float32, copy=True)
        return img
    
    def eval_transforms(self,img, min_size=600, max_size=1000):
        
        H, W = img.size
        scale1 = min_size / min(H, W)
        scale2 = max_size / max(H, W)
        scale = min(scale1, scale2)
        transformations = transforms.Compose([transforms.Resize(size=(int(W * scale),int(H *scale))),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        img = transformations(img)
        
        return img, [H,W]
    
    def random_scaling(self,img, min_size=600, max_size=1000):
        
        return img
    
    def random_box_size(self,img, min_size=600, max_size=1000):
        
        
        return img
    
    def flipbox_xy(self,box):
        
        return box
    

class TrainingObjectDataset(CustomDataset): 
    def __init__(self, config, height=224, width=224, transform=None, evaluation=False, flip=False, multi_labels=False):
        CustomDataset.__init__(self, config, height=224, width=224, transform=None, evaluation=False, flip=False, multi_labels=False)
        self.lmToBbox  = False
        
    def __getitem__(self, index):
        img = self.__getPILimage__(index)
                                
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
    

#Sub-class to returns 
class PredictionObjectDataset(CustomDataset): 
    def __init__(self, config, height=224, width=224, transform=None, evaluation=False, flip=False, multi_labels=False ):
        CustomDataset.__init__(self, config, height=224, width=224, transform=None, evaluation=False, flip=False, multi_labels=False)
        
    def __getitem__(self, index):
        img = self.__getPILimage__(index)
        img, scale, size = self.transforms(img)
 
        return img, scale, size 
        
    def __len__(self):
        return len(self.dataset_info.image_locations)
    
    def transforms(self, img, min_size=600, max_size=1000):
        H, W = img.size
        scale1 = min_size / min(H, W)
        scale2 = max_size / max(H, W)
        scale = min(scale1, scale2)
        transformations = transforms.Compose([transforms.Resize(size=(int(W * scale),int(H *scale)))])
        img = transformations(img)
        
        transformations = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        img = transformations(img)

        return img, scale, [H, W]

    
class ImageOnlyDataset(CustomDataset): 
    def __init__(self, config, height=224, width=224, transform=None, evaluation=False, flip=False, multi_labels=False ):
        CustomDataset.__init__(self, config, height=224, width=224, transform=None, evaluation=False, flip=False, multi_labels=False)
        
    def __getitem__(self, index):
        img = self.__getSkiimage__(index)
 
        return img
        
    def __len__(self):
        return len(self.dataset_info.image_locations)


class stackedHGObjectDataset(CustomDataset): 
    def __init__(self, config, bbox_data, resolution=256.0, flip=False):
        CustomDataset.__init__(self, config)
        self.bbox_data, self.image_index = self.__processBboxIndex__(bbox_data)
        
           
    def __getitem__(self, index):
        image_ind = self.bbox_data[index][1]
        img = self.__getSkiimage__(image_ind)
        if img.shape[2] == 4: #PNG remove alpha
            img = img[:,:,:3]
        img, center, scale = self.transforms(img, self.bbox_data[index][0].transpose())
        
        return img, center, scale, image_ind
        
    def __len__(self):
        return len(self.bbox_data)
    
    
    def __processBboxIndex__(self, data):
        #BBox data from a face detector or ground truth requires forammting so each detection has a un for get items function
        bbox = list()
        index = list()
        for i,j in enumerate(data):
            index.append(len(j))
            for k in range(len(j)):
                bbox.append([j[k],i])
        
        return bbox, index
    
    def __singleListToImageList__(self, single):
        #Takes ne single list of predicted landmrks and returns a list of the landmarks that belong to a specifc image
        #as some image have multiple detections
        new_list = list()
        cnt = 0
        for i,j in enumerate(self.image_index):
            temp_lm = np.zeros((j,single[cnt].shape[1]))
            for k in range(j):
                temp_lm[k] = single[cnt]
                cnt = cnt + 1
            new_list.append(temp_lm)
        return new_list
    
    def transforms(self, img, bbox):
        #Maybe need to change the bbox stuff around due to the wierd predctions
        center = [bbox[3] - (bbox[3] - bbox[1]) / 2.0, bbox[2] - (bbox[2] - bbox[0]) / 2.0]
        center[1] = center[1] - (bbox[2] - bbox[0]) * 0.12
        scale = (bbox[3] - bbox[1] + bbox[2] - bbox[0]) / 195.0
        inp = self.crop(img, center, scale,resolution=256.0)
        inp = torch.from_numpy(inp.transpose(
                        (2, 0, 1))).float().div(255.0)#.unsqueeze_(0)

        return inp, center, scale
    
    def crop(self,image, center, scale, resolution=256.0):
        # Crop around the center point
        """ Crops the image around the center. Input is expected to be an np.ndarray """
        ul = self.pointtransform([1, 1], center, scale, resolution, True)
        br = self.pointtransform([resolution, resolution], center, scale, resolution, True)
        # pad = math.ceil(torch.norm((ul - br).float()) / 2.0 - (br[0] - ul[0]) / 2.0)
        if image.ndim > 2:
            newDim = np.array([br[1] - ul[1], br[0] - ul[0],
                               image.shape[2]], dtype=np.int32)
            newImg = np.zeros(newDim, dtype=np.uint8)
        else:
            newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
            newImg = np.zeros(newDim, dtype=np.uint8)
        ht = image.shape[0]
        wd = image.shape[1]
        newX = np.array(
                [max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
        newY = np.array(
                [max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
        oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
        oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)
        newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1]
               ] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]
        newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)),
                        interpolation=cv2.INTER_LINEAR)
        return newImg


    def pointtransform(self, point, center, scale, resolution, invert=False):
        _pt = torch.ones(3)
        _pt[0] = point[0]
        _pt[1] = point[1]

        h = 200.0 * scale
        t = torch.eye(3)
        t[0, 0] = resolution / h
        t[1, 1] = resolution / h
        t[0, 2] = resolution * (-center[0] / h + 0.5)
        t[1, 2] = resolution * (-center[1] / h + 0.5)

        if invert:
            t = torch.inverse(t)

        new_point = (torch.matmul(t, _pt))[0:2]

        return new_point.int()

class stackedHGImageOnly(stackedHGObjectDataset):
    def __init__(self, config, bbox_data, resolution=256.0, flip=False):
        stackedHGObjectDataset.__init__(self, config, bbox_data)
        
    def __getitem__(self, index):
        image_ind = self.bbox_data[index][1]
        img = self.__getSkiimage__(image_ind)
        img = self.transforms(img, self.bbox_data[index][0].transpose())
        
        return img
        
    def __len__(self):
        return len(self.bbox_data)
    
    def transforms(self, img, bbox):
        center = torch.FloatTensor([bbox[3] - (bbox[3] - bbox[1]) / 2.0, bbox[2] - (bbox[2] - bbox[0]) / 2.0])
        center[1] = center[1] - (bbox[2] - bbox[0]) * 0.12
        scale = (bbox[3] - bbox[1] + bbox[2] - bbox[0]) / 195.0

        inp = self.crop(img, center, scale,resolution=256.0)

        return inp
    
    
class videoAction(Dataset):
    def __init__(self, info, sample_duration=5):
        from libs.spatial_transforms import (Compose, Normalize, Scale, CenterCrop, ToTensor)
        frames = info.video_frames_data
        images = info.image_locations
        self.data = self.get_dataset(frames,images,sample_duration)
        self.spatial_transform = Compose([Scale(1),
                                 CenterCrop(250),
                                 ToTensor(),
                                 Normalize([114.7748, 107.7354, 99.4750], [1, 1, 1])])
        self.temporal_transform = TemporalSampling(sample_duration)
        self.loader = self.get_default_video_loader()
        self.labels = np.asarray(info.palsy_level_data, dtype=None, order=None)
        self.label = np.asarray(self.labels[:,2], dtype='int32', order=None)
        
    def __getitem__(self, index):
        paths = self.data[index]

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(len(paths))
        clip = self.loader(paths, frame_indices)
        if self.spatial_transform is not None:
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.label[index]
        
        return clip, target
        
    def __len__(self):
        return len(self.data)
    
    def transforms(self, img, bbox):
        center = torch.FloatTensor([bbox[3] - (bbox[3] - bbox[1]) / 2.0, bbox[2] - (bbox[2] - bbox[0]) / 2.0])
        center[1] = center[1] - (bbox[2] - bbox[0]) * 0.12
        scale = (bbox[3] - bbox[1] + bbox[2] - bbox[0]) / 195.0

        inp = self.crop(img, center, scale,resolution=256.0)

        return inp
    
    def get_dataset(self, frames,images, sample_duration):
        dataset = []
        
        frames_index = np.unique(frames)
        
       
        for frame in frames_index:
            index =  frames == frame
            filtered_list = [i for (i, v) in zip(images, index) if v]
            dataset.append(filtered_list)
        
        return dataset
    
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


class TemporalSampling(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, length): 
        
        if length > self.size:
            out = np.arange(0,length)
            new_out = signal.resample(out, self.size)
            new_out[0] = 0
            new_out[self.size - 1] = length - 1 
            new_out = np.round(new_out, decimals=0)
            new_out = new_out.astype(int) 
        elif length < self.size:
            out = np.arange(0,length)
        elif length == self.size:
            out = np.arange(0,length)
        return new_out
    


    
    
