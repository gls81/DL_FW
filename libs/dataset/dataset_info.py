# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 11:31:12 2018

@author: Gary
"""
#Data Set loader for the LS3d-W dataset
from torch.utils.data.dataset import Dataset
#from torchvision import transforms
import pandas as pd 
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from skimage import io
from CustomDataTools import utils
import random 
import cv2
import functools
from scipy import signal
import os
#A class to hold useful inforation for a datset this would include labelling information and can do some of the process,
# the aim of this is to pass such an object to a dataset to access the stored information. Also simplifies the dataset objects codes so they can speciclise in the task that are importnat like retuning an item in the correct format
#for a given task.
class DatasetInformation():
    def __init__(self, config):
        
        self.name = config['DATA']['DATASET'] + '_' + config['DATA']['SUB_DATASET']
        self.subset_name = config['DATA']['SUB_DATASET']
        self.subset_name = self.subset_name.replace('/','_')
        
        self.base_path = config['DATA']['BASE_PATH']
        self.img_path = self.base_path + self.subset_name + "/images/"
        self.images_file =  self.base_path  + 'dataloader/' +  self.subset_name + "/Images_file.npy"
        self.image_locations = np.load(self.images_file)

        self.gt_bbox = config['DATA']['GT_BBOX']
        self.gt_landmarks = config['DATA']['GT_LM']
        self.lm_to_box = False
        
        if self.gt_bbox:
            self.gt_bbox_file = self.base_path  + 'dataloader/' +  self.subset_name + "/GT_BBOX.npy"
            self.gt_bbox_label_file = self.base_path  + 'dataloader/' +  self.subset_name + "/GT_BBOX_LABEL.npy"
            self.gt_bbox_score_file = self.base_path  + 'dataloader/' +  self.subset_name + "/GT_BBOX_SCORE.npy"
            self.gt_bbox_data = np.load(self.gt_bbox_file)
            self.gt_bbox_label = np.load(self.gt_bbox_label_file)
            self.gt_bbox_score = np.load(self.gt_bbox_score_file)
       
        if self.gt_landmarks:
            self.gt_lm_file = self.base_path  + 'dataloader/' +  self.subset_name + "/GT_LM.npy"
            self.gt_landmark_data = np.load(self.gt_lm_file)
            self.gt_lm_vis_file = self.base_path  + 'dataloader/' +  self.subset_name + "/GT_LM_VIS.npy"
            self.gt_landmark_visibility_data = np.load(self.gt_lm_vis_file)
            self.gt_lm_index_file = self.base_path  + 'dataloader/' +  self.subset_name + "/GT_LM_INDEX.npy"
            self.gt_landmark_index_data = np.load(self.gt_lm_index_file)
            self.gt_landmark_anno = config['DATA']['GT_LM_PROCESSING']
            
        if not self.gt_bbox and self.gt_landmarks:
            self.gt_bbox_data, self.gt_bbox_label, self.gt_bbox_score = self.__landmarksToBoxes__(self.gt_landmark_data,self.gt_landmark_visibility_data)
            self.lm_to_box = True
            
        if config['DATA']['VIDEO']:
            self.video_frames = True
            self.video_frames_file = self.base_path  + 'dataloader/' +  self.subset_name + "/VIDEO_INDEX.npy"
            self.video_frames_data = np.load(self.video_frames_file)
        else:
            self.video_frames = False  
            
        if config['DATA']['PALSY_LEVEL_LABELS']:
            self.palsy_level_label = True
            self.palsy_level_file = self.base_path  + 'dataloader/' +  self.subset_name + "/GT_PALSY_LEVEL_LABEL.npy"
            self.palsy_level_data= np.load(self.palsy_level_file)
            self.palsy_level_info_file = self.base_path  + 'dataloader/' +  self.subset_name + "/GT_PALSY_LEVEL_LABEL_INFO.npy"
            self.palsy_level_data_info = np.load(self.palsy_level_info_file)
        else:
            self.palsy_level_label = False  
        
        return
    
    def __loadExtra__(self, label_index):
        data = np.load(self.extra_gt_files[label_index])
        self.gt_extra_data[label_index] = data[:,0]
        return

    
    #returns all VideoLabels for a dataset
    def __getVideoLabels__(self):
        if self.video_frames:
            if  self.video_frames_data is None:
                self.__loadVideo__()
                
            return self.video_frames_data
        else: 
            return
    
    def __getExtraLabels__(self, ind=0):
        #Get extra labels
        if self.extra_gt_labels:
            if  self.gt_extra_data[ind] is None:
                self.__loadExtra__(ind) 
                
            return self.gt_extra_data[ind]
        else: 
            return
    
    def __landmarksToBoxes__(self, lm_data, lm_vis=None):
        bbox_list = [None] * len(lm_data)
        label_list = [None] * len(lm_data)
        score_list = [None] * len(lm_data)
        
        for i, lms in enumerate(lm_data):
            if lms.shape[1] != 0 and self.gt_landmark_index_data[i][0]: 
                bbox = np.zeros((lms.shape[0], 4),dtype=int)
                label = np.zeros((lms.shape[0],),dtype=int)
                score = np.ones((lms.shape[0],), dtype=int)
                #label_key = np.arange(self.landmark_number)
                for j, lm in enumerate(lms):
                    temp = np.reshape(lm, (-1, 2))
                    if lm_vis is not None:
                        landmark_index = lm_vis[i][j] == 1
                        temp = temp[landmark_index,:]
                    #Do keep need to be looked at later
                    bbox[j,1] = min(temp[:,0])
                    bbox[j,0] = min(temp[:,1])
                    bbox[j,3] = max(temp[:,0])
                    bbox[j,2] = max(temp[:,1])
                    label[j] = 0
            else:
                bbox = np.zeros((lms.shape[0], 4),dtype=int)
                label = np.zeros((lms.shape[0],),dtype=int)
                score = np.ones((lms.shape[0],), dtype=int)
                

            bbox_list[i] = bbox
            label_list[i] = label
            score_list[i] = score
                
        return bbox_list, label_list, score_list

    def __stackSequenceFeatures__(self, features):
        #Utiliy to take a set of N by M featurews for individual frames and stacked them into sequnces for video based datasets
        if self.video_frames:
            stacked_features = list()
            ind = np.asarray(self.video_frames_data, dtype=None, order=None)
            for ii in range(min(self.video_frames_data),max(self.video_frames_data)+1):
                frames_index = np.where(ind == ii)
                feature_mat = np.zeros((68,2,len(frames_index[0])))
                for jj, value in enumerate(frames_index[0]):
                    tmp_pred = features[value][0].reshape(int(len(features[value][0])/2),2)
                    feature_mat[:,:,jj] = tmp_pred 
                stacked_features.append(feature_mat)   
        return stacked_features


