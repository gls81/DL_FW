# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 11:25:12 2018

@author: gary
"""

import face_alignment
import numpy as np
from skimage import io
from scipy import io as mat
import glob
import os


def open_annotation_file(base_dir):
    
     
    box_file_path = os.path.join(base_dir,'train_roi_file.txt')
    test_image_file_path = os.path.join(base_dir,'test_img_file.txt')  
    test_box_file_path = os.path.join(base_dir,'test_roi_file.txt') 
    
    image_file = open(image_file_path,'r')
    box_file = open(box_file_path,'r')
    test_image_file = open(test_image_file_path,'r')
    test_box_file = open(test_box_file_path,'r')
    
    return image_file, box_file,test_image_file, test_box_file
 


def getLandmarks(fa,images_list,output_path):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i in images_list:
    
        print(i)       
        input = io.imread(i)
         #Check 3 channels 
        if(len(input.shape)<3):
              w, h = input.shape
              ret = np.empty((w,h,3), dtype=np.uint8)
              ret[:,:,2] = ret[:,:,1] = ret[:,:,0] = input
              input = ret
        
    
        st = i.split("/")
        filename = st[5].split(".")[0]
        #filename = st[9].split(".")[0]
        print(filename)

        preds = fa.get_landmarks(input,all_faces=True)          
        savename = output_path + '/' + filename + "_lms.mat"
        if preds is not None:        
            mat.savemat(savename, mdict={'finelms': preds})

    
    return

# Run the 3D face alignment on a test image, without CUDA.

  
dataset_name = 'LFW'
images_dir = 'D:/Uni/Datasets/' + dataset_name
output_path = 'D:/Uni/Code/3DMM_ResNet/' + dataset_name

image_file_path = os.path.join(images_dir,'train_img_file.txt') 
image_file = open(image_file_path,'r')


images_list = []

with image_file as f:
    for line in f:
        line =  images_dir + '/' + line.split()[1]
        line = line.replace("\\","/")
        images_list.append(line.rstrip())


#Create FAN object
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, enable_cuda=True, flip_input=True)
getLandmarks(fa,images_list,output_path)    








 
