# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 11:25:12 2018

@author: gary
"""

import face_alignment
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage import io
from scipy import io as mat
import glob
import os


def open_annotation_file(base_dir):
    
    image_file_path = os.path.join(base_dir,'train_img_file.txt')  
    box_file_path = os.path.join(base_dir,'train_roi_file.txt')
    test_image_file_path = os.path.join(base_dir,'test_img_file.txt')  
    test_box_file_path = os.path.join(base_dir,'test_roi_file.txt') 
    
    image_file = open(image_file_path,'r')
    box_file = open(box_file_path,'r')
    test_image_file = open(test_image_file_path,'r')
    test_box_file = open(test_box_file_path,'r')
    
    return image_file, box_file,test_image_file, test_box_file
 

def getLandmarks(fa,images_list,input_path,output_path):

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
        filename = st[7].split(".")[0]
        #filename = st[9].split(".")[0]
        print(filename)
        full_input_path = input_path#os.path.join(input_path, st[8],st[9],st[10],st[11])        
        full_output_path = output_path#os.path.join(output_path, st[8],st[9],st[10],st[11])

        if not os.path.exists(full_output_path):
            os.makedirs(full_output_path)
        
        filepath = full_input_path + '/' + filename + ".mat"
        faces = mat.loadmat(filepath)
        boxes = faces['bbox']
        if boxes.any():
            #print(boxes)
            preds = fa.get_landmarks_gs(input,boxes,all_faces=True)[:]            
            savename = full_output_path + '/' + filename + "lms.mat"        
            mat.savemat(savename, mdict={'finelms': preds})

    
    return

# Run the 3D face alignment on a test image, without CUDA.

dataset_name = 'Avengers2/'

base_dir = '/media/gary/SAMSUNG/Uni/Code/DeepJointPartModel'

images_dir = '/media/gary/SAMSUNG/Uni/Datasets/' + dataset_name

image_file, box_file,test_image_file, test_box_file = open_annotation_file(images_dir)

images_list = []

with image_file as f:
    for line in f:
        line =  images_dir + line.split()[1]
        line = line.replace("\\","/")
        images_list.append(line.rstrip())

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, enable_cuda=True, flip_input=False)
#Get list of images
#images_list = glob.glob("/media/gary/SAMSUNG/Uni/Datasets/FDDB/*.jpg")

input_path = os.path.join(base_dir, 'HGinput', 'Avengers2-Part')
output_path = os.path.join(base_dir, 'HGOutput', 'Avengers2-Part3D')
getLandmarks(fa,images_list,input_path,output_path)
input_path = os.path.join(base_dir, 'HGinput', 'Avengers2-Whole')
output_path = os.path.join(base_dir, 'HGOutput', 'Avengers2-Whole3D')

getLandmarks(fa,images_list,input_path,output_path)
input_path = os.path.join(base_dir, 'HGinput', 'Avengers2-Mer')
output_path = os.path.join(base_dir, 'HGOutput', 'Avengers2-Mer3D')
getLandmarks(fa,images_list,input_path,output_path)

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=True, flip_input=False)
input_path = os.path.join(base_dir, 'HGinput', 'Avengers2-Part')
output_path = os.path.join(base_dir, 'HGOutput', 'Avengers2-Part2D')
getLandmarks(fa,images_list,input_path,output_path)
input_path = os.path.join(base_dir, 'HGinput', 'Avengers2-Whole')
output_path = os.path.join(base_dir, 'HGOutput', 'Avengers2-Whole2D')

getLandmarks(fa,images_list,input_path,output_path)
input_path = os.path.join(base_dir, 'HGinput', 'Avengers2-Mer')
output_path = os.path.join(base_dir, 'HGOutput', 'Avengers2-Mer2D')
getLandmarks(fa,images_list,input_path,output_path)

#images_list = glob.glob("/media/gary/SAMSUNG/Uni/Code/DeepTree/Original_images/*.jpg")
#images_list = glob.glob("/media/gary/SAMSUNG/Uni/Datasets/300W/V1/01_Indoor/*.png")
#images_list = glob.glob("/media/gary/SAMSUNG/Uni/Datasets/300W/V1/02_Outdoor/*.png")



    
