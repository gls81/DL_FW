# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 14:31:28 2018

A test to see what happens when we put a whole image rather than a face through the FAN

@author: Gary
"""
import torch
from torchvision import transforms
#from ..CustomDataTools.data_loader import CustomDatasetFromConfig
import face_alignment
#import face_alignment.fan_pytorch as face_alignment
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage import io
from scipy import io as mat

# Device configuration
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Load a dataset 

#from CustomDataTools.configs.Wider_config import cfg
##transformations = transforms.Compose([transforms.Resize(size=(224,244)),transforms.ToTensor()])
#dataset = CustomDatasetFromConfig(cfg)
#
##Load one image
#img = dataset.__getimage__(1)
img = io.imread('D:/Uni/Datasets/WiderFaces/WIDER_val/images/7--Cheering/7_Cheering_Cheering_7_884.jpg')
#img = 'D:/Uni/Datasets/WiderFaces/WIDER_val/images/7--Cheering/7_Cheering_Cheering_7_884.jpg'
#Do Face Detection
#face = fd.FaceDetection()
#bbox = face.detect_faces(img)

#BBOX IS THE IMAGE SIZE
bbox = [1,1,img.shape[1],img.shape[0]]
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=True, flip_input=False)
preds = fa.get_landmarks_with_fd(img)
#preds, out = fa.get_landmarks(img,bbox)
#pts = fa.get_landmarks(img,bbox,all_faces=False)[0]

#View heatmap
#hm = out.squeeze()
#for i in range(0,68):
#    test = hm[i]
#    plt.imshow(test, cmap='hot', interpolation='nearest')
#    plt.show()

#plot_landmarks(img,pts)
preds = preds[-1]
#fig = plt.figure(figsize=plt.figaspect(.5))
#ax = fig.add_subplot(1, 2, 1)
fig1 = plt.figure(figsize = (5,5))

plt.imshow(img, aspect='auto')
plt.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
plt.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
plt.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
plt.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
plt.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
plt.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
plt.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
plt.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
plt.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=6,linestyle='-',color='w',lw=2) 
plt.axis('off')



#ax.view_init(elev=90., azim=90.)
#ax.set_xlim(ax.get_xlim()[::-1])
plt.show()
fig1.savefig('example.png', dpi = 1000)



