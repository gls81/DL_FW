# -*- coding: utf-8 -*-
"""
Created on Thu May  3 11:38:08 2018

@author: gary
"""

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


images_list = glob.glob("/media/gary/SAMSUNG/*.jpg")

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=True, flip_input=False)


for i in images_list:
    input = io.imread(i)
    preds = fa.get_landmarks(input)[-1]
    st = i.split("/")
    filename = st[4].split(".")[0]
    
    dataout = "/media/gary/SAMSUNG"
    savename = dataout + '/' + filename + "2dlms.mat"        
    mat.savemat(savename, mdict={'finelms': preds})
    fileout = open(dataout + '/'  + filename + ".pts","w")
    for j in range(0,preds.shape[0]):
        fileout.write("%f %f\n" % (preds[j,0], preds[j,1]))
    fileout.close()

#    fig = plt.figure(figsize=plt.figaspect(.5))
#    ax = fig.add_subplot(1, 2, 1)
#    ax.imshow(input)
#    ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
#    ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
#    ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
#    ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
#    ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
#    ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
#    ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
#    ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
#    ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=6,linestyle='-',color='w',lw=2) 
#    ax.axis('off')
#
#    ax = fig.add_subplot(1, 2, 2, projection='3d')
#    surf = ax.scatter(preds[:,0]*1.2,preds[:,1],preds[:,2],c="cyan", alpha=1.0, edgecolor='b')
#    ax.plot3D(preds[:17,0]*1.2,preds[:17,1], preds[:17,2], color='blue' )
#    ax.plot3D(preds[17:22,0]*1.2,preds[17:22,1],preds[17:22,2], color='blue')
#    ax.plot3D(preds[22:27,0]*1.2,preds[22:27,1],preds[22:27,2], color='blue')
#    ax.plot3D(preds[27:31,0]*1.2,preds[27:31,1],preds[27:31,2], color='blue')
#    ax.plot3D(preds[31:36,0]*1.2,preds[31:36,1],preds[31:36,2], color='blue')
#    ax.plot3D(preds[36:42,0]*1.2,preds[36:42,1],preds[36:42,2], color='blue')
#    ax.plot3D(preds[42:48,0]*1.2,preds[42:48,1],preds[42:48,2], color='blue')
#    ax.plot3D(preds[48:,0]*1.2,preds[48:,1],preds[48:,2], color='blue' )
#
#    ax.view_init(elev=90., azim=90.)
#    ax.set_xlim(ax.get_xlim()[::-1])
#    plt.show()

    
