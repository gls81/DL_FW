# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 12:52:35 2018

@author: Gary
Function: Process to CNTK type format for image files, output is a txt file that can be rewad by CNTK and by dataloader class in pytorch
This format is a list of images within a specifc dataset.
"""

import numpy as np
import os
import glob
import pandas as pd
import scipy.io as sio
import shutil
#from torch.utils.serialization import load_lua
import pickle
#from easydict import EasyDict as edict
#from pathlib import Path
import json
from sklearn.model_selection import KFold

config_dir = 'D:/Research/Code/DL_FW/CustomDataTools/configs/'


def create_annotation_file(set_type,base_dir, mode='w'):
    
    #Sanity check for / in set_type 
    check = set_type.split('/')
    if len(check) != 1:
        set_type =  "_".join(check) 
    
    file_name = set_type + '.txt'
    file_path = os.path.join(base_dir,file_name)
    
    #if os.path.isfile(file_path):
    #    os.remove(file_path)
    file = open(file_path,mode)
    
    return file


def getImageList(images_dir, st, ext_list):

    image_list = []
        
    for ext in ext_list: 
        image_search_path = images_dir + st + ext
        image_list += glob.glob(image_search_path,recursive=True)

    return image_list


def create_anno_text(config): 
    base_dir = config['DATA']['BASE_PATH']
    dataset_name = config['DATA']['DATASET'] + '_' + config['DATA']['SUB_DATASET'] 
    anno_dir = base_dir +  'dataloader/' + config['DATA']['SUB_DATASET'] 
    
    #Keep our files in a folder in the base dir
    if not os.path.exists(anno_dir):
        os.makedirs(anno_dir)  
    else:
        shutil.rmtree(anno_dir)           #removes all the subdirectories!
        os.makedirs(anno_dir)
    
    if config['DATA']['SUB_DIRS']:
        st = '/**/*'
    else:
        st = '/*'
            
    image_file = create_annotation_file('Images_file',anno_dir)
    #Get images folder for ds
    images_dir = base_dir + config['DATA']['SUB_DATASET']  + '\\images'   
    ext = ['.jpg','.png']
    image_list = getImageList(images_dir, st, ext)
        
    #If the dataset is a number of videos split into frames create a index
    if config['DATA']['VIDEO']:
        video_file = create_annotation_file('Video_index_file',anno_dir)
        video_ind = 0
            
            
    #Going to use the full path to the image
    for i, j in enumerate(image_list):
        image_name = j.split('\\')
        image_path = "/".join(image_name)              
        img_line = "{}\t{}\t0\n".format(i, image_path)
        image_file.write(img_line)
        
        if config['DATA']['VIDEO']:
            if i != 0:
                prev_image_name = image_list[i-1].split('\\')
                if image_name[3] !=  prev_image_name[3]:
                    video_ind = video_ind + 1
            video_line = "{}\t{}\n".format(i, video_ind)
            video_file.write(video_line)
            
        
        if config['DATA']['GT_BBOX']:
            ext = config['DATA']['GT_BBOX_EXT']
            bbox_name = list(image_name)
            writeBBOXfile(dataset_name, bbox_name, anno_dir, ext, i)
            
        if config['DATA']['GT_LM']:
            ext = config['DATA']['GT_LM_EXT']
            lm_name = list(image_name)
            writeLMfile(dataset_name, lm_name, anno_dir, ext, i, config['DATA']['GT_LM_PROCESSING'])
            
    if config['DATA']['EXTRA_LABELS']:
        label_data = pd.read_csv(config['DATA']['EXTRA_LABELS_FILE'], delimiter='\t', header=None)
        label_data = np.asarray(label_data.iloc[:])
        for l,label_name in enumerate(config['DATA']['EXTRA_LABELS_NAMES']):
            file_name = 'Extra _' + label_name + '_file'
            label_file = create_annotation_file(file_name,anno_dir)
            
            for line in range(len(label_data)):
                label_line = "{}\t{}\n".format(line, label_data[line,l])
                label_file.write(label_line)
            label_file.close()     
    return    
            
            #writeExtraLabelsfile(dataset_name, image_name, anno_dir, i)
        
    image_file.close()

def getSearchString(sub):
    if sub:
        ss = '/**/*'
    else:
        ss = '/*'
    return ss

def loadJSON(file):
    with open(file) as f:
        data = json.load(f)
    
    return data

def checkLabels(a,b):
    flag = False
    for label in a:
        if label in b:
            flag = True
        else:
            return False
    
    return flag
    

def create_anno_numpy(config_file):

    pwd = os.getcwd()
    config_file = os.path.join(pwd,"configs","data", config_file)
    config = loadJSON(config_file)

    labels_file = os.path.join(pwd,"configs", 'labels.json')
    labels = loadJSON(labels_file)

    save_dir = os.path.join(pwd,"datasets") #'D:/Research/Experiments/Base/Datasets/'
    #datasets_dir =  'D:/Research/Datasets/'

    #To deal with datasets combinations
    hybrid = config['HYBRID']
    ext = ['.jpg','.png']

    if hybrid:
        match_labels = False
        base_configs = config['PATH']
        dataset_name = config['DATASET']
        anno_image_dir = save_dir  + dataset_name + '/dataloader/' + config['SUB_DATASET'] + '/image/'
        anno_video_dir = save_dir  + dataset_name + '/dataloader/' + config['SUB_DATASET'] +  '/video/' 
        image_list = []
        hybrid_index = []
        hybrid_split = []
        for i,ds in enumerate(base_configs):
            tmp_list = []
            config_file = config_dir + ds + '.json'
            ds_config = loadJSON(config_file)
            match_labels = checkLabels(config['LABELS'],ds_config['LABELS'])
            if not match_labels:
                print("Labels do not match this wont work")  
            assert match_labels in [True]
            ss = getSearchString(ds_config['SUB_DIRS'])
            images_dir = ds_config['PATH'] + config['SUB_DATASET']  + '/images'
            tmp_list = getImageList(images_dir, ss, ext)
            hybrid_index = hybrid_index + [i] * len(tmp_list)
            hybrid_split.append(len(hybrid_index))
            image_list = image_list + tmp_list
        
    else:
        ss = getSearchString(config['SUB_DIRS'])
        base_dir = os.path.join(save_dir, config['PATH'])
        #base_dir = config['PATH']
        dataset_name = config['DATASET'] + '_' + config['SUB_DATASET']
        anno_image_dir = os.path.join(base_dir, 'dataloader' , config['SUB_DATASET'], 'image')
        anno_video_dir = os.path.join(base_dir, 'dataloader' , config['SUB_DATASET'], 'video')
        #Get images folder for ds
        images_dir = os.path.join(base_dir,config['SUB_DATASET'], 'images' )
        image_list = getImageList(images_dir, ss, ext)
    
    
    image_anno = [None] * len(image_list)
    
        
    #Keep our files in a folder in the base dir
    if not os.path.exists(anno_image_dir):
        os.makedirs(anno_image_dir)
    else:
        shutil.rmtree(anno_image_dir)           #removes all the subdirectories!
        os.makedirs(anno_image_dir)
        
    if not os.path.exists(anno_video_dir):
        os.makedirs(anno_video_dir)  
    else:
        shutil.rmtree(anno_video_dir)           #removes all the subdirectories!
        os.makedirs(anno_video_dir)
    
    #Going to use the full path to the image
    for i, j in enumerate(image_list):
        image_name = j.split('\\')
        image_path = "/".join(image_name)
        image_anno[i] =  image_path   
    savename = anno_image_dir + 'Images_file.npy'
    with open(savename, "wb") as fp:   #Pickling
            pickle.dump(image_anno, fp)
    savename = anno_image_dir + 'Images_file.mat'       
    sio.savemat(savename, {'images': image_anno})
    
    
# Code to try and genrlsie the labbel creation WIP
#    for lab in labels:
#       if lab in config['LABELS']: 
#           data_dic = {}
#           for key in labels[lab]:
#               data_dic[key] =  [None] * len(image_anno)
#           ext = config['LABELS'][lab]['EXT']
    #fold = KFold(len(image_anno))

    # kf = KFold(n_splits=10)
    # for train, test in kf.split(image_anno):
    #     te = test
    #     tr = train
                       
    if 'VIDEO' in config['LABELS']:
        video_ind = 1
        video_index = list()
        for i, name in enumerate(image_anno):
            if i != 0:
                prev_name = image_anno[i-1].split('/')
                name = name.split('/')
                if name[-2] !=  prev_name[-2]:
                    video_ind = video_ind + 1
            video_index.append(video_ind)
        savename = anno_image_dir + 'VIDEO_INDEX.npy'
        with open(savename, "wb") as fp:   #Pickling
            pickle.dump(video_index, fp)
        savename = anno_video_dir + 'VIDEO_INDEX.npy'
        with open(savename, "wb") as fp:   #Pickling
            pickle.dump(video_index, fp)
    
    if 'SUBJECT' in config['LABELS']:
        subject_list = [None] * len(image_anno)
        for i, name in enumerate(image_anno):
            if hybrid:
                subject_list[i] = str(hybrid_index[i])  + '.' + str(singleLabel(dataset_name, name, config['LABELS']['SUBJECT']['EXT'], i, labels['SUBJECT']['LABEL']['FOLDER']))
            else:
                subject_list[i] = singleLabel(dataset_name, name, config['LABELS']['SUBJECT']['EXT'], i, labels['SUBJECT']['LABEL']['FOLDER'])
        savename = anno_image_dir + labels['SUBJECT']['LABEL']['SAVENAME']
        with open(savename, "wb") as fp:   #Pickling
            pickle.dump(subject_list, fp)
        if 'VIDEO' in config['LABELS']:
            subject_list_video = imageToVideoLabels(video_index,subject_list, label=True)
            savename = anno_video_dir + labels['SUBJECT']['LABEL']['SAVENAME']
            with open(savename, "wb") as fp:   #Pickling
                pickle.dump(subject_list_video, fp)
                
    if 'FACE' in config['LABELS']:
        bbox_list = [None] * len(image_anno)
        bbox_label = [None] * len(image_anno)
        bbox_scores = [None] * len(image_anno) #Note this is more for trouble shooting all scores are 1 for gt
        ext =  config['LABELS']['FACE']['EXT']
        for i, name in enumerate(image_anno):
            bbox_list[i], bbox_label[i], bbox_scores[i] = writeBBOXlist(dataset_name, name, ext, i)
        savename = anno_image_dir + 'GT_BBOX.npy'
        with open(savename, "wb") as fp:   #Pickling
            pickle.dump(bbox_list, fp)
        savename = anno_image_dir + 'GT_BBOX_LABEL.npy'
        with open(savename, "wb") as fp:   #Pickling
            pickle.dump(bbox_label, fp) 
        savename = anno_image_dir + 'GT_BBOX_SCORE.npy'
        with open(savename, "wb") as fp:   #Pickling
            pickle.dump(bbox_scores, fp) 
        
    if 'FACE_LM' in config['LABELS']:
        landmark_list = [None] * len(image_anno)
        visibilty_list = [None] * len(image_anno)
        gt_list =  np.zeros((len(image_anno), 1), dtype=bool)#[None] * len(image_anno)
        ext = config['LABELS']['FACE_LM']['EXT']
        for i, name in enumerate(image_anno):
            #print(name)
            landmark_list[i], visibilty_list[i], gt_list[i] = writeLMlist(dataset_name, name, ext, config['LABELS']['FACE_LM']['ANNO'])
        savename = anno_image_dir + '/GT_LM.npy'
        with open(savename, "wb") as fp:   #Pickling
            pickle.dump(landmark_list, fp)
        savename = anno_image_dir + 'GT_LM_VIS.npy'   
        with open(savename, "wb") as fp:   #Pickling
            pickle.dump(visibilty_list, fp) 
        savename = anno_image_dir + 'GT_LM_INDEX.npy'  #Index is used for which images have gts as some do not have gt for every image in the set 
        with open(savename, "wb") as fp:   #Pickling
            pickle.dump(gt_list, fp)
        savename = anno_image_dir + 'GT_LM_INDEX.mat'       
        sio.savemat(savename, {'gt_lms_index': gt_list})
    
    
    if 'PALSY_BINARY' in config['LABELS']:
        palsy_level_list = [None] * len(image_anno)
        for i, name in enumerate(image_anno):
            palsy_level_list[i] = singleLabel(dataset_name, name, config['LABELS']['PALSY_BINARY']['EXT'], i, labels['PALSY_BINARY']['LABEL']['FOLDER'])
        savename = anno_image_dir + 'GT_PALSY_BINARY_LABEL.npy'
        with open(savename, "wb") as fp:   #Pickling
            pickle.dump(palsy_level_list, fp)
        if 'VIDEO' in config['LABELS']:
            palsy_level_video = imageToVideoLabels(video_index,palsy_level_list, label=True)
            savename = anno_video_dir + '/GT_PALSY_BINARY_LABEL.npy'
            with open(savename, "wb") as fp:   #Pickling
                pickle.dump(palsy_level_video, fp)
                
    if 'PALSY_LEVELS' in config['LABELS']:
        palsy_level_list = [None] * len(image_anno)
        for i, name in enumerate(image_anno):
            palsy_level_list[i] = singleLabel(dataset_name, name, config['LABELS']['PALSY_LEVELS']['EXT'], i, labels['PALSY_LEVELS']['LABEL']['FOLDER'])
        savename = anno_image_dir + labels['PALSY_LEVELS']['LABEL']['SAVENAME']
        with open(savename, "wb") as fp:   #Pickling
            pickle.dump(palsy_level_list, fp)
        if 'VIDEO' in config['LABELS']:
            palsy_level_video = imageToVideoLabels(video_index,palsy_level_list, label=True)
            savename = anno_video_dir + labels['PALSY_LEVELS']['LABEL']['SAVENAME']
            with open(savename, "wb") as fp:   #Pickling
                pickle.dump(palsy_level_video, fp)
                
    if 'SMILE' in config['LABELS']:
        palsy_level_list = [None] * len(image_anno)
        for i, name in enumerate(image_anno):
            palsy_level_list[i] = singleLabel(dataset_name, name, config['LABELS']['SMILE']['EXT'], i, labels['SMILE']['LABEL']['FOLDER'])
        savename = anno_image_dir + 'GT_SMILE_LABEL.npy'
        with open(savename, "wb") as fp:   #Pickling
            pickle.dump(palsy_level_list, fp)
        if 'VIDEO' in config['LABELS']:
            palsy_level_video = imageToVideoLabels(video_index,palsy_level_list,label=True)
            savename = anno_video_dir + '/GT_SMILE_LABEL.npy'
            with open(savename, "wb") as fp:   #Pickling
                pickle.dump(palsy_level_video, fp)
                
    if 'MOUTH' in config['LABELS']:
        palsy_level_list = [None] * len(image_anno)
        for i, name in enumerate(image_anno):
            palsy_level_list[i] = singleLabel(dataset_name, name, config['LABELS']['MOUTH']['EXT'], i, labels['MOUTH']['LABEL']['FOLDER'])
        savename = anno_image_dir + labels['MOUTH']['LABEL']['SAVENAME']
        with open(savename, "wb") as fp:   #Pickling
            pickle.dump(palsy_level_list, fp)
        if 'VIDEO' in config['LABELS']:
            palsy_level_video = imageToVideoLabels(video_index,palsy_level_list,label=True)
            savename = anno_video_dir + labels['MOUTH']['LABEL']['SAVENAME']
            with open(savename, "wb") as fp:   #Pickling
                pickle.dump(palsy_level_video, fp)
                
    if 'EYE' in config['LABELS']:
        palsy_level_list = [None] * len(image_anno)
        for i, name in enumerate(image_anno):
            palsy_level_list[i] = singleLabel(dataset_name, name, config['LABELS']['EYE']['EXT'], i, labels['EYE']['LABEL']['FOLDER'])
        savename = anno_image_dir + labels['EYE']['LABEL']['SAVENAME']
        with open(savename, "wb") as fp:   #Pickling
            pickle.dump(palsy_level_list, fp)
        if 'VIDEO' in config['LABELS']:
            palsy_level_video = imageToVideoLabels(video_index,palsy_level_list,label=True)
            savename = anno_video_dir + labels['EYE']['LABEL']['SAVENAME']
            with open(savename, "wb") as fp:   #Pickling
                pickle.dump(palsy_level_video, fp)
                
    if 'BROW' in config['LABELS']:
        palsy_level_list = [None] * len(image_anno)
        for i, name in enumerate(image_anno):
            palsy_level_list[i] = singleLabel(dataset_name, name, config['LABELS']['BROW']['EXT'], i, labels['BROW']['LABEL']['FOLDER'])
        savename = anno_image_dir + labels['BROW']['LABEL']['SAVENAME']
        with open(savename, "wb") as fp:   #Pickling
            pickle.dump(palsy_level_list, fp)
        if 'VIDEO' in config['LABELS']:
            palsy_level_video = imageToVideoLabels(video_index,palsy_level_list,label=True)
            savename = anno_video_dir + labels['BROW']['LABEL']['SAVENAME']
            with open(savename, "wb") as fp:   #Pickling
                pickle.dump(palsy_level_video, fp)
                
    if 'EXPRESSION_CK' in config['LABELS']:
        palsy_level_list = [None] * len(image_anno)
        for i, name in enumerate(image_anno):
            palsy_level_list[i] = singleLabel(dataset_name, name, config['LABELS']['EXPRESSION_CK']['EXT'], i, labels['EXPRESSION_CK']['LABEL']['FOLDER'])
        savename = anno_image_dir + labels['EXPRESSION_CK']['LABEL']['SAVENAME']
        with open(savename, "wb") as fp:   #Pickling
            pickle.dump(palsy_level_list, fp)
        if 'VIDEO' in config['LABELS']:
            palsy_level_video = imageToVideoLabels(video_index,palsy_level_list,label=True)
            savename = anno_video_dir + labels['EXPRESSION_CK']['LABEL']['SAVENAME']
            with open(savename, "wb") as fp:   #Pickling
                pickle.dump(palsy_level_video, fp)
    
    
    if 'VIDEO' in config['LABELS']:
        image_video_anno = imageToVideoLabels(video_index,image_anno)
        
        if hybrid:
            hybrid_video_index = imageToVideoLabels(video_index,hybrid_index,label=True)    
            if 'EXTEND' in config:
                indexes = [0]
                indexes.append(np.where(np.asarray(hybrid_video_index)[:-1] != np.asarray(hybrid_video_index)[1:])[0][0])
            for ii, extend in enumerate(config["EXTEND"]): 
                if extend:
                    for jj in range(indexes[ii]+1,len(hybrid_video_index)):
                        image_video_anno[jj]  = videoExtender(image_video_anno[jj])
                    
        
        savename = anno_video_dir + '/Images_file.npy'
        with open(savename, "wb") as fp:   #Pickling
            pickle.dump(image_video_anno, fp)

    #fold = KFold(len(image_video_anno))
    
    
    #Need to do this bit.
#    if 'PALSY_GRADES' in config['LABELS']:
#        data = pd.read_csv(base_dir + config['SUB_DATASET'] + '/extra_labels.txt', delimiter='\t', header=None).values
#        palsy_level_list = [None] * len(data)
#        for i in range(len(data)):
#            palsy_level_list[i] = [data[i][0],data[i][1],data[i][2],data[i][3],data[i][4],data[i][5],data[i][6],data[i][7]] #,data[i][8],data[i][9]] 
#        savename = anno_dir + '/GT_PALSY_LEVEL_LABEL.npy'
#        with open(savename, "wb") as fp:   #Pickling
#            pickle.dump(palsy_level_list, fp)
#        
#        data_info = [None] * data.shape[1]
#        data_info = palsyLevelLabelInfo(data_info)
#        savename = anno_dir + '/GT_PALSY_LEVEL_LABEL_INFO.npy'
#        with open(savename, "wb") as fp:   #Pickling
#            pickle.dump(data_info, fp)


def videoExtender(video):
    #Takes a set so video frames reverses the list and concatenates. The purpose is to extend seuqnces like the CK+ which only have from nuetral to full expression, extending mirrors and bring them back to neutral
    copy = video.copy()
    copy.reverse()
    extended = video + copy  
    
    return extended


def imageToVideoLabels(frames,data, label=False):
    #Takes the annotation for individual and put these into video labels
    dataset = []
        
    frames_index = np.unique(frames)
           
    for frame in frames_index:
        index =  frames == frame
        filtered_list = [i for (i, v) in zip(data, index) if v]
        dataset.append(filtered_list)
        
    if label:
        labels = []
        for i in dataset:
            labels.append(i[0])
        return labels        
    else:
        return dataset

def singleLabel(ds, file_name, ext, ind, folder):
    file_name = file_name.replace("/images/", folder)
    file_path = file_name[:-3] + ext  
    file_data = pd.read_csv(file_path, delimiter='\t', header=None).values
    label = file_data[0][0]

    return label
    
        
def palsyLevelLabelInfo(data_info):
    
    data_info[0] =    {"name":"Seq_ID", "type":"Label", "values":None}
    data_info[1] = {"name":"Subject", "type":"Label", "values":None}
    data_info[2] = {"name":"Palsy", "type":"Class", "values":[0,1]}
    data_info[3] = {"name":"Level", "type":"Class", "values":[1,2,3,4,5,6]}
    data_info[4] = {"name":"Motion", "type":"Class", "values":[0,1]}
    data_info[5] = {"name":"Eye", "type":"Class", "values":[0,1]}
    data_info[6] = {"name":"Mouth", "type":"Class", "values":[0,1]}
    data_info[7] = {"name":"Brow", "type":"Class", "values":[0,1]}
    #data_info[8] = {"name":"Nose", "type":"Class", "values":[0,1]}
    #data_info[9] = {"name":"Side", "type":"Class", "values":[0,1]}
                       
    return data_info
    
    

    bbox_name = bbox_name.replace("/images/", "/gt_bbox/")
    bbox_path = bbox_name[:-3] + ext  
    bbox_data = pd.read_csv(bbox_path, delimiter='\t', header=None).values
    bbox = np.zeros((len(bbox_data),4))
    label = np.zeros((len(bbox_data),), dtype=np.int)
    scores = np.ones((len(bbox_data),))
    for i,j in enumerate(bbox_data):
        bbox[i,:] = [str(j[0]),str(j[1]),str(j[2]),str(j[3])] 
        label[i] = str(j[4])
    return bbox, label, scores           


def writeBBOXlist(ds, bbox_name, ext, ind):
    
    bbox_name = bbox_name.replace("/images/", "/gt_bbox/")
    bbox_path = bbox_name[:-3] + ext  
    bbox_data = pd.read_csv(bbox_path, delimiter='\t', header=None).values
    bbox = np.zeros((len(bbox_data),4))
    label = np.zeros((len(bbox_data),), dtype=np.int)
    scores = np.ones((len(bbox_data),))
    for i,j in enumerate(bbox_data):
        bbox[i,:] = [str(j[0]),str(j[1]),str(j[2]),str(j[3])] 
        label[i] = str(j[4])
    return bbox, label, scores

def writeBBOXfile(ds, bbox_name, anno_dir, ext, ind):
    
    outfile = create_annotation_file('GT_bbox_file',anno_dir, mode='a')
    for i, j in enumerate(bbox_name):
        if j =='images':
            bbox_name[i] = 'gt_bbox'
    bbox_name[-1] = bbox_name[-1][:-3] + ext
    bbox_path = "/".join(bbox_name) 
    print(bbox_path)

    bbox_data = pd.read_csv(bbox_path, delimiter='\t', header=None).values
    bbox = ''
    for i in bbox_data:
        bbox += "{} {} {} {} {} ;".format(str(i[0]),str(i[1]),str(i[2]),str(i[3]),str(i[4])) 
    
    bbox_line = "{}\t{}\n".format(ind, bbox)
 
    outfile.write(bbox_line)
    outfile.close()
    return


def writeLMlist(ds, lm_name, ext, processing):
    
    if 'IBUG' in processing:
        lm_total_num = 68
    elif 'AFLW' in processing: 
        lm_total_num = 21
    
    lm_name = lm_name.replace("/images/", "/gt_lm/")
    lm_path = lm_name[:-3] + ext
     
    if os.path.isfile(lm_path):
        has_gt = True
        if ext == 't7':
            lm_data = load_lua(lm_path,long_size=8)
            lm_data = lm_data.data.numpy()
            lm_data = np.reshape(lm_data, lm_total_num*2)
            lm_data = lm_data[..., np.newaxis]
            lm_data = lm_data.T
        #lm_data = np.array2string(lm_data, precision=1, separator=' ')
        #lm_data = lm_data[2:-1]        
        elif 'IBUG' in processing:
            if ext == 'PTS' or ext == 'pts':
                #Using file as Pandas has issue with double space sused in some files
                f = open(lm_path, 'r')
                lines = f.readlines()
                f.close()
                lm_data = lines[3:71]
                tmp_lms = np.zeros((lm_total_num,2))
                tmp_vis = np.ones((1,lm_total_num))
                for i,j in enumerate(lm_data):
                    tmp = j.split(' ')
                    tmp_lms[i,0] = str(tmp[0]).rstrip()
                    tmp_lms[i,1] = str(tmp[1]).rstrip()
                    if int(tmp_lms[i,0]) == 0 and int(tmp_lms[i,1]) == 0:
                        tmp_vis[:,i] = 0
                tmp_lms = np.reshape(tmp_lms,lm_total_num*2)
                tmp_lms = tmp_lms[np.newaxis, ...]
            elif ext == 'TXT':
                lm_data = pd.read_csv(lm_path, delimiter='\t', header=None).values
                tmp_vis = np.ones((1,lm_total_num))
                for i,j in enumerate(lm_data):
                    if lm_data[i,0] == 0 and lm_data[i,1] == 0:
                        tmp_vis[:,i] = 0   
                tmp_lms = np.reshape(lm_data,lm_total_num*2)
                tmp_lms = tmp_lms[np.newaxis, ...]          
        elif 'AFLW' in processing:
            lm_data = pd.read_csv(lm_path, delimiter='\t', header=None).values   
            tmp_lms = np.zeros((len(lm_data),lm_total_num*2))
            tmp_vis = np.ones((len(lm_data),lm_total_num))
            for i,lms in enumerate(lm_data):
                lm_index = 0
                tmp_lm = np.zeros((lm_total_num,2))
                for j in range(0,lm_total_num*2,2):
                    #lm += "{} ".format(str(i[j]))
                    tmp_lm[lm_index,0] = str(lms[j])
                    tmp_lm[lm_index,1] = str(lms[j+1])
                    if int(tmp_lm[lm_index,0]) == 0 and int(tmp_lm[lm_index,1]) == 0:
                        tmp_vis[i][lm_index] = 0
                    lm_index = lm_index + 1
                tmp_lm = np.reshape(tmp_lm,lm_total_num*2)
                tmp_lms[i] = tmp_lm[np.newaxis, ...]
            
            
    else:
        has_gt = False
        tmp_lms = np.zeros((lm_total_num,2))
        tmp_vis = np.zeros((1,lm_total_num))
        tmp_lms = np.reshape(tmp_lms,lm_total_num*2)
        tmp_lms = tmp_lms[np.newaxis, ...]
        print('LM File Not Found Check this out maybe!!!!')
    return tmp_lms, tmp_vis, has_gt

def writeLMfile(ds, lm_name, anno_dir, ext, ind, processing):
    
    if 'IBUG' in processing:
        lm_total_num = 68
    elif 'AFLW' in processing: 
        lm_total_num = 21
        
    outfile = create_annotation_file('GT_lm_file',anno_dir, mode='a')
    for i, j in enumerate(lm_name):
        if j =='images':
            lm_name[i] = 'gt_lm'
    
    lm_name[-1] = lm_name[-1][:-3] + ext
    lm_path = "/".join(lm_name) 
    
    if os.path.isfile(lm_path):
        if ext == 't7':
            lm_data = load_lua(lm_path,long_size=8)
            lm_data = lm_data.data.numpy()
            lm_data = np.reshape(lm_data, lm_total_num*2)
            lm_data = lm_data[..., np.newaxis]
            lm_data = lm_data.T
        #lm_data = np.array2string(lm_data, precision=1, separator=' ')
        #lm_data = lm_data[2:-1]        
        elif 'IBUG' in processing:
            #Using file as Pandas has issue with double space sused in some files
            f = open(lm_path, 'r')
            lines = f.readlines()
            f.close()
            lm_data = lines[3:71]
            lm = ''
            for i in lm_data:
                tmp = i.split(' ')
                tmp[0] = str(tmp[0]).rstrip()
                tmp[1] = str(tmp[1]).rstrip()
                lm += "{} ".format(str(tmp[0])) 
                lm += "{} ".format(str(tmp[1]))
            lm += ";"   
            lm_line = "{}\t{}\n".format(ind, lm)
            #lm_data = pd.read_csv(lm_path, delimiter=' ', header=None).values
            #lm_data = lm_data[3:-1]
        elif 'AFLW' in processing:
            lm_data = pd.read_csv(lm_path, delimiter='\t', header=None).values   
            lm = ''
            for i in lm_data:
                for j in range(0,lm_total_num*2):
                    lm += "{} ".format(str(i[j])) 
                lm += ";"
            lm_line = "{}\t{}\n".format(ind, lm)
            
    else:
    #There are two potential things here one is we dont see the fiel when we should and the other is for datasets where all are not
        
        print('LM File Not Found Check this out!!!!')
    
    outfile.write(lm_line)
    outfile.close()
    return

  
def writeExtraLabelsfilePalsy(ds, name, anno_dir, ind):
    
    if "CKGenPalsy" in ds:
        writePalsyfile(ds, name, anno_dir, ind)
    
    return

def writePalsyfile(ds, name, anno_dir, ind):
    
    outfile = create_annotation_file(ds + '_multi_file',anno_dir, mode='a')
    name[1] = 'gt_palsy'
    name[-1] = name[-1][:-3] + 'txt'
    path = "/".join(name) 
    
    data = pd.read_csv(path, delimiter='\t', header=None).values
    for i in data:
        labels = "{} {} {};".format(str(i[0]),str(i[1]),str(i[2])) 
    
    line = "{}\t{}\n".format(ind, labels)
 
    outfile.write(line)
    outfile.close()
    return


def splitCreator(config, labels, method, video=False):
    
    labels_file = config_dir + 'labels.json'
    all_labels = loadJSON(labels_file)
    save_dir = 'D:/Research/Experiments/Base/Datasets/'
    dataset_name = config['DATASET']

    if video:
        save_dir  = save_dir  + dataset_name + '/dataloader/' + config['SUB_DATASET'] + '/video/'
    else:
        save_dir  = save_dir  + dataset_name + '/dataloader/' + config['SUB_DATASET'] + '/image/'
            
    if method == "LOSO":
        if video:
            label_path = save_dir + all_labels['SUBJECT']['LABEL']['SAVENAME'] 
            subject_labels = np.load(label_path)
        
        subject_set = set(subject_labels)
        subject_set = sorted(subject_set)
        subject_text_file = save_dir + "SUBJECT_REFERENCE_LIST.txt"
        with open(subject_text_file, 'w') as f:
            for i, item in enumerate(subject_set):
                f.write("%s : %s\n" % (str(i),item))
        
        train_list = [None] * len(subject_set)  
        test_list = [None] * len(subject_set) 
        
        for i,sub_id in enumerate(subject_set):
            train_list[i], test_list[i] = LOSO(sub_id, np.asarray(subject_labels))
            
        savename = save_dir + 'LOSO_TRAIN.npy'
        with open(savename, "wb") as fp:   #Pickling
            pickle.dump(train_list, fp)
        savename = save_dir + 'LOSO_TEST.npy'
        with open(savename, "wb") as fp:   #Pickling
            pickle.dump(test_list, fp)
            #Save these splits
    
#    #Basic Random Split
#    validation_split = .2
#    # Creating data indices for training and validation splits:
#    dataset_size = len(images)
#    indices = list(range(dataset_size))
#    split = int(np.floor(validation_split * dataset_size))
#    train_indices, val_indices = indices[split:], indices[:split]
        
    
    return


def LOSO( subject_id, subject_data):
    train_ind = subject_data != subject_id
    test_ind = subject_data == subject_id
     
    return train_ind, test_ind
    
#def __LOVO__(self,index,length):
#         train_ind = index 
#         test_ind = np.arange(length)
#         test_ind = test_ind[np.arange(len(test_ind))!=index]
#        
#         return train_ind, test_ind
#     
def KFold__(dataset_size):
        #batch_size = 16
        validation_split = .1
        shuffle_dataset = True
        random_seed= 42
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        
        return train_indices, val_indices



if __name__ == '__main__':

    #Use this line to choose the dataset to create annotations for
    config_file = 'AFW_Original_images.json' #'CK+_Data.json'     #'Hybrid/FaceAction_Data.json'
    dataset_cfg = create_anno_numpy(config_file)
    #config_file = 'D:/Research/Code/DL_FW/CustomDataTools/configs/FacialPalsy_Intelli_data.json'

    #splitCreator(config, ["PALSY_LEVELS"], method="KFOLD", video=True)
#    ds = 16
#    data = pd.read_csv('D:/Research/Code/DL_FW/CustomDataTools/configs/Dataset_Details.txt', delimiter='\t', header=None)
#    
#    if ds is not None:
#        data_row = ds
#        ds_data = data.loc[[data_row]]
#        dataset_cfg = createConfig(ds_data)
#    
#        save_name = 'D:/Research/Code/DL_FW/CustomDataTools/configs/' + dataset_cfg['DATA']['DATASET'] + '_' + dataset_cfg['DATA']['SUB_DATASET'] + '.cfg'
#        with open(save_name, "wb") as fp:   #Pickling
#            pickle.dump(dataset_cfg, fp)  
#        create_anno_numpy(dataset_cfg)
#        
#    else:
#        for i in range(1,len(data)):
#    
#            data_row = i
#            ds_data = data.loc[[data_row]]
#            dataset_cfg = createConfig(ds_data)
#    
#            save_name = 'D:/Research/Code/DL_FW/CustomDataTools/configs/' + dataset_cfg['DATA']['DATASET'] + '_' + dataset_cfg['DATA']['SUB_DATASET'] + '.cfg'
#            with open(save_name, "wb") as fp:   #Pickling
#                pickle.dump(dataset_cfg, fp)  
#            create_anno_numpy(dataset_cfg)
    


    
    
    
    

    
    


