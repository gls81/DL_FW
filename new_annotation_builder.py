#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 11:20:57 2019

@author: gary
"""
import os
import json
import sys
import glob
import numpy as np
import pandas as pd
import scipy.io as sio
import shutil
import pickle


pwd = os.getcwd()
datasets_dir = os.path.join(pwd,"datasets")
configs_dir = os.path.join(pwd,"configs","datasets")
labels_file = os.path.join(pwd,"configs",'labels.json')


def loadJSON(file):
    with open(file) as f:
        data = json.load(f)
    
    return data

def saveJSON(data, filename):
    with open(filename, 'w') as fp:
        json.dump(data, fp)

    return

def creatJSONConfigFile(dataset,name,labels, sub_dirs):
    config = {
            "DATASET": dataset,
            "SUB_DATASET": name,
            "HYBRID" : False,
            "SUB_DIRS": sub_dirs,
            "LABELS":{}
    }

    config["LABELS"] = labels

    return config

def getLabelData(gt_labels,labels):

    data = {}

    for label in labels:
        for gt_label in gt_labels:
            if label == gt_labels[gt_label]["PATH"]:
                data[gt_label] = { "EXT" : "txt" }

    return data


def checkSubDir(path):
    sub = False
    folder = os.listdir(path)
    for file in folder:
        file_path =  os.path.join(path,file)
        if os.path.isdir(file_path):
            sub = True

    return sub

def checkLabels(a,b):
    flag = False
    for label in a:
        if label in b:
            flag = True
        else:
            return False
    
    return flag

def getSearchString(sub):
    if sub:
        ss = '/**/*'
    else:
        ss = '/*'
    return ss

def getImageList(images_dir, st, ext_list):

    image_list = []
        
    for ext in ext_list: 
        image_search_path = images_dir + st + ext
        image_list += glob.glob(image_search_path,recursive=True)

    return image_list

def singleLabel(ds, file_name, ext, ind, folder):
    file_name = file_name.replace("/images/", folder)
    file_path = file_name[:-3] + ext  
    file_data = pd.read_csv(file_path, delimiter='\t', header=None).values
    label = file_data[0][0]

    return label

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

def videoExtender(video):
    #Takes a set so video frames reverses the list and concatenates. The purpose is to extend seuqnces like the CK+ which only have from nuetral to full expression, extending mirrors and bring them back to neutral
    copy = video.copy()
    copy.reverse()
    extended = video + copy  
    
    return extended

def writeBBOXlist(ds, bbox_name, ext, ind , target_dir):
    
    bbox_name = bbox_name.replace("/images/", "/" + target_dir + "/")
    bbox_path = bbox_name[:-3] + ext  
    bbox_data = pd.read_csv(bbox_path, delimiter='\t', header=None).values
    bbox = np.zeros((len(bbox_data),4))
    label = np.zeros((len(bbox_data),), dtype=np.int)
    scores = np.ones((len(bbox_data),))
    for i,j in enumerate(bbox_data):
        bbox[i,:] = [str(j[0]),str(j[1]),str(j[2]),str(j[3])] 
        label[i] = str(j[4])
    return bbox, label, scores

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


def processBBOX(image_anno, ext, label_info, anno_image_dir, anno_video_dir=None, video=False, video_index=None):

        bbox_list = [None] * len(image_anno)
        bbox_label = [None] * len(image_anno)
        bbox_scores = [None] * len(image_anno) #Note this is more for trouble shooting all scores are 1 for gt

        for i, name in enumerate(image_anno):
            bbox_list[i], bbox_label[i], bbox_scores[i] = writeBBOXlist(dataset_name, name, ext, i, label_info["PATH"])
        savename = os.path.join(anno_image_dir, label_info["BBOX"]["SAVENAME"])
        with open(savename, "wb") as fp:   #Pickling
            pickle.dump(bbox_list, fp)
        savename = os.path.join(anno_image_dir, label_info["LABEL"]["SAVENAME"])
        with open(savename, "wb") as fp:   #Pickling
            pickle.dump(bbox_label, fp) 
        savename = os.path.join(anno_image_dir, label_info["SCORE"]["SAVENAME"])
        with open(savename, "wb") as fp:   #Pickling
            pickle.dump(bbox_scores, fp)

def processLandmarks(image_anno, ext, label_info, anno_image_dir, anno_video_dir=None, video=False, video_index=None):
        landmark_list = [None] * len(image_anno)
        visibilty_list = [None] * len(image_anno)
        gt_list =  np.zeros((len(image_anno), 1), dtype=bool)#[None] * len(image_anno)
        ext = config['LABELS']['FACE_LM']['EXT']
        for i, name in enumerate(image_anno):
            #print(name)
            landmark_list[i], visibilty_list[i], gt_list[i] = writeLMlist(dataset_name, name, ext, config['LABELS']['FACE_LM']['ANNO'])
        savename = os.path.join(anno_image_dir, label_info["LM"]["SAVENAME"])
        with open(savename, "wb") as fp:   #Pickling
            pickle.dump(landmark_list, fp)
        savename = os.path.join(anno_image_dir, label_info["VISIBILTY"]["SAVENAME"])
        with open(savename, "wb") as fp:   #Pickling
            pickle.dump(visibilty_list, fp) 
        savename = os.path.join(anno_image_dir, label_info["INDEX"]["SAVENAME"])
        with open(savename, "wb") as fp:   #Pickling
            pickle.dump(gt_list, fp)
        savename = anno_image_dir + 'GT_LM_INDEX.mat'       
        sio.savemat(savename, {'gt_lms_index': gt_list})

def processSingleLabel(image_anno, ext, label_info, anno_image_dir, anno_video_dir=None, video=False, video_index=None):
    label_list = [None] * len(image_anno)
    for i, name in enumerate(image_anno):
        label_list[i] = singleLabel(dataset_name, name, ext, i, label_info["PATH"])
    savename = os.path.join(anno_image_dir, label_info["LABEL"]["SAVENAME"])
    with open(savename, "wb") as fp:   #Pickling
        pickle.dump(label_list, fp)
    if video:
        palsy_level_video = imageToVideoLabels(video_index, label_list, label=True)
        savename = os.path.join(anno_video_dir, label_info["LABEL"]["SAVENAME"])
        with open(savename, "wb") as fp:   #Pickling
            pickle.dump(palsy_level_video, fp)



def create_anno_numpy(config_file):

    label_funcs_map = {"OD": processBBOX, "LM": processLandmarks, "AD": processSingleLabel}
    config = loadJSON(config_file)
    labels = loadJSON(labels_file)

    #To deal with datasets combinations
    hybrid = config['HYBRID']
    ext = ['.jpg','.png']

    if 'VIDEO' in config['LABELS']:
        video = True
    else:
        video = False

    if hybrid:
        match_labels = False
        base_configs = config['PATH']
        dataset_name = config['DATASET']
        anno_image_dir = datasets_dir  + dataset_name + '/dataloader/' + config['SUB_DATASET'] + '/image/'
        anno_video_dir = datasets_dir  + dataset_name + '/dataloader/' + config['SUB_DATASET'] +  '/video/'
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
        base_dir = os.path.join(datasets_dir, config['DATASET'])
        dataset_name = config['SUB_DATASET']
        anno_image_dir = os.path.join(base_dir, 'dataloader' , config['SUB_DATASET'], 'image')
        anno_video_dir = os.path.join(base_dir, 'dataloader' , config['SUB_DATASET'], 'video')
        images_dir = os.path.join(base_dir, config['SUB_DATASET'], 'images' )
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

    # kf = KFold(n_splits=10)
    # for train, test in kf.split(image_anno):
    #     te = test
    #     tr = train
                       
    if video:
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

    for label in config['LABELS']:
        ty = labels[label]["TYPE"]
        if video:
            label_funcs_map[ty](image_anno,config['LABELS'][label]['EXT'], labels[label], anno_image_dir, anno_video_dir=anno_video_dir, video=video, video_index=video_index)
        else:
            label_funcs_map[ty](image_anno,config['LABELS'][label]['EXT'], labels[label], anno_image_dir)


    if video:
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


def create_config_files(dataset_name):
    excluded_sub_dirs = ["dataloader","Source"]
    labels = loadJSON(labels_file)
    config_files = []


    dataset_path = os.path.join(datasets_dir,dataset_name)
    if os.path.isdir(dataset_path):
        con_dir = os.path.join(configs_dir,dataset_name)
        if not os.path.isdir(con_dir):
            os.mkdir(con_dir)

        sub_entries = os.listdir(dataset_path)
        for sub in sub_entries:
            cur_sub_dir = os.path.join(dataset_path,sub)
            if os.path.isdir(cur_sub_dir) and sub not in excluded_sub_dirs:
                label_entries = os.listdir(cur_sub_dir)
                if "images" in label_entries:
                    config_save_name = sub + ".json"
                    config_save_path = os.path.join(con_dir,config_save_name)
                    images_dir = os.path.join(cur_sub_dir,"images")
                    sub_dirs = checkSubDir(images_dir)
                    label_entries = os.listdir(cur_sub_dir)
                    label_data = getLabelData(labels, label_entries)
                    config = creatJSONConfigFile(dataset_name,sub,label_data, sub_dirs)
                    saveJSON(config,config_save_path)
                    config_files.append(config_save_path)
                else:
                    sys.exit("There is no images folder in this dataset!!!")
    else:
        sys.exit("There is no existing dataset directory!!!")

    return config_files



if __name__ == '__main__':

    dataset_name = 'AFW'
    config_files = create_config_files(dataset_name)
    for config_file in config_files:
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




#excluded_datasets = ["Basel3D","Semaine","MUG","BP4D"] removed as decision to give a specific dataset rahter than build all


