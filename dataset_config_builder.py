#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 11:20:57 2019

@author: gary
"""

import os
import json

def loadJSON(file):
    with open(file) as f:
        data = json.load(f)
    
    return data

def creatJSONCOnfigFile(dataset,name,labels, sub_dirs):
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

pwd = os.getcwd()
datasets_dir = os.path.join(pwd,"datasets")
configs_dir = os.path.join(pwd,"configs","datasets")
labels_file = os.path.join(pwd,"configs",'labels.json')
labels = loadJSON(labels_file)

excluded_datasets = ["Basel3D","Semaine","MUG","BP4D"]
excluded_sub_dirs = ["dataloader","Source"]

entries = os.listdir(datasets_dir)
for entry in entries:
    cur_dir = os.path.join(datasets_dir,entry)
    if os.path.isdir(cur_dir) and entry not in excluded_datasets:
        con_dir = os.path.join(configs_dir,entry)
        if not os.path.isdir(con_dir):
            os.mkdir(con_dir)

        sub_entries = os.listdir(cur_dir)
        for sub in sub_entries:
            cur_sub_dir = os.path.join(cur_dir,sub)
            if os.path.isdir(cur_sub_dir) and sub not in excluded_sub_dirs:
                label_entries = os.listdir(cur_sub_dir)
                if "images" in label_entries:
                    images_dir = os.path.join(cur_sub_dir,"images")
                    sub_dirs = checkSubDir(images_dir)
                    label_entries = os.listdir(cur_sub_dir)
                    label_data = getLabelData(labels, label_entries)
                    creatJSONCOnfigFile(entry,sub,label_data, sub_dirs)

