# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 14:59:33 2019

@author: Gary
"""
from __future__ import absolute_import
from libs.ExperimentController import ExperimentController
import json
import platform
import os

os_type = platform.system()
pwd = os.getcwd()
if os_type == "Linux":
    os_path = "/home/gary/Research"
elif os_type == "Windows":
    os_path = "D:/"

experiment_path = pwd
#confg_path = experiment_path + "Configs/3DCNN/Resnet18"
#files = ["D:/Research/Experiments/Testing/Configs/newTestFD.json"] #Test for Action Pasly stuff
#files = [experiment_path + "IEEE_Access_Face_Action/Configs/test.json"]
file = pwd + "/test_config.json"
with open(file) as f:
    config = json.load(f)

#Add additionbal config for pathes
config["LABEL_INFO_PATH"] = pwd + "/config/data/labels.json"
config["OS_PATH"] = pwd
config["EVAL_PATH"] = experiment_path

trainer = ExperimentController(config)
trainer.run()
#trainer.visualize("LL")
#trainer.generateResults(singles=True)
#trainer.plotMultiExpResults([0,1])
#trainer.plotMultiExpResultsAgg([0,1],[2,3])