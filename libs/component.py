# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 13:11:26 2019

@author: Gary
"""
import os
import pickle
import numpy as np
from libs.dataset.customDataset import DatasetInformation
import copy
import json


class Component():
    def __init__(self, cfg, dataset, pred_dic_keys, training, pipeline_input, pipeline_output, video):

        self.dataset_name = dataset
        self.video = video
        self.__getModelConfigNames__(cfg, training)

        with open(self.base_config_file) as f:
            comp_cfg = json.load(f)

        if self.architecture == 'Ground_Truth':
            self.use_gt_data = True
        else:
            self.use_gt_data = False
        
        if 'EXTERNAL' in comp_cfg:
            self.external_process = comp_cfg["EXTERNAL"]
        else:
            self.external_process = False
        
        self.override = cfg["OVERRIDE"]
        
        self.pipeline_input = pipeline_input#["ARCH"]
        self.pipeline_output = pipeline_output
        self.training = training
        
        if 'ANNO' in comp_cfg:
            self.annotation = comp_cfg['ANNO']
    
        #If there is no key it means the model is either a framework spefic or external and therefore exists
        if 'MODEL_PATH' in comp_cfg['PARAMETERS']['MODEL']:
            self.model_path = comp_cfg['PARAMETERS']['MODEL']['MODEL_PATH']
            self.check_model_status = False
        else:
            self.check_model_status = True
        
        self.parameters = self.__setParameters__(cfg["COMPONENT"]['PARAMS'],comp_cfg['PARAMETERS'])

        self.predictions_file_name = self.save_model_name[:-4] + '_predictions.txt'
        if not os.path.exists(self.paths['OUTPUT']):
            os.makedirs(self.paths['OUTPUT'])

        #self.metrics_file_name = self.base_path + cfg["SAVE_NAME"][:-4] + '_metrics.txt'
        
        if 'PROTOCOL_NAME' in cfg["COMPONENT"]:
            self.protocol = cfg["COMPONENT"]["PROTOCOL_NAME"]
            self.protocol_ind = cfg["COMPONENT"]["PROTOCOL_INDEX"][cfg["PROTOCOL_INDEX"]]
            self.protocols = True
            self.paths['METRIC'] =  self.paths['METRIC'] + "/" + str(self.protocol_ind)
        else:
            self.protocols = False
        
        if not self.external_process:
            self.__setLibs__()
            
     #   self.predictions = self.__setPredictionsDictionary__(pred_dic_keys)
    
    def run(self, data=None): 
        
        if data is not None:
            for key in self.expected_input:
                if key in data:
                    data = data[key]
                    break #We want to stop at the first type so list must be priortiy ordered 
        
        self.dataset = DatasetInformation(self.dataset_name, self.paths["DATA"], required_labels=self.labels, videos=self.video, protocols=self.protocols, external_data=data)
        
        if self.protocols:
            self.dataset.__filterIndex__(self.protocol_ind,self.protocol,self.training)
        
        if self.training:
            
            self.parameters['MODEL']['NUM_FINE_TUNE_CLASSES'] = len(self.dataset.labels_legend[self.labels[0]]['RANGE'])
            self.model_interface = self.model(self.architecture, self.parameters, train=self.training)
            self.predictions = self.model_interface.train(self.dataset, self.save_config_file)
            
            if not self.__checkModelExists__(self.save_config_file):
                self.model_interface = self.model(self.architecture, self.parameters, train=self.training)
                self.predictions = self.model_interface.train(self.dataset, self.save_config_file)
        else:
            if self.checkPredictionsExist() and not self.override:
                self.predictions = self.loadPredictions()
            else:
                self.model_interface = self.model(self.architecture, self.parameters, train=self.training)
                self.predictions = self.model_interface.eval(self.dataset)
                self.model_interface.visualise(self.dataset, self.predictions, self.metrics_path)
                self.savePredictions()
                #self.model_interface.metrics(self.dataset, self.predictions)
                
        if self.pipeline_output:
           return self.predictions
        else:
            return

    def __setPaths__(self, paths):

        self.paths = paths
        self.paths["MODEL_BASE"] = self.paths ["MODEL"] + self.type + "/" + self.model + "/"
        self.paths["MODEL_SAVE"] = self.paths ["MODEL"] + self.type + "/" + self.task + "/"
        self.paths["OUTPUT"] = self.paths["OUTPUT"] + self.dataset_name + '/' +  self.type + '/' + self.task + '/' + self.method
        self.paths["METRIC"] = self.paths["METRIC"] + self.dataset_name + '/' +  self.type + '/' + self.task + '/' + self.method
        return

    def __getModelConfigNames__(self, cfg, train):
          
        self.method = cfg["COMPONENT"]['METHOD']
        self.architecture = cfg["COMPONENT"]['ARCH']
        self.task =  cfg['TASK']
        self.type = cfg["COMPONENT"]['TYPE']
        self.labels = cfg["COMPONENT"]['LABELS']
        
        if train:
            self.model = "BASE"
        else:
            self.model = self.task

        self.__setPaths__(cfg["PATHS"])

        if cfg["COMPONENT"]["PARAMS"]:
            params = cfg["COMPONENT"]['PARAMS']
            param_string = ''
            for p in sorted(params.keys()):
                if param_string == '':
                    param_string = str(params[p])
                else:
                    param_string = param_string + '_'+ str(params[p])
        else:
            param_string = ''
             
        if "PROTOCOL_NAME" in cfg["COMPONENT"]:
            protocol_name = cfg["COMPONENT"]["PROTOCOL_NAME"]
            protocol_data = cfg["COMPONENT"]["PROTOCOL_INDEX"][cfg["PROTOCOL_INDEX"]]
            protocol_string = protocol_name + '_' + str(protocol_data)
            
        if cfg["COMPONENT"]["PARAMS"] and "PROTOCOL_NAME" in cfg["COMPONENT"]:
            self.save_model_name = self.architecture + '_' + protocol_string + '_p_' + param_string + '.pth'
        elif cfg["COMPONENT"]["PARAMS"]:
            self.save_model_name = self.architecture + '_p_' + param_string + '.pth'
        elif "PROTOCOL_NAME" in cfg["COMPONENT"]:
            self.save_model_name = self.architecture + '_' + protocol_string + '.pth'
        else:
            self.save_model_name = self.architecture + '.pth'
        
        if train:
            self.base_config_file = self.paths["MODEL_BASE"] +  self.method + '/' + self.architecture + '.json'
            self.save_config_file = self.paths["MODEL_SAVE"]  +  self.method + '/' + self.save_model_name[:-4] + '.json'
        else:
            self.base_config_file = self.paths["MODEL_BASE"] +  self.method + '/' + self.save_model_name[:-4] + '.json'
            
        return
    
        
    def __setParameters__(self,new,base):
        for p in sorted(new.keys()):
            base[p] =  new[p]
        
        #base['MODEL']['NUM_FINE_TUNE_CLASSES'] = classes
        
        return base
    
    def __setPredictionsDictionary__(self, pred_dic_keys):
        
        if self.checkPredictionsExist() and not self.override:
            predictions = self.loadPredictions()
            #self.export()
        elif self.external_process:
            #Need to exit the code or check for a file to convert
            print("External Need and link to import these")
        else:
            predictions = {}
            for key in pred_dic_keys:
                predictions[key] = None

        return predictions

    def __checkModelExists__(self,model_name):
        print(model_name)
        print(os.path.isfile(model_name))
        return os.path.isfile(model_name)

    def checkPredictionsExist(self):
        exists = False
        if os.path.isfile(self.preds_path + '/' + self.predictions_file_name):
            exists = True
        return exists
    
    def savePredictions(self):
        with open(self.preds_path + '/' + self.predictions_file_name, "wb") as fp:   #Pickling
            pickle.dump(self.predictions, fp)    
        return

    def loadPredictions(self):
        with open(self.preds_path + '/' + self.predictions_file_name, "rb") as fp:   # Unpickling
            predictions = pickle.load(fp)
        return predictions
    

class ActionDetection(Component):
    def __init__(self, cfg, dataset, training=False,pipeline_input=None, pipeline_output=False):
        
        pred_dic_keys = ["PREDS","GT" ,"SCORES"]
        video = True
        
        super().__init__(cfg, dataset, pred_dic_keys, training ,pipeline_input, pipeline_output, video)        
        """
        Args:
            cfg (string): the dataloader object being used
            """
        self.expected_input = ["BBOXES","LM"]
        
#        if self.training:    
#            self.model_save_name = cfg["TRAIN_MODEL_SAVE_PATH"] + cfg["SAVE_NAME"]
#            self.config_save_name = cfg["CONFIG_SAVE_PATH"] + cfg["SAVE_NAME"][:-3] + "json"
#            self.log_save_path = cfg["TRAIN_LOG_SAVE_PATH"]
#            self.__createEvalConfig__()
#            self.runner = Trainer(self.method,self.parameters,self.generate_model,self.model_save_name, self.config_save_name, self.log_save_path)
            
    def __setLibs__(self):
        
        if self.method == "3DCNN":
            from libs.architectures.AC.videoCNN.interface import Model
        self.model = Model
    
        return
    
    def __createEvalConfig__(self):
        eval_params = {}
        eval_params['PARAMETERS'] = copy.deepcopy(self.parameters)
        #eval_params['NAME'] = self.research_name
        eval_params['TYPE'] = "AC"
        eval_params['METHOD'] = self.method
        eval_params['ARCH'] = self.architecture
        #eval_params["DATASETS"] = self.dataset_name
        eval_params['PARAMETERS']['MODEL']['MODEL_PATH'] = self.model_save_name
        eval_params['PARAMETERS']['TRAIN'] = False
        eval_params['PARAMETERS']['MODEL']['NUM_CLASSES'] = self.parameters['MODEL']['NUM_FINE_TUNE_CLASSES']
        
        file_name = self.config_save_name
        with open(file_name, 'w') as outfile:
            json.dump(eval_params, outfile)
        
        return
    
    
class ObjectDetection(Component):
    def __init__(self, cfg, dataset, training=False, pipeline_input=None, pipeline_output=False):
        pred_dic_keys = ["BBOXES","LABELS" ,"SCORES"]
        video = False
        super().__init__(cfg, dataset, pred_dic_keys, training ,pipeline_input, pipeline_output, video) 
        """
        Args:
            cfg (dic): With model and method details
            path (string): Path to results data
            
            """

    def evaluate(self):
#        if os.path.isfile(self.metrics_path + '/' + self.metrics_file_name):
#            with open(self.metrics_path + '/' + self.metrics_file_name, "rb") as fp:   # Unpickling
#                self.metrics = pickle.load(fp)
#        else:
        if self.dataset_info.gt_bbox:
            self.__getVOCEvaluation__()
        else:
            self.metrics = None
            
    
    def __setLibs__(self):
        
        if self.method == "Simple_Faster_RCNN":
            from libs.architectures.OD.Simple_Faster_RCNN.interface import Model
            self.model = Model

        return
        
        
    def __getVOCEvaluation__(self):
        from object_detection.simple_faster_rcnn_pytorch_gs.utils.eval_tool import eval_detection_voc
        
        self.bboxes = self.__padBbox__(self.bboxes)
        self.metrics = eval_detection_voc(
            self.bboxes, self.labels, self.scores,
            self.dataset_info.gt_bbox_data, self.dataset_info.gt_bbox_label,
            use_07_metric=True)

        with open(self.metrics_path  + '/' + self.metrics_file_name, "wb") as fp:   #Pickling
            pickle.dump(self.metrics, fp)
            
        return  

    def  __padBbox__(self, bboxes):
        
        for ii, bbox in enumerate(bboxes):
            for jj in range(len(bbox)):
                if "AFW" in self.dataset_info.name and ii != 46:
                    height = (bbox[jj][2] - bbox[jj][0]) / 100 * 30
                    bbox[jj][0] = bbox[jj][0] + height
                elif "FDDB" in self.dataset_info.name:
                    height = (bbox[jj][2] - bbox[jj][0]) / 100 * 10
                    width = (bbox[jj][3] - bbox[jj][1]) / 100 * 10
                    bbox[jj][0] = bbox[jj][0] - height 
                    bbox[jj][2] = bbox[jj][2] + height 
                    bbox[jj][1] = bbox[jj][1] - width 
                    bbox[jj][3] = bbox[jj][3] + width 
                elif "AFLW" in self.dataset_info.name:
                    height = (bbox[jj][2] - bbox[jj][0]) / 100 * 20
                    width = (bbox[jj][3] - bbox[jj][1]) / 100 * 20
                    #bbox[0][i][0] = bbox[0][i][0] + height 
                    #bbox[0][i][2] = bbox[0][i][2] + height 
                    bbox[jj][1] = bbox[jj][1] - width 
                    bbox[jj][3] = bbox[jj][3] + width     
        return bboxes
     
    def saveGTData(self, path, box, label):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + '/' + self.box_file_name, "wb") as fp:   #Pickling
            pickle.dump(box, fp)    
        with open(path + '/' + self.label_file_name, "wb") as fp:   #Pickling
            pickle.dump(label, fp)
        return 
    
    def export(self):
        #Export the bbox detection to matlab
        FrameStack = np.empty((len(self.bboxes),), dtype=np.object)
        for i in range(len(self.bboxes)):
            FrameStack[i] = self.bboxes[i]
        
        import scipy.io as sio
        fn = self.base_path + '/' + self.name + '_bboxes.mat'
        #sio.savemat(fn, {'boxes': self.bboxes})
        sio.savemat(fn, {'boxes': FrameStack})
        return
    
    
class LandmarkLocalisation(Component):
    #def __init__(self, cfg, dataset, detection_method=None, external=None, protocol_ind=None,protocol_type=None):
    def __init__(self, cfg, dataset, training=False,pipeline_input=None, pipeline_output=False):
        pred_dic_keys = ["LM","UNSCALED_LM" ,"SCORES"]
        video = False
        super().__init__(cfg, dataset, pred_dic_keys, training ,pipeline_input, pipeline_output, video) 
        """
        Args:
            cfg (string): the dataloader object being used
            """
            
        self.expected_input = ["BBOXES"]
#        if self.pipeline_input is not None:    
#            self.landmarks_file_name = self.pipeline_input + '+' + self.name + '_landmarks.txt'
#            self.unscaled_landmarks_file_name= self.pipeline_input + '+' + self.name + '_unscaled_landmarks.txt'
#            self.heatmaps_score_file_name = self.pipeline_input + '+' + self.name + '_heatmaps_scores.txt'
#        else:
#            self.landmarks_file_name = self.name + '_landmarks.txt'
#            self.unscaled_landmarks_file_name= self.name + '_unscaled_landmarks.txt'
#            self.heatmaps_score_file_name = self.name + '_heatmaps_scores.txt'
#        
#        self.base_path = base_path + 'LandmarkLocalisation/' + self.method
#        self.metrics_path = metrics_path + 'LandmarkLocalisation/' + self.method
#        if not os.path.exists(self.metrics_path):
#            os.makedirs(self.metrics_path)
#            
            
#        if self.checkPredictionsExist(self.base_path):
#            with open(self.base_path + '/' + self.landmarks_file_name, "rb") as fp:   # Unpickling
#                self.landmark = pickle.load(fp)  
#        elif self.external:
#            self.landmarks =  self.convertData()
#        else:
#            self.landmarks, self.heatmap_scores, self.unscaled_landmarks = None, None, None
    
    
#    def run(self, data=None):
#        
#        self.model_interface = self.model(self.architecture, self.parameters, train=self.training)
#        self.dataset = DatasetInformation(self.dataset_name, required_labels=self.labels, videos=self.video, protocols=self.protocols)
#        
#        if self.training:
#            if not self.__checkModelExists__(self.model_save_name):
#                self.train_model(self.dataset)
#        else:
#            if self.checkPredictionsExist() and not self.override:
#                self.predictions = self.loadPredictions()
#            else:
#                self.predictions["LM"], self.predictions["UNSCALED_LM"], self.predictions["SCORES"] = self.model_interface.eval(self.dataset,data["BBOXES"])
#                self.savePredictions()
#                
#        if self.pipeline_output:
#           return self.predictions
#        else:
#            return
    
#        elif self.external_method:
#            self.landmarks =  self.convertData()
#        else:
#        if bboxes is None:
#                bboxes = self.dataset_info #Get the frames 
#            self.__getLandmarkLocalisations__(bboxes)
#            self.saveData(self.base_path)
#        
#        if self.pipeline_output:
#            return self.landmarks
#        else:
#            return 
        
    def __setLibs__(self):
        
        if self.method == "FAN":
            from libs.architectures.LL.FAN.interface import Model
            self.model = Model

        return
    
    
    def runBoxFromExternal(self, bboxes):
        
        if self.checkPredictionsExist(self.base_path):
            with open(self.base_path + '/' + self.landmarks_file_name, "rb") as fp:   # Unpickling
                self.landmarks = pickle.load(fp)  
        elif self.external_method:
            self.landmarks =  self.convertData()
        else:
            self.__getLandmarkLocalisations__(bboxes)
            self.saveData(self.base_path)
        #self.cropToLandmarks()
        return self
    

    def evaluate(self):
        if  'FACE_LM' in self.dataset_info.labels:
            from libs.landmarkEvaluation import eval_landmarks_nme
            #fro dataset get lms as boxes
            
            pred_box, pred_label, pred_score = self.dataset_info.__landmarksToBoxes__(self.landmarks)
            self.landmarks = self.remapLandamrks()
            self.metrics = eval_landmarks_nme(self.landmarks, pred_box, pred_label, pred_score, self.dataset_info.labels['FACE_LM']['LM'], self.dataset_info.labels['FACE_LM']['VISIBILTY'],
                                              self.dataset_info.labels['FACE_BBOX']['BBOX'], self.dataset_info.labels['FACE_BBOX']['LABEL'], self.dataset_info.labels['FACE_LM']['INDEX'])
        else:
            self.metrics = None
    
    def __getLandmarkLocalisations__(self, bboxes):
        if not self.use_gt_data:
            lm = LandmarkLocalisation(metrics=False)
            lm.__loadMethodModel__()
            dataloader = getDataloader(self.dataset_info, self.method, 1, bbox_data=bboxes)
            self.landmarks, self.heatmap_scores, self.unscaled_landmarks = lm.__getPredictionsWithMetrics__(dataloader,bboxes=bboxes)
            self.landmarks = dataloader.dataset.__singleListToImageList__(self.landmarks)
            self.heatmap_scores = dataloader.dataset.__singleListToImageList__(self.heatmap_scores)
            self.unscaled_landmarks = dataloader.dataset.__singleListToImageList__(self.unscaled_landmarks)
            del lm
        else:
            self.landmarks = self.dataset_info.gt_landmark_data
            self.heatmap_scores = None
            self.unscaled_landmarks = None
        return

    
    def convertData(self):
        #Converts dat for externally produced data currently from matlab codes to the correct python format
        import scipy.io as sio
        #import h5py
        fn = self.base_path + '/' + self.face_detection_base + '+' + self.name + '_landmarks.mat'
        #matlab = h5py.File(fn)
        matlab = sio.loadmat(fn)
        data = matlab['data']
        landmarks = list()
        for i in range(data.shape[1]):
            lms = data[0,i][0]
            landmarks.append(np.reshape(lms, (1,lms.shape[0]*2)))
        with open(self.base_path + '/' + self.landmarks_file_name, "wb") as fp:   #Pickling
            pickle.dump(landmarks, fp) 
        return landmarks
    
    def remapLandamrks(self):
        #Method to arrange landmarks from there output annotation type to the datasets type if required for evaluation purposes
        if self.annotation != self.dataset_info.labels['FACE_LM']['ANNO']:
            with open('D:/Research/Code/DL_FW/configs/evaluation/landmarkMapping/' + self.dataset_info.labels['FACE_LM']['ANNO'] + '.cfg' , "rb") as fp:   # Unpickling
                mapping = pickle.load(fp)
            remap_lms = list()
            for i in range(len(self.landmarks)):
                for j, lms in enumerate(self.landmarks[i]):
                    if lms.shape[0] != 0:
                        tmp_source = lms.reshape(int(len(lms)/2),2)
                        tmp_target = np.zeros((len(mapping['INDEX']),2))
                        for k in range(len(mapping['INDEX'])):
                            if mapping[self.annotation][k] != 0:
                                #print(i,j,k,mapping[self.annotation][k])
                                tmp_target[mapping['INDEX'][k]-1] = tmp_source[mapping[self.annotation][k]-1]
                        remap_lms.append(np.reshape(tmp_target, (1,tmp_target.shape[0]*2)))
                            #key values are k and mapping[self.annotation][k]
#                        if mapping['INDEX'][k] == mapping[self.annotation][k]:
#                            tmp_target[k] = tmp_source[k]
#                        elif mapping[self.annotation][k] == 0:
#                            print(k)
#                        else:
#                           tmp_target[mapping[self.annotation][k]] = tmp_source[k] 
                    else:
                        remap_lms.append(lms) 
            return remap_lms
        return self.landmarks
    
    
#    def cropToLandmarks(self):
#        
#        for i,image_path in enumerate(self.dataset_info.image_locations):
#            cropper = visualizeResults()
#            cropper.cropImageFromLandmarks(image_path, self.landmarks[i])
#        
#        return