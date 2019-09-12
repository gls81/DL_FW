# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:46:07 2019

@author: Gary
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from libs.visualise import visualizeResults
import itertools
import os
import pickle
import json
from libs.component import ActionDetection, LandmarkLocalisation, ObjectDetection
#from CustomDataTools.CustomDataset import DatasetInformation
from torch.utils.data.dataset import Dataset
#from libs.data.dataset_helpers import getSkiimage
from torch.utils import data as data_
from tqdm import tqdm

class ExperimentController():
    def __init__(self, config):
        """
        Args:
            cfg (string): the dataloader object being used
            
            """      
        self.name = config['NAME']
        self.type = config['TYPE']

        self.paths = {}
        self.paths["BASE"] = config["OS_PATH"]
        self.paths["OUTPUT"] = config["OS_PATH"]  + "/output/"
        self.paths["MODEL"] = config["OS_PATH"]  + "/config/models/"
        self.paths["METRIC"] = config["OS_PATH"]  + "/exp/" + self.name + "/"
        self.paths["DATA"] = config["OS_PATH"]  + "/config/data/"
        #self.base_results_path = config["OS_PATH"]  + 'Research/Experiments/Base/Results/'
        #self.base_config_path = config["OS_PATH"]  + 'Research/Experiments/Base/Configs/'
        #self.expr_results_path = config["OS_PATH"]  + 'Research/Experiments/' + self.name

        with open(config["LABEL_INFO_PATH"]) as f:
            self.label_info = json.load(f) 
        
        if config["TRAIN"]:
            self.datasets = config["TRAIN_DATASET"]
            self.eval_datasets = config["EVAL_DATASETS"]
            self.train = True
        else:
            self.datasets = config["EVAL_DATASETS"]
            self.train = False
            
        #Experiment Options
        self.metrics_options = config["METRICS"]

        self.experiments  = self.__buildExperiments__(config)
        
        return
    
    def __buildExperiments__(self,config):
                   
        experiments = [None] * len(config["METHODS"])
        
        for ii,exp in enumerate(config["METHODS"]):
            experiment = {}
            exp["TASK"] = "-".join(exp["MAIN"]["LABELS"])

            if "PROTOCOL_INDEX" in exp['MAIN']:
                num_output_models = len(exp['MAIN']["PROTOCOL_INDEX"])
            else:
                num_output_models = 1

            for ds in self.datasets:
                experiment[ds] = {}
                #Add some experiment overview to use when producing outputs ie. CM and ROC
                experiment[ds]['TITLE'] = config["METHODS"][ii]["OUTPUT"]["NAME"]
                experiment[ds]['PATH'] = self.paths["METRIC"]  +  '/' +  exp["TASK"]

                if "PRE" in config["METHODS"][ii]:
                    exp["OVERRIDE"] = False
                    experiment[ds]["PRE_OUTPUT"] = [None] *  len(config["METHODS"][ii]["PRE"])
                    experiment[ds]['PRE'] = self.__getPipelineComponent__(exp,ds,pre=True,train=self.train)
                
                experiment[ds]['OUTPUT'] = [None] * num_output_models
                
                for mod in range(num_output_models):
                    #exp["OVERRIDE"] = self.override
                    exp["MODEL_INDEX"] = mod
                    experiment[ds]['OUTPUT'][mod] = self.__getPipelineComponent__(exp,ds, pre=False,train=self.train)
                        
                experiment[ds]["PREDICTIONS"] = None
                experiment[ds]["PREDICTIONS_PER_MODEL"] = [None] * num_output_models
                experiment[ds]["METRICS"] = [None]
                experiment[ds]["METRICS_PER_MODEL"] = [None] * num_output_models

            experiments[ii] = experiment
            
        return experiments
    
    
    def __getDatasetInfo__(self,config):
    
        dataset_info = [None] * len(config["METHODS"])
        
        for ii,exp in enumerate(config["METHODS"]):
            for ds in self.datasets:
                dataset_info[ii] = {}
                labels = []
                video = []
                if "PRE" in exp:
                    for jj,meth in exp["PRE"]:
                        vid = meth["VIDEO"]
                        for lab in meth["LABELS"]:
                            labels.append(lab)
                            video.append(vid)
                    
                vid = exp["MAIN"]["VIDEO"]
                for lab in exp["MAIN"]["LABELS"]:
                    labels.append(lab)
                    video.append(vid)
                
                dataset_info[ii][ds] =  DatasetInformation(ds,labels, videos=video, protocols=True) 
                
        return dataset_info
        
    
    def __getPipelineComponent__(self, config, dataset, pre=False,train=False):
        #base_output_path_ds =  self.base_output_path + dataset + '/'
        #expr_metrics_path_ds =   self.expr_metrics_path + "/" + dataset + '/'

        if pre:
            component = self.createPreProcessingComponent(config,dataset)
        else:
            component = self.createOutputComponent(config,dataset,train=train)
                
        return component
             
    def createOutputComponent(self, config, dataset, train):
         
        comp_config = {}
        comp_config["COMPONENT"] = config["MAIN"]
        comp_config["OVERRIDE"] = config["OVERRIDE"]
        comp_config["TASK"] = config["TASK"]
        comp_config["PATHS"] = self.paths

        if 'PROTOCOL_NAME' in config["MAIN"]:
            comp_config["PROTOCOL_INDEX"] = config["MODEL_INDEX"]
        
        component = self.getType(self.type)(comp_config, dataset, training=train)
        
        return component
    
    def createPreProcessingComponent(self, config, dataset):
        
        components = [None] * len(config["PRE"])
        last_stage_data = None
        
        for i,stage in enumerate(config["PRE"]):
            comp_config = {}
            comp_config["COMPONENT"] = stage
            comp_config["OVERRIDE"] = config["OVERRIDE"]
            comp_config["TASK"] = "-".join(stage["LABELS"])
            comp_config["PATHS"] = self.paths

            components[i] = self.getType(stage["TYPE"])(comp_config, dataset, pipeline_input=last_stage_data, pipeline_output=True)

            last_stage_data = comp_config

        return components
    

    def run(self, overide=False):
        for exp in self.experiments:
            for ds in exp:
                if self.createModelClasses(exp[ds]["PREDICTIONS"],overide):
              
                    if "PRE" in exp[ds]:
                        for i, stage in enumerate(exp[ds]['PRE']):
                            if i == 0:
                                output = stage.run()
                                exp[ds]["PRE_OUTPUT"][i] = output
                            else:
                                output = stage.run(output)
                                exp[ds]["PRE_OUTPUT"][i] = output
                
                        for i, stage in enumerate(exp[ds]['OUTPUT']):
                            stage.run(output)
                    else:
                        for i, stage in enumerate(exp[ds]['OUTPUT']):
                            stage.run()
                    if not self.train:
                        self.__getTotals__(exp, ds)
        return
    
    
    def createModelClasses(self, predictions, override):
        create = False
        if predictions is None:
            create = True
        if override:
            create = True
        return create
        
    def __checkModelExists__(self,model_name):
        print(model_name)
        print(os.path.isfile(model_name))
        return os.path.isfile(model_name)
              
    def getType(self, model_type):
        
        if model_type == "OD":
            return ObjectDetection
        elif model_type == "LL":
            return LandmarkLocalisation
        elif model_type == "AC":
            return ActionDetection
        elif model_type == "SS":
            return SematicSegmentation
        elif model_type == "IS":
            return InstanceSegmentation
        return

    def visualize(self, options):
        self.vis = visualizeResults() 
        self.sub_vis_path = self.expr_results_path + '/Visuals/' 
        if not os.path.exists(self.sub_vis_path):
            os.makedirs(self.sub_vis_path)
        
        for exp in self.experiments:
            for ds in exp:
                dataset = imageOnly(exp[ds]["OUTPUT"][0].dataset)
                dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=False, \
                                  # pin_memory=True,
                                  num_workers=0)
                
                #Get a key list of availble stuff 
                #key_dic = {}
                
                for out in exp[ds]["PRE_OUTPUT"]:
                    if "BBOXES" in out:
                        bboxes = out["BBOXES"]
                        bboxes_class = out["LABELS"]
                        
                
                if "LM" in exp[ds]["PREDICTIONS"]:
                        landmarks = exp[ds]["PREDICTIONS"]["LM"]
                        
                for ii, (img) in enumerate(dataloader):
                    print(ii)
                    if options == 'LL':
                        #save_name = self.getVisSaveDirectory('FaceDetection/', ii)
                        save_name = self.sub_vis_path + str(ii) + '.jpg'
                        self.vis.visualize_bbox(img, save_name, gt_data=None, pred_data=bboxes[ii].astype(int))
            
                        #self.vis.visualize_landmarks(img, save_name, pred_data=landmarks[ii], print_label=False)
                    
#                    if self.criteria == 'Landmark':             
#                save_name = self.getVisSaveDirectory('LandmarksLocalisation/' + self.Landmark_Localisation_Stage.method + '/', ii)
#                if self.dataset_info.palsy_level_label: #and self.dataset_info.gt_landmark_index_data[ii][0]
#                    
#                else:
#                    self.vis.visualize_landmarks(img, save_name,gt_data=None, pred_data=self.Landmark_Localisation_Stage.landmarks[ii], print_label=False)
#                #save_name = self.getVisSaveDirectory('FaceDetection/'+ self.Landmark_Localisation_Stage.method + '/', ii)
#                #self.vis.visualize_bbox(img, save_name, gt_data=None, pred_data=self.Face_Detection_Stage.bboxes[ii].astype(int))
#            

#                
#            if self.criteria == 'Palsy':             
#                save_name = self.getVisSaveDirectory('LandmarksLocalisation/' + self.Landmark_Localisation_Stage.method + '/', ii)
#                if self.dataset_info.palsy_level_label: #and self.dataset_info.gt_landmark_index_data[ii][0]
#                    self.vis.visualize_landmarks(img, save_name,gt_data=self.dataset_info.gt_landmark_data[ii], pred_data=self.Landmark_Localisation_Stage.landmarks[ii], print_label=False)
#                else:
#                    self.vis.visualize_landmarks(img, save_name,gt_data=None, pred_data=self.Landmark_Localisation_Stage.landmarks[ii], print_label=False)

                        
                        
    def getVisSaveDirectory(self, dir_name, image_ind):
        tmp_path = self.sub_vis_path + dir_name
        if self.dataset_info.video_frames:
            video_index = self.dataset_info.video_frames_data[image_ind]
            tmp_path = tmp_path + str(video_index) + '/'
            if not os.path.exists(tmp_path):
                os.makedirs(tmp_path)
            save_name = tmp_path + str(int(image_ind)) + '.jpg'
        else: #
            if not os.path.exists(tmp_path):
                os.makedirs(tmp_path)
            save_name = tmp_path + str(int(image_ind)) + '.jpg'
            
        return save_name
    
    def __getTotals__(self, exp, ds):
        #If we are using LOSO we have a n models we want to get the total predictions from this.
        
        totals = {}
        
        if len(exp[ds]['OUTPUT']) == 1:
            exp[ds]["PREDICTIONS"] = exp[ds]['OUTPUT'][0].predictions
            exp[ds]["PREDICTIONS_PER_MODEL"][0] = exp[ds]['OUTPUT'][0].predictions 
        else:
            for key in exp[ds]['OUTPUT'][0].predictions:
                totals[key] = []
                
            for ii, model in enumerate(exp[ds]['OUTPUT']):
                exp[ds]["PREDICTIONS_PER_MODEL"][ii] = model.predictions
                    
                for key in model.predictions:
                    tmp = model.predictions[key].tolist()
                    totals[key]  = totals[key] +  tmp
                
                exp[ds]["PREDICTIONS"] = totals
        

        if not os.path.exists(exp[ds]['PATH']):
            os.makedirs(exp[ds]['PATH'])
        
        save_name = exp[ds]['PATH'] + '/' + ds + '_ExpPredictons.txt'
        self.__saveData__(save_name, totals)
        save_name = exp[ds]['PATH'] + '/' + ds + '_ExpPredictonsPreModel.txt'
        self.__saveData__(save_name, exp[ds]["PREDICTIONS_PER_MODEL"])

        return 
    
    def __saveData__(self, save_name, data):
        
        with open(save_name, "wb") as fp:   #Pickling
            pickle.dump(data, fp)
        return 
            
    def __oldplotConfusionMatrix__(self):
        
        for ds in self.datasets:
            total_predictions = []
            total_ground_truth = []
            
            for ii, ex in enumerate(self.experiments):
                if ds == ex.dataset_info.name:
                    ex.evaluate()
                    if ex.metrics is not None:
                        if isinstance(ex.metrics["Predictions"],dict):
                            for k in ex.metrics["Predictions"]:
                                if k == 'MOUTH':
                                    name = ex.output_name + " - UPPER"
                                elif k == "BROW":
                                    name = ex.output_name + " - LOWER"
                                else:
                                    name = ex.output_name + " - ALL"
                                self.cfplot(ex.metrics["Predictions"][k], ex.metrics["GT"],ex.metrics["Labels"],name,ex.file_save_name)    
                        elif ex.metrics["Predictions"].shape[1] > 1:
                            for jj in range(ex.metrics["Predictions"].shape[1]):
                                self.cfplot(ex.metrics["Predictions"][:,jj], ex.metrics["GT"][:,jj],ex.metrics["Labels"],ex.output_name,ex.file_save_name) 
                        else:
                            self.cfplot(ex.metrics["Predictions"], ex.metrics["GT"],ex.metrics["Labels"], ex.output_name, ex.file_save_name) 
                            tmp_pred = ex.metrics["Predictions"].tolist()
                            tmp_gt = ex.metrics["GT"].tolist()
                            total_predictions = total_predictions + tmp_pred
                            total_ground_truth = total_ground_truth + tmp_gt
                self.cfplot(np.asarray(total_predictions), np.asarray(total_ground_truth),ex.metrics["Labels"], ex.output_name, ex.file_save_name) 
    
    
    def generateResults(self, singles=False):
        
        for exp in self.experiments:
            for ds in exp:
                 exp[ds]["METRICS"] = self.__get_metrics__(exp[ds]["PREDICTIONS"])
                 exp[ds]["METRICS"]["F1"] = exp[ds]["METRICS"]['CR']['weighted avg']['f1-score']                  
                 scores = []
                 for ii, prediction in enumerate(exp[ds]["PREDICTIONS_PER_MODEL"]):
                     exp[ds]["METRICS_PER_MODEL"][ii] = self.__get_metrics__(prediction)
                     scores.append(exp[ds]["METRICS_PER_MODEL"][ii]['CR']['weighted avg']['f1-score'])
                     exp[ds]["METRICS"]["F1_PER_MODEL"] = scores
            
        for exp in self.experiments:
            for ds in exp:
                 self.__plot_metrics__(exp[ds]["METRICS"], exp[ds]['PATH'])
#                 for ii, prediction in enumerate(exp[ds]["PREDICTIONS_PER_MODEL"]):
#                     self.__plot_metrics__(exp[ds]["METRICS_PER_MODEL"][ii], exp[ds]['PATH'])
               
 
        return
    
    def plotMultiExpResults(self, index):
        
        scores = [None] * len(index)
        
        for ii , ind in enumerate(index):
            exp = self.experiments[ind]
            for ds in exp:
                scores[ii] = exp[ds]["METRICS"]["F1_PER_MODEL"]
 
        self.plotFScore(scores, exp[ds]['PATH'], "", ["Mouth Motion","Palsy Grading"])
            
        return
    
    def plotMultiExpResultsAgg(self, indexA, indexB):
        
        scoresA = [None] * len(indexA)
        scoresB = [None] * len(indexB)
        scores = [None] * 2
        
        for ii , ind in enumerate(indexA):
            exp = self.experiments[ind]
            for ds in exp:
                scoresA[ii] = exp[ds]["METRICS"]["F1_PER_MODEL"]
                
        for ii , ind in enumerate(indexB):
            exp = self.experiments[ind]
            for ds in exp:
                scoresB[ii] = exp[ds]["METRICS"]["F1_PER_MODEL"]
        
        scores[0] = [sum(x) /2 for x in zip(scoresA[0], scoresA[1])]
        scores[1] = [sum(x) /2 for x in zip(scoresB[0], scoresB[1])]
 
        self.plotFScore(scores, exp[ds]['PATH'], "",["3DPalsyNet (Ours) - Average F1 Score 0.85","3D CNN - Average F1 Score 0.68"])
            
        return
    
    
    def __get_metrics__(self, predictions):
        
        
        if "PREDS" in predictions:
            predictions, gt = np.asarray(predictions["PREDS"]), np.asarray(predictions["GT"])
        #elif "BBOXES" in predictions:
            
        
        
        
        metrics = {}
        if self.metrics_options["CM"]:
            metrics['CM'] = self.getConfusionMatrix(predictions, gt)
            #self.plotConfusionMatrix(metrics['CM'], save_path, 'Name')
        if self.metrics_options["CR"]:
            metrics['CR'] = self.getClassifcationReport(predictions, gt)
            #self.plotFScore(metrics['CR']['weighted av'], save_path)    
#        if AP:
#            metrics['AP'] = getAP()
#        
#        if MSE:
#            metrics['MSE'] = getMSE()
                        
        return metrics
    
    def __plot_metrics__(self, metrics, save_path):
    
        if self.metrics_options["CM"]:
            self.plotConfusionMatrix(metrics['CM'], save_path, '')
        #if self.metrics_options["CR"]:
            #self.plotFScore(metrics['F1_PER_MODEL'], save_path, "")

        return  
    
    def getConfusionMatrix(self, preds, gt):
        from sklearn.metrics import confusion_matrix
        if len(gt.shape) != len(preds.shape):
            if len(gt.shape) == 1:
                gt =   gt[..., np.newaxis]
                
        cm = confusion_matrix(gt, preds)
        return cm
    
    def getClassifcationReport(self, preds, gt):
        from sklearn.metrics import classification_report
        if len(gt.shape) != len(preds.shape):
            if len(gt.shape) == 1:
                gt =   gt[..., np.newaxis]
        
        cr = classification_report(gt, preds, output_dict=True)
        return cr
    
    
    def __getVOCEvaluation__(self):
        from object_detection.simple_faster_rcnn_pytorch_gs.utils.eval_tool import eval_detection_voc
        
        self.bboxes = self.__padBbox__(self.bboxes)
        metrics = eval_detection_voc(
            self.bboxes, self.labels, self.scores,
            self.dataset_info.gt_bbox_data, self.dataset_info.gt_bbox_label,
            use_07_metric=True)
            
        return metrics

    def getNME(self):
        if  'FACE_LM' in self.dataset_info.labels:
            from libs.landmarkEvaluation import eval_landmarks_nme
            #fro dataset get lms as boxes
            
            pred_box, pred_label, pred_score = self.dataset_info.__landmarksToBoxes__(self.landmarks)
            self.landmarks = self.remapLandamrks()
            metrics = eval_landmarks_nme(self.landmarks, pred_box, pred_label, pred_score, self.dataset_info.labels['FACE_LM']['LM'], self.dataset_info.labels['FACE_LM']['VISIBILTY'],
                                              self.dataset_info.labels['FACE_BBOX']['BBOX'], self.dataset_info.labels['FACE_BBOX']['LABEL'], self.dataset_info.labels['FACE_LM']['INDEX'])    


        return metrics 

    def plotConfusionMatrix(self, cm, save_path, title, normalize=True):

        plt.figure()
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        #print(cm)
        
        if cm.shape[0] == 4:
            labels = ["None", "Smile", "Open", "Other"]
        else: 
            labels = np.arange(1,cm.shape[0]+1)
        
        
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                                 horizontalalignment="center",
                                 color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        save_name = save_path + '_ConfusionMatrix' + '.pdf'
        plt.savefig(save_name, format='pdf', dpi=1000)
        plt.show()
       
        
    def plotFScore(self, scores, save_path, title, legend):
        
        labels = []
        for ii,_ in enumerate(scores[0]):
            label = ii + 1
            labels.append(str(label))
        
        
            #label = ex.Face_Detection_Stage.method + ' - AP: ' + str(round(ex.metrics['ap'][0],4) * 100)
        for ii, score in enumerate(scores):
            plt.plot(labels, score,'o-', label=legend[ii])
            
        
            #print(ex.metrics['pos_total'][0,0],ex.metrics['pos_total'][0,1],ex.metrics['pos_total'][0,2])
        #save_name = self.results_path + '/' + ds + '_APCurve' + '.eps'
        plt.ylim(top=1.1)  # adjust the top leaving bottom unchanged
        plt.ylim(bottom=0)
        #plt.yaxis([0,1])
        plt.xlabel('LOSO Test Set')
        plt.ylabel('F1 Score')
        #plt.title(ds)
        plt.legend()
        save_name = save_path + '_FScorePlot' + '.png'
        plt.savefig(save_name, format='png', dpi=1000)
        plt.show()
        
        return
        
        
    def cfplot(self, preds, gt, labels, title, output_file):
        normalize = True
        from sklearn.metrics import confusion_matrix, classification_report
        if len(gt.shape) != len(preds.shape):
            if len(gt.shape) == 1:
                gt =   gt[..., np.newaxis]
        
        cr = classification_report(gt, preds, output_dict=True)
        #print(cr)
        cm = confusion_matrix(gt, preds)
        plt.figure()
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        #print(cm)
        labels = np.arange(cm.shape[0])
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                                 horizontalalignment="center",
                                 color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        save_name = output_file + '_APCurve' + '.eps'
        plt.savefig(save_name, format='eps', dpi=1000)
        save_name = output_file + '_classification_report' + '.txt'
#        with open(save_name, "w") as text_file:
#            text_file.write(cr)
        
    
    def __plotAPCurve__(self):
        for ds in self.datasets:
            for ii, ex in enumerate(self.experiments):
                if ds == ex.dataset_info.name:
                    ex.evaluate()
                    if ex.metrics is not None:
                        label = ex.Face_Detection_Stage.method + ' - AP: ' + str(round(ex.metrics['ap'][0],4) * 100)
                        plt.plot(ex.metrics['rec'][0], ex.metrics['prec'][0], linewidth=2.0, label=label)
                        print(ex.metrics['pos_total'][0,0],ex.metrics['pos_total'][0,1],ex.metrics['pos_total'][0,2])
            save_name = self.results_path + '/' + ds + '_APCurve' + '.eps'
            plt.xlabel('recall')
            plt.ylabel('precision')
            plt.title(ds)
            plt.legend()
            plt.savefig(save_name, format='eps', dpi=1000)
            plt.show()


    def __plotROCAUC__(self,index=None):
        results_dic = dict()
        lms = dict()
        for ds in self.datasets:
            #Get the evaluation for each method for a singl dataset
            for ii, ex in enumerate(self.experiments):
                if ds == ex.dataset_info.name:
                    ex.evaluate()
                    label = self.experiment_details[ii]['METHOD_DETAILS']['LL']['METHOD']
                    results_dic[label] = ex.metrics
                    lms[label] = ex.Landmark_Localisation_Stage.landmarks
                    ds_size = len(results_dic[label]['error_list'])
            
            #Find the experiment with the smallest detections
            #Note this does not work if we have mismatched detections
#            size = math.inf
#            for dic in results_dic:
#                if len(results_dic[dic]['accuracy']) < size:
#                    size = len(results_dic[dic]['accuracy'])
#                    base_set = dic
#                    
#            sample_index = list()
#            for ii, jj in enumerate(results_dic[base_set]['error_list']):
#                if jj[0].shape[0] != 0:
#                    sample_index.append(ii)
            
            sample_index = list()
            for ind in range(ds_size):
                flag = False
                for jj, dic in enumerate(results_dic):
                    #print(ind)
                    #print(results_dic[dic]['error_list'][ind][0].shape[0])
                    if results_dic[dic]['error_list'][ind][0].shape[0] != 0 and jj == 0:#jj[0].shape[0] != 0:
                        flag = True
                    elif results_dic[dic]['error_list'][ind][0].shape[0] != 0 and jj > 0 and flag:
                        flag = True
                    else:
                        flag = False
                if flag:            
                    sample_index.append(ind)
            size = len(sample_index)
            
            for dic in results_dic:
                error = np.empty((0,1), float)
                for ii in sample_index:
                    err = results_dic[dic]['error_list'][ii][0][0]
                    error = np.append(error, np.array([[err]]), axis=0)
                results_dic[dic]['plot_err'] = error
                results_dic[dic]['plot_sorted_err'] = np.sort(error.T) 
            print(ii,jj)     
            
            accuracy =  np.linspace(0, 1, num=size)
            
            
            
            for dic in results_dic:
                label = dic
                plt.plot(results_dic[dic]['plot_sorted_err'].T,accuracy,linewidth=2.0, label=label)
            plt.xlabel('Average localization error as fraction of face size')
            plt.ylabel('Fraction of the num. of testing faces')
            plt.legend()
            save_name = self.results_path + '/' + ds + '_ROC' + '.eps'
            plt.savefig(save_name, format='eps', dpi=1000)
            plt.show()
            
            vis = visualizeResults()
            ref_lms = pd.read_csv('D:/Research/Code/DL_FW/configs/refLandmarks.txt', delimiter='\t', header=None)
            ref_lms = ref_lms.values
            ref_lms_index = ref_lms!= 0
            img_path = 'D:/Research/Code/DL_FW/configs/meanFace.jpg'
            for dic in results_dic:    
                list_ind = list()
                for lm in range(ref_lms.shape[0]):
                    if ref_lms_index[lm].all() and results_dic[dic]['landmark_index'][lm]:
                        list_ind.append(lm)
                index = np.asarray(list_ind)       
                save_name = self.results_path + '/' + ds + '_' + dic + '_LM_Error' + '.jpg'
                #vis.visualize_landmarks_error(img_path, save_name, ref_lms[results_dic[dic]['landmark_index']],  results_dic[dic]['av_error_landmark'])
                vis.visualize_landmarks_error(img_path, save_name, ref_lms[index],  results_dic[dic]['av_error_landmark'][index])
#            #plt.plot(lm_gt_a['error_sorted'].T,lm_gt_a['accuracy'],linewidth=2.0, label=label)
#            plt.plot(gt_a.T,accuracy,linewidth=2.0, label=label)
#            label = 'FAN GT Boxes - Excluding jaw line'
#            plt.plot(gt_s.T,accuracy,linewidth=2.0, label=label)
#            #plt.plot(lm_gt_s['error_sorted'].T,lm_gt_s['accuracy'],linewidth=2.0, label=label)
#            
    #For each datasert we need to gather all the metric in a dictionairy, find the one with the most non predictions and use this as the basis
#    the trim the errro so that eacxh usese on the same set of data.
#    
#    Theen we can plot it or what ever
            
            
#        pred_err = np.empty((0,1), float)
#        gt_err = np.empty((0,1),float)
#        for ii, jj in enumerate(lm_p_s['error_list']):
#            if jj[0].shape[0] != 0 and lm_gt_s['error_list'][ii][0].shape[0] !=0:
#                err = lm_gt_s['error_list'][ii][0][0]
#                gt_err = np.append(gt_err, np.array([[err]]), axis=0)
#                pred_err = np.append(pred_err, np.array([[jj[0][0]]]), axis=0)
#       
#        accuracy =  np.linspace(0, 1, num=len(pred_err))
#
#        gt_s = np.sort(gt_err.T)
#        pred_s = np.sort(pred_err.T)
#
#
#        label = 'FAN GT Boxes - All'
#        #plt.plot(lm_gt_a['error_sorted'].T,lm_gt_a['accuracy'],linewidth=2.0, label=label)
#        plt.plot(gt_a.T,accuracy,linewidth=2.0, label=label)
#        label = 'FAN GT Boxes - Excluding jaw line'
#        plt.plot(gt_s.T,accuracy,linewidth=2.0, label=label)
#        #plt.plot(lm_gt_s['error_sorted'].T,lm_gt_s['accuracy'],linewidth=2.0, label=label)
#
#        label = 'IDM - All'
#
#        #plt.plot(lm_p_a['error_sorted'].T,lm_p_a['accuracy'],linewidth=2.0, label=label)
#        plt.plot(pred_a.T,accuracy,linewidth=2.0, label=label)
#        label = 'IDM - Excluding jaw line'
#        plt.plot(pred_s.T,accuracy,linewidth=2.0, label=label)
#        #plt.plot(lm_p_s['error_sorted'].T,lm_p_s['accuracy'],linewidth=2.0, label=label)
#
#        plt.xlabel('Average localization error as fraction of face size')
#        plt.ylabel('Fraction of the num. of testing faces')
#        plt.legend()
#
#        save_name = data_experiment_path + '/' + cfg['DATA']['DATASET'] + 'ROC.eps'
#        plt.savefig(save_name, format='eps', dpi=1000)
#        plt.show()
    

class imageOnly(Dataset):
    def __init__(self, config):
        self.image_paths = config.image_locations[0]
        
    def __getitem__(self, index):
        img = getSkiimage(self.image_paths[index])
        return img
        
    def __len__(self):
        return len(self.image_paths)
    
    
