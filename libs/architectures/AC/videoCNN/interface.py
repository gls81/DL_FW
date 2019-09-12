import os
import torch
from torch.autograd import Variable
from torch.utils import data as data_
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
import functools
from libs.architectures.AC.videoCNN.model import generate_model 
from libs.architectures.AC.videoCNN.data.stats import get_mean, get_std

from libs.dataset.dataset_helpers import getSkiimage
#from libs.visualise import visualizeResults

from torch import optim

import json
import copy



class Model():
    def __init__(self, architecture, parameters, train=False, validation_dataset=None, visualise=False):
        """
        Args:
            parameters (FaceExperiment): object with experiment settings
        """  
        self.type = '3DCNN'
        self.config = parameters
        self.architecture = architecture
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __getModel__(self):
        model = generate_model(self.config)    

        return model
        
    def __getEvalDataset__(self, ds):
        mean = get_mean(self.config['NORM_VALUE'], dataset= self.config['MEAN_DATASET'])
        std = get_std(self.config['NORM_VALUE'])
         
        if not self.config['MEAN_NORMALISATION'] and not self.config['STD_NORMALISATION']:
             norm_method = Normalize([0, 0, 0], [1, 1, 1])
        elif not self.config['STD_NORMALISATION']:
            norm_method = Normalize(mean, [1, 1, 1])
        else:
            norm_method = Normalize(mean, std)
        
        crop_method = BoundingBoxCrop(self.config['SAMPLE_SIZE'])
        spatial_transform = Compose([               
                Grayscale(),
                ToTensor(self.config['NORM_VALUE']), norm_method
                ])
        
        temporal_transform =TemporalSampling(self.config['SAMPLE_DURATION'])
        dataset = videoData(ds, crop_method, spatial_transform, temporal_transform)
        
        dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=False, \
                                  # pin_memory=True,
                                  num_workers=0)
        return dataloader

    def __getTrainDataset__(self, ds, classBalance=True):
        mean = get_mean(self.config['NORM_VALUE'], dataset= self.config['MEAN_DATASET'])
        std = get_std(self.config['NORM_VALUE'])
         
        if not self.config['MEAN_NORMALISATION'] and not self.config['STD_NORMALISATION']:
             norm_method = Normalize([0, 0, 0], [1, 1, 1])
        elif not self.config['STD_NORMALISATION']:
            norm_method = Normalize(mean, [1, 1, 1])
        else:
            norm_method = Normalize(mean, std)
        
        
        
        crop_method = BoundingBoxCrop(self.config['SAMPLE_SIZE'])
        spatial_transform = Compose([
                ColorJitter(brightness=0.5),
                RandomRotation(180),
                Grayscale(),
                RandomHorizontalFlip(),
                ToTensor(self.config['NORM_VALUE']), norm_method
                ])
        
        temporal_transform =TemporalSampling(self.config['SAMPLE_DURATION'])
        dataset = videoData(ds, crop_method, spatial_transform, temporal_transform)
        
        if classBalance:
            target = dataset.labels   
            #target = dataset.labels[:,1] for multilabel not working
            class_sample_count = np.array(
                    [len(np.where(target == t)[0]) for t in np.unique(target)])
            weight = 1. / class_sample_count
        
            samples_weight = np.array([weight[t] for t in target])
            samples_weight = torch.from_numpy(samples_weight)
            samples_weight = samples_weight.double()
            sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
        else:
            sampler=None
        
        dataloader = data_.DataLoader(dataset,
                                      batch_size=self.config['BATCH_SIZE'],
                                      num_workers=0, sampler=sampler,
                                      pin_memory=False)
        return dataloader
    
    def __getVisDataset__(self, ds):

        dataset = videoVis(ds)
        return dataset

    def eval(self, dataset):
        dataloader = self.__getEvalDataset__(dataset)
        self.model = self.__getModel__()
        self.model = self.model.eval()
        
        row = dataloader.dataset.__len__()
        col = dataloader.dataset.data.total_labels
        predictions =  np.ones((row,col), dtype=int)
        gt_labels = np.ones((row,col), dtype=int)
        scores = np.ones((row,dataloader.dataset.data.total_classes), dtype=float)
        
        for j, (inputs, targets) in enumerate(dataloader):
            targets = targets.cuda(async=True)
            inputs = Variable(inputs)
            targets = Variable(targets)
            targets = targets.long()
            _, outputs = self.model(inputs)
            
            if dataloader.dataset.data.total_labels > 1:
                sigmoid_outputs = torch.sigmoid(outputs)
                tmp = sigmoid_outputs.cpu().detach().numpy()
                predicted = np.around(tmp)
                predictions[j,:] = predicted
                gt_labels[j,:] = targets.cpu().detach().numpy()
            else:
                _, predicted = torch.max(outputs.data, 1)
                predictions[j,:] = predicted.cpu().detach().numpy()    
                gt_labels[j,:] = targets.cpu().detach().numpy()
                scores[j,:] = outputs.cpu().detach().numpy()
                print(predictions[j,:])
                print(gt_labels[j,:])
                
                      
        return {"PREDS" : predictions, "GT" : gt_labels, "SCORES" : scores}
    
    
    def visualise(self, dataset, predictions, path):
        dataloader = self.__getVisDataset__(dataset)
        self.vis = visualizeResults()
        
        for key in dataset.labels_legend:
            legend = dataset.labels_legend[key]["LEGEND"]
        
        for index in range(dataloader.__len__()):
            frames, label, bbox = dataloader.__getitem__(index)
            gt_lab = legend[label]
            pred_lab = legend[predictions["PREDS"][index][0]]
            save_path = path + "/"  + str(index) + "/" 
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for jj, image in enumerate(frames):
                file_number = str(jj)
                save_name = save_path + file_number.zfill(3) + ".jpg"
                self.vis.visualize_bbox(image, save_name, gt_data=bbox, gt_label=gt_lab, pred_label=pred_lab)

        return
    
    
    def train(self, dataset, save_path, plotter=True, logger=True):
        #self.model.vis.text(dataset.db.label_names, win='labels')
        dataloader = self.__getTrainDataset__(dataset)
        
        for j, (inputs, targets) in enumerate(dataloader):
                print('train at sample {}'.format(j))
        
        
#        self.model, self.model_parameters = generate_model(self.config) 
#        model_save_name = save_path[:-4] + '.pth'
#        
#    
#        if plotter:    
#            plotter = VisdomLinePlotter(env_name="TEST")
#        
#        if logger:
#            self.losses = AverageMeter()
#            self.accuracies = AverageMeter()
#            self.data_time = AverageMeter()
#            self.accu_log = []
#        
#        
#        
#        if self.config["LOSS"] == "SOFTMAX":
#             self.criterion = nn.CrossEntropyLoss()
#             self.optimizer = self.setOptimizer(self.config, self.model_parameters)
#             self.loss = self.getSoftmaxLoss
#             self.loss_log = []
#        elif self.config["LOSS"] == "CENTRE+SOFTMAX":
#            self.criterion = nn.CrossEntropyLoss()
#            self.optimizer = self.setOptimizer(self.config,self.model_parameters)
#            self.criterion_centre = CenterLoss(num_classes=self.parameters["MODEL"]["NUM_FINE_TUNE_CLASSES"], feat_dim=512, use_gpu=True) #512 need to chage at some pooint for other models features sizes
#            self.optimizer_centre = torch.optim.SGD(self.criterion_centre.parameters(), lr=0.3)
#            self.loss = self.getSoftmaxCentreLoss
#            self.losses_center = AverageMeter()
#            self.losses_softmax = AverageMeter()
#            self.loss_log = []
#            self.loss_center_log = []
#        
#        #self.config['EPOCHS'] = 1
#        
#        for i in range(self.config['START_EPOCH'], self.config['EPOCHS'] + 1):
#            print('train at epoch {}'.format(i))
#            self.model.train()
#            
#            for j, (inputs, targets) in enumerate(dataloader):
#                print('train at sample {}'.format(j))
#                
#                if self.config['CUDA']:
#                    targets = targets.cuda(async=True)
#                inputs = Variable(inputs)
#                targets = Variable(targets)
#                targets = targets.long()
#  
#                self.loss(self.model,inputs,targets)
#               
#            plotter.plot('loss', 'train', 'Class Loss', i, self.losses.avg)
#            plotter.plot('accuracy', 'train', 'Accuracy', i, self.accuracies.avg)
#            self.loss_log.append(self.losses.avg)
#            self.accu_log.append(self.accuracies.avg)
#
#        states = {
#                'epoch': i + 1,
#                'state_dict': self.model.state_dict(),
#                'optimizer': self.optimizer.state_dict(),
#                }
#        torch.save(states, model_save_name)
#        self.__createEvalConfig__(model_save_name,save_path)
    
    def setOptimizer(self, parameters, model_weights):
        
        if parameters['TRAINER']['NESTEROV']:
            dampening = 0
        else:
            dampening = parameters['TRAINER']['DAMPENING']
        
        if parameters['TRAINER']['OPTIMIZER'] ==  'SGD':
            optimizer = optim.SGD(
                    model_weights,
                    lr=parameters['TRAINER']['LEARNING_RATE'],
                    momentum=parameters['TRAINER']['MOMENTUM'],
                    dampening=dampening,
                    weight_decay=parameters['TRAINER']['WEIGHT_DECAY'],
                    nesterov=parameters['TRAINER']['NESTEROV'])
        elif parameters['TRAINER']['OPTIMIZER'] ==  'ADAM':
            optimizer = optim.Adam(
                    model_weights,
                    lr=opt.learning_rate,
                    momentum=opt.momentum,
                    dampening=dampening,
                    weight_decay=opt.weight_decay,
                    nesterov=self.parameters['TRAINER']['NESTEROV'])
        
        return optimizer
    
    
    def getSoftmaxLoss(self,model,inputs,targets):
     
        features,outputs = model(inputs)
        loss = self.criterion(outputs, targets)
        self.losses.update(loss.item(), inputs.size(0))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        acc = calculate_accuracy(outputs, targets)
        self.accuracies.update(acc, inputs.size(0))
    
        return
    

    def getSoftmaxCentreLoss(self,model,inputs,targets):
        
        features, outputs = model(inputs)
        loss_soft = self.criterion(outputs, targets)
        loss_cent = self.criterion_centre(features, targets)
        loss_cent *= 0.001
        
        self.losses_softmax.update(loss_soft.item(), inputs.size(0))
        self.losses_center.update(loss_cent.item(), inputs.size(0))
        
        loss = loss_soft + loss_cent 
        self.losses.update(loss.item(), inputs.size(0))
        
        self.optimizer.zero_grad()
        self.optimizer_centre.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        acc = calculate_accuracy(outputs, targets)
        self.accuracies.update(acc, inputs.size(0))
        
        
        for param in self.criterion_centre.parameters():
            param.grad.data *= (1. / 0.001)
        self.optimizer_centre.step()
        
        return
    
    def __createEvalConfig__(self, model_save_name, save_path):
        eval_params = {}
        eval_params['PARAMETERS'] = copy.deepcopy(self.config)
        #eval_params['NAME'] = self.research_name
        #eval_params['TYPE'] = self.type
        #eval_params['METHOD'] = self.method
        #eval_params['ARCH'] = self.arch
        #eval_params["DATASETS"] = self.dataset_name
        eval_params['PARAMETERS']['MODEL']['MODEL_PATH'] = model_save_name
        eval_params['PARAMETERS']['TRAIN'] = False
        eval_params['PARAMETERS']['MODEL']['NUM_CLASSES'] = self.config['MODEL']['NUM_FINE_TUNE_CLASSES']
        
        file_name = save_path
        with open(file_name, 'w') as outfile:
            json.dump(eval_params, outfile)
        
        return
    


def generate_dataset(opt, labels, dataset_info, landmark_data=None, index=None):
#    if opt['LANDMARK_DATA']:
#        with open(opt['DATASET']['LANDMARK_DATA_FILE'], "rb") as fp:   # Unpickling
#            landmark_data = pickle.load(fp)
    mean = get_mean(opt['NORM_VALUE'], dataset=opt['MEAN_DATASET'])
    std = get_std(opt['NORM_VALUE'])
        
    if not opt['MEAN_NORMALISATION'] and not opt['STD_NORMALISATION']:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt['STD_NORMALISATION']:
        norm_method = Normalize(mean, [1, 1, 1])
    else:
        norm_method = Normalize(mean, std)            
    crop_method = BoundingBoxCrop(opt['SAMPLE_SIZE']) 
    if opt["TRAIN"]:
        spatial_transform = Compose([
                ColorJitter(brightness=0.5),
                RandomRotation(180),
                Grayscale(),
                RandomHorizontalFlip(),
                ToTensor(opt['NORM_VALUE']), norm_method
                ])
    else:
        spatial_transform = Compose([               
                Grayscale(),
                ToTensor(opt['NORM_VALUE']), norm_method
                ])       
    temporal_transform = TemporalSampling(opt['SAMPLE_DURATION'])       
    datatset = videoData(dataset_info, labels['NAMES'], crop_method, spatial_transform, temporal_transform,external_bbox=landmark_data)
 
    return datatset


class videoData(Dataset):
    def __init__(self, dataset_info , crop_method=None, spatial_transform=None, temporal_transform=None):
        
        self.data = dataset_info
        
        self.labels = []
        for label in dataset_info.labels:
            self.labels.append(dataset_info.labels[label]['LABEL'])
        
        if len(self.labels) == 1:
            self.labels = self.labels[0]    
        else:
            self.labels = np.transpose(np.stack(self.labels))
        
        #tranform external data
        if self.data.external_data is not None:
           self.data.external_data = self.data.landmarksToBoxesOnly(self.data.external_data)     
           #self.bbox_crop1 = self.get_bbox_Crop(self.data.external_data)
        self.crop_method = crop_method  
        self.spatial_transform = spatial_transform        
        self.temporal_transform = temporal_transform
        self.loader = self.get_default_video_loader()
        
    def __getitem__(self, index):
        paths = self.data.image_locations[index]
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(len(paths))
            print(frame_indices)
            print(paths)
        clip = self.loader(paths, frame_indices)
        #if self.crop_method is not None:
            #clip = [self.crop_method(img, self.bbox_crop[index]) for img in clip]
        clip = self.crop_method(clip, self.data.external_data[index])
        clip = self.spatial_transform(clip)
#        for i, img in enumerate(clip):
#            vs = visualizeResults()
#            save_name = 'D:/tmp/' + str(index) + '_' + str(i) + '.jpg'
#            vs.save_image(img,save_name)
                 
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        target = self.labels[index]
        
        return clip, target
        
    def __len__(self):
        return len(self.data.image_locations)
      
#    def get_bbox_Crop(self, boxdata):
#        crop = []
#        for boxes in boxdata:
#            x1 = math.inf  
#            x2 = -math.inf
#            y1 = math.inf
#            y2 = -math.inf
#            for box in boxes:
#                box = np.reshape(box[0], (-1, 2))
#                #box = box[43:44,:]
#                tmp_x1 = int(min(box[:,0]))
#                if  tmp_x1 < x1:
#                    x1 = tmp_x1
#                tmp_x2 = int(max(box[:,0]))
#                if  tmp_x2 > x2:
#                    x2 = tmp_x2
#                tmp_y1 = int(min(box[:,1]))
#                if  tmp_y1 < y1:
#                    y1 = tmp_y1
#                tmp_y2  = int(max(box[:,1]))    
#                if  tmp_y2 > y2:
#                    y2 = tmp_y2
#            crop.append([x1,x2,y1,y2])
#            
#        return crop
    
    def pil_loader(self, path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')


    def accimage_loader(path):
        try:
            return accimage.Image(path)
        except IOError:
            # Potentially a decoding problem, fall back to PIL.Image
            return pil_loader(path)


    def get_default_image_loader(self):
        from torchvision import get_image_backend
        if get_image_backend() == 'accimage':
            import accimage
            return accimage_loader
        else:
            return self.pil_loader


    def video_loader(video_dir_path, frame_indices, image_loader):
        video = []
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
            if os.path.exists(image_path):
                video.append(image_loader(image_path))
            else:
                return video

        return video


    def palsy_video_loader(self, image_paths, frame_indices, image_loader):
        video = []
        for i in frame_indices:
            image_path = image_paths[i]
            if os.path.exists(image_path):
                video.append(image_loader(image_path))
            else:
                return video

        return video

    def get_default_video_loader(self):
        image_loader = self.get_default_image_loader()
        return functools.partial(self.palsy_video_loader, image_loader=image_loader)
    
    
class videoVis(Dataset):
    def __init__(self, dataset_info):
        self.data = dataset_info
        self.labels = []
        for label in dataset_info.labels:
            self.labels.append(dataset_info.labels[label]['LABEL'])
        if len(self.labels) == 1:
            self.labels = self.labels[0]    
        else:
            self.labels = np.transpose(np.stack(self.labels))
        
        self.loader = self.get_default_video_loader()
        
    def __getitem__(self, index):
        paths = self.data.image_locations[index]
        frame_indices = np.arange(len(paths))
        clip = self.loader(paths, frame_indices)
        bbox_list = []
        bbox_list.append(self.data.external_data[index])
        
        return clip, self.labels[index], bbox_list
        
    def __len__(self):
        return len(self.data.image_locations)
          
    def pil_loader(self, path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def accimage_loader(path):
        try:
            return accimage.Image(path)
        except IOError:
            # Potentially a decoding problem, fall back to PIL.Image
            return getSkiimage(path)


    def get_default_image_loader(self):
        from torchvision import get_image_backend
        if get_image_backend() == 'accimage':
            import accimage
            return accimage_loader
        else:
            return self.pil_loader

    def video_loader(video_dir_path, frame_indices, image_loader):
        video = []
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
            if os.path.exists(image_path):
                video.append(image_loader(image_path))
            else:
                return video

        return video



    def palsy_video_loader(self, image_paths, frame_indices, image_loader):
        video = []
        for i in frame_indices:
            image_path = image_paths[i]
            if os.path.exists(image_path):
                video.append(getSkiimage(image_path))
            else:
                return video

        return video

    def get_default_video_loader(self):
        image_loader = self.get_default_image_loader()
        return functools.partial(self.palsy_video_loader, image_loader=image_loader)