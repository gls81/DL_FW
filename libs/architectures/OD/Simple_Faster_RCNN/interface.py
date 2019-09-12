import os
import torch
from torch.autograd import Variable
from torch.utils import data as data_
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
#import ipdb

from libs.architectures.OD.Simple_Faster_RCNN.model import FasterRCNNModel
from libs.architectures.OD.Simple_Faster_RCNN.trainer import FasterRCNNTrainer
#from libs.architectures.OD.Simple_Faster_RCNN.dataset import trainingData
#from configs.training.config import opt
from libs.architectures.OD.Simple_Faster_RCNN.data.dataset import Dataset, TestDataset, inverse_normalize
#from libs.data.dataset_helpers import getPILimage
#from libs.data.bbox_helpers import resize_bbox, random_flip, flip_bbox
#from libs.architectures.OD.Simple_Faster_RCNN.utils import array_tool as at
#from libs.architectures.OD.Simple_Faster_RCNN.utils.vis_tool import visdom_bbox

class Model():
    def __init__(self, architecture, parameters, train=False, validation_dataset=None):
        """
        Args:
            parameters (FaceExperiment): object with experiment settings
        """
  
        self.type = 'SimpleFasterRCNN'
        self.config = parameters
        self.architecture = architecture
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        

    def __getModel__(self):
        base_model = FasterRCNNModel(n_fg_class=1, base_model=self.config["MODEL"]["TYPE"])  
        trainer = FasterRCNNTrainer(base_model).to(self.device)
        if "MODEL_PATH" in self.config["MODEL"]:
            trainer.load(self.config["MODEL"]["MODEL_PATH"])  

        return trainer
        
    def __getEvalDataset__(self, ds):
        dataset = PredictionObjectDataset(ds)
        dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=False, \
                                  # pin_memory=True,
                                  num_workers=0)
        return dataloader

    
    def __getTrainDataset__(self, ds):
        dataset = TrainingObjectDatasetNEW(ds)
        dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True)
                                  # pin_memory=True,
                                  #num_workers=opt.num_workers)
        return dataloader
    
    def __getVisDataset__(self, ds):        
        #dataset = videoVis(ds)
        return

    def eval(self, dataset):
        self.model = self.__getModel__()
        dataloader = self.__getEvalDataset__(dataset)
        
        pred_bboxes, pred_labels, pred_scores = list(), list(), list()
                
        for ii, (imgs, scale, sizes) in tqdm(enumerate(dataloader)):
            sizes = [sizes[1][0].item(), sizes[0][0].item()]
            pred_bboxes_, pred_labels_, pred_scores_ = self.model.faster_rcnn.predict(imgs, [sizes])
            pred_bboxes += pred_bboxes_
            pred_labels += pred_labels_
            pred_scores += pred_scores_

        return {"BBOXES" : pred_bboxes, "LABELS" : pred_labels, "SCORES" : pred_scores} 
    
    def metrics(self, dataset, predictions):
        dataset = PredictionObjectDataset(dataset)
        from libs.architectures.OD.Simple_Faster_RCNN.utils.eval_tool import eval_detection_voc
     
        
        #self.bboxes = self.__padBbox__(self.bboxes)
        metrics = eval_detection_voc(
        predictions["BBOXES"], predictions["LABELS"], predictions["SCORES"],
        dataset.labels["FACE"]["BBOX"],  dataset.labels["FACE"]["LABEL"],
            use_07_metric=True)
        
        return metrics
    
    def visualise(self, dataset, predictions, path):
        
        return
    
    def train(self, dataset):
        self.model = self.__getModel__()
        dataloader = self.__getTrainDataset__(dataset)
        #self.model.vis.text(dataset.db.label_names, win='labels')
        best_map = 0
        lr_ = self.config["TRAINER"]["LEARNING_RATE"]
        for epoch in range(self.config["EPOCHS"]):
            self.model.reset_meters()
            for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
                scale = scale.item()
                img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
                img, bbox, label = Variable(img), Variable(bbox), Variable(label)
                self.model.train_step(img, bbox, label, scale)

                if (ii + 1) % 10 == 0:
#                    if os.path.exists(opt.debug_file):
#                        ipdb.set_trace()

                    # plot loss
                    self.model.vis.plot_many(self.model.get_meter_data())

                    # plot groud truth bboxes
                    ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                    gt_img = visdom_bbox(ori_img_,
                                         at.tonumpy(bbox_[0]),
                                         at.tonumpy(label_[0]))
                    self.model.vis.img('gt_img', gt_img)

                    # plot predicti bboxes
                    _bboxes, _labels, _scores = self.model.faster_rcnn.predict([ori_img_], visualize=True)
                    pred_img = visdom_bbox(ori_img_,
                                           at.tonumpy(_bboxes[0]),
                                           at.tonumpy(_labels[0]).reshape(-1),
                                           at.tonumpy(_scores[0]))
                    self.model.vis.img('pred_img', pred_img)

                    # rpn confusion matrix(meter)
                    self.model.vis.text(str(self.model.rpn_cm.value().tolist()), win='rpn_cm')
                    # roi confusion matrix
                    self.model.vis.img('roi_cm', at.totensor(self.model.roi_cm.conf, False).float())
            eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)

            if eval_result['map'] > best_map:
                best_map = eval_result['map']
                best_path = self.model.save(best_map=best_map)
            if epoch == 9:
                self.model.load(best_path)
                self.model.faster_rcnn.scale_lr(opt.lr_decay)
                lr_ = lr_ * opt.lr_decay

            self.model.vis.plot('test_map', eval_result['map'])
            log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(self.model.get_meter_data()))
            self.model.vis.log(log_info)
            if epoch == 13: 
                break





class TrainingObjectDatasetNEW(Dataset):
    def __init__(self, dataset_info, height=224, width=224, transform=None, evaluation=False, flip=False):
        
        self.dataset_info = dataset_info
        self.image_paths = dataset_info.image_locations
        self.labels = dataset_info.labels
        
        if len(self.labels) > 1:
            self.multi_label = True
        else:
            self.multi_label = False
        
        #Images transformation parameters
        self.height = height
        self.width = width
        self.transform = transform
        self.evaluation = evaluation
        self.flip = flip
        self.use_lm_box_scaling = True
        
        
    def __getitem__(self, index):
        img = getPILimage(self.image_paths[index])
                                
        bbox = self.labels['FACE']["BBOX"][index]
        label = self.labels['FACE']["LABEL"][index]
        
        img, scale, size, bbox = self.transforms(img, bbox)
 
        torch.from_numpy(bbox)
        torch.from_numpy(label)
        
        return img, bbox, label, scale
    
    def __len__(self):
        return len(self.image_paths)
    
    
    def transforms(self, img, bbox, min_size=600, max_size=1000):
        H, W = img.size
        scale1 = min_size / min(H, W)
        scale2 = max_size / max(H, W)
        scale = min(scale1, scale2)
        transformations = transforms.Compose([transforms.Resize(size=(int(W * scale),int(H *scale)))])
        img = transformations(img)
        
        o_H, o_W = img.size
        scale = o_H / H
        bbox = resize_bbox(bbox, (H, W), (o_H, o_W))
        
        # horizontally flip
#        img, params = random_flip(
#            img, x_random=True, return_param=True)
#        bbox = flip_bbox(
#            bbox, (o_H, o_W), x_flip=params['x_flip'])
        
        transformations = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        img = transformations(img)

        return img, scale, [H, W], bbox


#Sub-class to returns 
class PredictionObjectDataset(Dataset): 
    def __init__(self, dataset_info, height=224, width=224, transform=None, evaluation=False, flip=False):
        
        self.dataset_info = dataset_info
        self.image_paths = dataset_info.image_locations
        self.labels = dataset_info.labels
        
        if len(self.labels) > 1:
            self.multi_label = True
        else:
            self.multi_label = False
        
        #Images transformation parameters
        self.height = height
        self.width = width
        self.transform = transform
        self.evaluation = evaluation
        self.flip = flip
        self.use_lm_box_scaling = True
        
        
    def __getitem__(self, index):
        img = getPILimage(self.image_paths[index])
                                
        #bbox = self.labels['FACE']["BBOX"][index]
        #label = self.labels['FACE']["LABEL"][index]
        
        img, scale, size = self.transforms(img)
 
        #torch.from_numpy(label)
        
        return img, scale, size
    
    def __len__(self):
        return len(self.image_paths)
    
    def transforms(self, img, min_size=600, max_size=1000):
        H, W = img.size
        scale1 = min_size / min(H, W)
        scale2 = max_size / max(H, W)
        scale = min(scale1, scale2)
        transformations = transforms.Compose([transforms.Resize(size=(int(W * scale),int(H *scale)))])
        img = transformations(img)
        
        transformations = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        img = transformations(img)

        return img, scale, [H, W]



