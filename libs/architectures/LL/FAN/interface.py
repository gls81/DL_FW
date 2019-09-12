import os
import torch
from torch.autograd import Variable
from torch.utils import data as data_
from tqdm import tqdm
from torch.utils.data.dataset import Dataset

#from libs.data.dataset_helpers import getSkiimage


import cv2
from libs.architectures.LL.FAN import face_alignment

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
        self.model = self.__getModel__()
        
      
    def __getModel__(self):
        model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=True, flip_input=False)    

        return model
        
    def __getEvalDataset__(self, ds, bboxes):
        dataset = stackedHGObjectDataset(ds, bboxes)
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

    def eval(self,dataset, bboxes):
        dataloader = self.__getEvalDataset__(dataset,bboxes)
        landmarks, heatmaps, unscaled_landmarks = list(), list(), list()
                
        for ii, (img, center, scale, img_index) in tqdm(enumerate(dataloader)):
            preds, unscaled_pred, scores = self.model.get_landmarks(img, center, scale)
            tmp = np.reshape(preds[0], (136,1))
            landmarks.append(tmp.transpose())
            heatmaps.append(scores)
            tmp = np.reshape(unscaled_pred[0], (136,1))
            unscaled_landmarks.append(tmp.transpose())
            
        landmarks = dataloader.dataset.__singleListToImageList__(landmarks)
        heatmaps = dataloader.dataset.__singleListToImageList__(heatmaps)
        unscaled_landmarks = dataloader.dataset.__singleListToImageList__(unscaled_landmarks)
            
        return landmarks, heatmaps, unscaled_landmarks
            

    
    
    def train(self):

        #self.model.vis.text(dataset.db.label_names, win='labels')
        best_map = 0
        lr_ = self.config["TRAINER"]["LEARNING_RATE"]
        for epoch in range(self.config["EPOCHS"]):
            self.model.reset_meters()
            for ii, (img, bbox_, label_, scale) in tqdm(enumerate(self.dataloader)):
                scale = scale.item()
                img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
                img, bbox, label = Variable(img), Variable(bbox), Variable(label)
                self.model.train_step(img, bbox, label, scale)

                if (ii + 1) % opt.plot_every == 0:
                    if os.path.exists(opt.debug_file):
                        ipdb.set_trace()

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



class stackedHGObjectDataset(Dataset): 
    def __init__(self, dataset_info, bbox_data, resolution=256.0, flip=False):

        self.dataset_info = dataset_info
        self.image_paths = dataset_info.image_locations
        self.labels = dataset_info.labels
        self.bbox_data, self.image_index = self.__processBboxIndex__(bbox_data)
        
           
    def __getitem__(self, index):
        image_ind = self.bbox_data[index][1]
        img = getSkiimage(self.image_paths[image_ind])
        if img.shape[2] == 4: #PNG remove alpha
            img = img[:,:,:3]
        img, center, scale = self.transforms(img, self.bbox_data[index][0].transpose())
        
        return img, center, scale, image_ind
        
    def __len__(self):
        return len(self.bbox_data)
    
    
    def __processBboxIndex__(self, data):
        #BBox data from a face detector or ground truth requires forammting so each detection has a un for get items function
        bbox = list()
        index = list()
        for i,j in enumerate(data):
            index.append(len(j))
            for k in range(len(j)):
                bbox.append([j[k],i])
        
        return bbox, index
    
    def __singleListToImageList__(self, single):
        #Takes ne single list of predicted landmrks and returns a list of the landmarks that belong to a specifc image
        #as some image have multiple detections
        new_list = list()
        cnt = 0
        for i,j in enumerate(self.image_index):
            temp_lm = np.zeros((j,single[cnt].shape[1]))
            for k in range(j):
                temp_lm[k] = single[cnt]
                cnt = cnt + 1
            new_list.append(temp_lm)
        return new_list
    
    def transforms(self, img, bbox):
        #Maybe need to change the bbox stuff around due to the wierd predctions
        center = [bbox[3] - (bbox[3] - bbox[1]) / 2.0, bbox[2] - (bbox[2] - bbox[0]) / 2.0]
        center[1] = center[1] - (bbox[2] - bbox[0]) * 0.12
        scale = (bbox[3] - bbox[1] + bbox[2] - bbox[0]) / 195.0
        inp = self.crop(img, center, scale,resolution=256.0)
        inp = torch.from_numpy(inp.transpose(
                        (2, 0, 1))).float().div(255.0)#.unsqueeze_(0)

        return inp, center, scale
    
    def crop(self,image, center, scale, resolution=256.0):
        # Crop around the center point
        """ Crops the image around the center. Input is expected to be an np.ndarray """
        ul = self.pointtransform([1, 1], center, scale, resolution, True)
        br = self.pointtransform([resolution, resolution], center, scale, resolution, True)
        # pad = math.ceil(torch.norm((ul - br).float()) / 2.0 - (br[0] - ul[0]) / 2.0)
        if image.ndim > 2:
            newDim = np.array([br[1] - ul[1], br[0] - ul[0],
                               image.shape[2]], dtype=np.int32)
            newImg = np.zeros(newDim, dtype=np.uint8)
        else:
            newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
            newImg = np.zeros(newDim, dtype=np.uint8)
        ht = image.shape[0]
        wd = image.shape[1]
        newX = np.array(
                [max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
        newY = np.array(
                [max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
        oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
        oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)
        newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1]
               ] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]
        newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)),
                        interpolation=cv2.INTER_LINEAR)
        return newImg


    def pointtransform(self, point, center, scale, resolution, invert=False):
        _pt = torch.ones(3)
        _pt[0] = point[0]
        _pt[1] = point[1]

        h = 200.0 * scale
        t = torch.eye(3)
        t[0, 0] = resolution / h
        t[1, 1] = resolution / h
        t[0, 2] = resolution * (-center[0] / h + 0.5)
        t[1, 2] = resolution * (-center[1] / h + 0.5)

        if invert:
            t = torch.inverse(t)

        new_point = (torch.matmul(t, _pt))[0:2]

        return new_point.int()

class stackedHGImageOnly(stackedHGObjectDataset):
    def __init__(self, config, bbox_data, resolution=256.0, flip=False):
        stackedHGObjectDataset.__init__(self, config, bbox_data)
        
    def __getitem__(self, index):
        image_ind = self.bbox_data[index][1]
        img = self.__getSkiimage__(image_ind)
        img = self.transforms(img, self.bbox_data[index][0].transpose())
        
        return img
        
    def __len__(self):
        return len(self.bbox_data)
    
    def transforms(self, img, bbox):
        center = torch.FloatTensor([bbox[3] - (bbox[3] - bbox[1]) / 2.0, bbox[2] - (bbox[2] - bbox[0]) / 2.0])
        center[1] = center[1] - (bbox[2] - bbox[0]) * 0.12
        scale = (bbox[3] - bbox[1] + bbox[2] - bbox[0]) / 195.0

        inp = self.crop(img, center, scale,resolution=256.0)

        return inp



