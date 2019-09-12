import torch
from torch import nn

from libs.architectures.yolo.models import darknet

#Generates a model based upon availble architectures of this type
def generate_model(opt):
    cuda = torch.cuda.is_available() and opt['CUDA'] 
    
    assert opt['MODEL']['TYPE'] in ['yolov3','yolov3-tiny']

    if opt['MODEL']['TYPE'] == 'yolov3':
        cfg_file = opt['MODEL']['CONFIG_PATH'] + opt['MODEL']['TYPE'] + '.cfg'
        model = darknet.darknet(cfg_file)
        
        if opt['MODEL']['LOAD_MODEL']:
            print('loading pretrained weights {}'.format(opt['MODEL']['MODEL_PATH']))
            model.load_weights(opt['MODEL']['MODEL_PATH'])
            
    if cuda:
        model = model.cuda()


    return model, model.parameters()

#Generates any transformation specifc to this type of training and returns a training and test if avaiible dataset 
def generate_dataset(opt):
    from libs.data.datasets import TrainingObjectDataset
    from libs.loading_dataset_utils import getDatasetInfo

    
    dataset_info = getDatasetInfo(opt['TRAIN_DATASET']['NAME'])
    datatset = TrainingObjectDataset(dataset_info)

    return datatset
