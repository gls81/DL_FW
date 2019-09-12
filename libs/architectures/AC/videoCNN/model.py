import torch
from torch import nn

from libs.architectures.AC.videoCNN.models import resnet, pre_act_resnet, wide_resnet, resnext, densenet


def generate_model(opt):
    if not opt['TRAIN']:
#        assert opt.mode in ['score', 'feature']
#        if opt.mode == 'score':
#            last_fc = True
#        elif opt.mode == 'feature':
#            last_fc = False
        last_fc = True
    else:
        last_fc = True        
    

    assert opt['MODEL']['TYPE'] in ['resnet', 'preresnet', 'wideresnet', 'resnext', 'densenet']

    if opt['MODEL']['TYPE'] == 'resnet':
        assert opt['MODEL']['DEPTH'] in [10, 18, 34, 50, 101, 152, 200]
        
        from libs.architectures.AC.videoCNN.models.resnet import get_fine_tuning_parameters

        if opt['MODEL']['DEPTH'] == 10:
            model = resnet.resnet10(num_classes=opt['MODEL']['NUM_CLASSES'], shortcut_type=opt['MODEL']['RESENT_SHORTCUT'],
                                    sample_size=opt['SAMPLE_SIZE'], sample_duration=opt['SAMPLE_DURATION'],
                                    last_fc=last_fc)
        elif opt['MODEL']['DEPTH'] == 18:
            model = resnet.resnet18(num_classes=opt['MODEL']['NUM_CLASSES'], shortcut_type=opt['MODEL']['RESENT_SHORTCUT'],
                                    sample_size=opt['SAMPLE_SIZE'], sample_duration=opt['SAMPLE_DURATION'],
                                    last_fc=last_fc)
        elif opt['MODEL']['DEPTH'] == 34:
            model = resnet.resnet34(num_classes=opt['MODEL']['NUM_CLASSES'], shortcut_type=opt['MODEL']['RESENT_SHORTCUT'],
                                    sample_size=opt['SAMPLE_SIZE'], sample_duration=opt['SAMPLE_DURATION'],
                                    last_fc=last_fc)
        elif opt['MODEL']['DEPTH'] == 50:
            model = resnet.resnet50(num_classes=opt['MODEL']['NUM_CLASSES'], shortcut_type=opt['MODEL']['RESENT_SHORTCUT'],
                                    sample_size=opt['SAMPLE_SIZE'], sample_duration=opt['SAMPLE_DURATION'],
                                    last_fc=last_fc)
        elif opt['MODEL']['DEPTH'] == 101:
            model = resnet.resnet101(num_classes=opt['MODEL']['NUM_CLASSES'], shortcut_type=opt['MODEL']['RESENT_SHORTCUT'],
                                     sample_size=opt['SAMPLE_SIZE'], sample_duration=opt['SAMPLE_DURATION'],
                                     last_fc=last_fc)
        elif opt['MODEL']['DEPTH'] == 152:
            model = resnet.resnet152(num_classes=opt['MODEL']['NUM_CLASSES'], shortcut_type=opt['MODEL']['RESENT_SHORTCUT'],
                                     sample_size=opt['SAMPLE_SIZE'], sample_duration=opt['SAMPLE_DURATION'],
                                     last_fc=last_fc)
        elif opt['MODEL']['DEPTH'] == 200:
            model = resnet.resnet200(num_classes=opt['MODEL']['NUM_CLASSES'], shortcut_type=opt['MODEL']['RESENT_SHORTCUT'],
                                     sample_size=opt['SAMPLE_SIZE'], sample_duration=opt['SAMPLE_DURATION'],
                                     last_fc=last_fc)
    elif opt['MODEL']['TYPE'] == 'wideresnet':
        assert opt['MODEL']['DEPTH'] in [50]

        if opt['MODEL']['DEPTH'] == 50:
            model = wide_resnet.resnet50(num_classes=opt['MODEL']['NUM_CLASSES'], shortcut_type=opt['MODEL']['RESENT_SHORTCUT'], k=opt.wide_resnet_k,
                                         sample_size=opt['SAMPLE_SIZE'], sample_duration=opt['SAMPLE_DURATION'],
                                         last_fc=last_fc)
    elif opt['MODEL']['TYPE'] == 'resnext':
        assert opt['MODEL']['DEPTH'] in [50, 101, 152]

        if opt['MODEL']['DEPTH'] == 50:
            model = resnext.resnet50(num_classes=opt['MODEL']['NUM_CLASSES'], shortcut_type=opt['MODEL']['RESENT_SHORTCUT'], cardinality=opt.resnext_cardinality,
                                     sample_size=opt['SAMPLE_SIZE'], sample_duration=opt['SAMPLE_DURATION'],
                                     last_fc=last_fc)
        elif opt['MODEL']['DEPTH'] == 101:
            model = resnext.resnet101(num_classes=opt['MODEL']['NUM_CLASSES'], shortcut_type=opt['MODEL']['RESENT_SHORTCUT'], cardinality=opt.resnext_cardinality,
                                      sample_size=opt['SAMPLE_SIZE'], sample_duration=opt['SAMPLE_DURATION'],
                                      last_fc=last_fc)
        elif opt['MODEL']['DEPTH'] == 152:
            model = resnext.resnet152(num_classes=opt['MODEL']['NUM_CLASSES'], shortcut_type=opt['MODEL']['RESENT_SHORTCUT'], cardinality=opt.resnext_cardinality,
                                      sample_size=opt['SAMPLE_SIZE'], sample_duration=opt['SAMPLE_DURATION'],
                                      last_fc=last_fc)
    elif opt['MODEL']['TYPE'] == 'preresnet':
        assert opt['MODEL']['DEPTH'] in [18, 34, 50, 101, 152, 200]

        if opt['MODEL']['DEPTH'] == 18:
            model = pre_act_resnet.resnet18(num_classes=opt['MODEL']['NUM_CLASSES'], shortcut_type=opt['MODEL']['RESENT_SHORTCUT'],
                                            sample_size=opt['SAMPLE_SIZE'], sample_duration=opt['SAMPLE_DURATION'],
                                            last_fc=last_fc)
        elif opt['MODEL']['DEPTH'] == 34:
            model = pre_act_resnet.resnet34(num_classes=opt['MODEL']['NUM_CLASSES'], shortcut_type=opt['MODEL']['RESENT_SHORTCUT'],
                                            sample_size=opt['SAMPLE_SIZE'], sample_duration=opt['SAMPLE_DURATION'],
                                            last_fc=last_fc)
        elif opt['MODEL']['DEPTH'] == 50:
            model = pre_act_resnet.resnet50(num_classes=opt['MODEL']['NUM_CLASSES'], shortcut_type=opt['MODEL']['RESENT_SHORTCUT'],
                                            sample_size=opt['SAMPLE_SIZE'], sample_duration=opt['SAMPLE_DURATION'],
                                            last_fc=last_fc)
        elif opt['MODEL']['DEPTH'] == 101:
            model = pre_act_resnet.resnet101(num_classes=opt['MODEL']['NUM_CLASSES'], shortcut_type=opt['MODEL']['RESENT_SHORTCUT'],
                                             sample_size=opt['SAMPLE_SIZE'], sample_duration=opt['SAMPLE_DURATION'],
                                             last_fc=last_fc)
        elif opt['MODEL']['DEPTH'] == 152:
            model = pre_act_resnet.resnet152(num_classes=opt['MODEL']['NUM_CLASSES'], shortcut_type=opt['MODEL']['RESENT_SHORTCUT'],
                                             sample_size=opt['SAMPLE_SIZE'], sample_duration=opt['SAMPLE_DURATION'],
                                             last_fc=last_fc)
        elif opt['MODEL']['DEPTH'] == 200:
            model = pre_act_resnet.resnet200(num_classes=opt['MODEL']['NUM_CLASSES'], shortcut_type=opt['MODEL']['RESENT_SHORTCUT'],
                                             sample_size=opt['SAMPLE_SIZE'], sample_duration=opt['SAMPLE_DURATION'],
                                             last_fc=last_fc)
    elif opt['MODEL']['TYPE'] == 'densenet':
        assert opt['MODEL']['DEPTH'] in [121, 169, 201, 264]

        if opt['MODEL']['DEPTH'] == 121:
            model = densenet.densenet121(num_classes=opt['MODEL']['NUM_CLASSES'],
                                         sample_size=opt['SAMPLE_SIZE'], sample_duration=opt['SAMPLE_DURATION'],
                                         last_fc=last_fc)
        elif opt['MODEL']['DEPTH'] == 169:
            model = densenet.densenet169(num_classes=opt['MODEL']['NUM_CLASSES'],
                                         sample_size=opt['SAMPLE_SIZE'], sample_duration=opt['SAMPLE_DURATION'],
                                         last_fc=last_fc)
        elif opt['MODEL']['DEPTH'] == 201:
            model = densenet.densenet201(num_classes=opt['MODEL']['NUM_CLASSES'],
                                         sample_size=opt['SAMPLE_SIZE'], sample_duration=opt['SAMPLE_DURATION'],
                                         last_fc=last_fc)
        elif opt['MODEL']['DEPTH'] == 264:
            model = densenet.densenet264(num_classes=opt['MODEL']['NUM_CLASSES'],
                                         sample_size=opt['SAMPLE_SIZE'], sample_duration=opt['SAMPLE_DURATION'],
                                         last_fc=last_fc)

    if opt['CUDA']:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)
                
        if opt['MODEL']['LOAD_MODEL']:
            print('loading pretrained model {}'.format(opt['MODEL']['MODEL_PATH']))
            pretrain = torch.load(opt['MODEL']['MODEL_PATH'])
            #assert opt.arch == pretrain['arch']

            model.load_state_dict(pretrain['state_dict'])

            if opt['MODEL']['TYPE'] == 'densenet':
                model.module.classifier = nn.Linear(
                    model.module.classifier.in_features, opt['MODEL']['NUM_FINE_TUNE_CLASSES'])
                model.module.classifier = model.module.classifier.cuda()
            elif opt['TRAIN']:
                model.module.fc = nn.Linear(model.module.fc.in_features,
                                            opt['MODEL']['NUM_FINE_TUNE_CLASSES'])
                model.module.fc = model.module.fc.cuda()
                
            if opt['TRAIN']:    
                parameters = get_fine_tuning_parameters(model, opt['MODEL']['FINE_TUNE_START_INDEX'])
                return model, parameters
            else:
                return model
    else:
        if opt.prepath:
            print('loading pretrained model {}'.format(opt.prepath))
            pretrain = torch.load(opt.prepath)
            assert opt.arch == pretrain['arch']

            model.load_state_dict(pretrain['state_dict'])

            if opt.model == 'densenet':
                model.classifier = nn.Linear(
                    model.classifier.in_features, opt.n_finetune_classes)
            else:
                model.fc = nn.Linear(mod+el.fc.in_features,
                                            opt.n_finetune_classes)

            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters


    return model, model.parameters()


#Generates any transformation specifc to this type of training and returns a training and test if avaiible dataset 
def generate_dataset(opt, labels, dataset_info, landmark_data=None, index=None):
    from libs.architectures.AC.videoCNN.data.stats import get_mean, get_std
    from libs.transforms.spatial_video import Compose, Grayscale, RandomRotation, ColorJitter,Normalize, Scale, BoundingBoxCrop, CenterCrop, CornerCrop, MultiScaleCornerCrop, MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor
    from libs.transforms.temporal import LoopPadding, TemporalRandomCrop, TemporalSampling
    from libs.transforms.target import ClassLabel
    from libs.architectures.AC.videoCNN.data.dataset import videoData
    from libs.data.datasets import TrainingObjectDataset
    from CustomDataTools.CustomDataset import DatasetInformation
    import pickle
    
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
        
        
    scales = [opt['INITIAL_SCALE']]
    for i in range(1, opt['NUM_SCALES']):
        scales.append(scales[-1] *opt['SCALE_STEP'])
        
#    assert opt['DATASET']['CROP'] in ['random', 'corner', 'center', 'object']
#    if opt['DATASET']['CROP'] == 'random':
#        crop_method = MultiScaleRandomCrop(scales,opt['SAMPLE_SIZE'])
#    elif opt['DATASET']['CROP'] == 'corner':
#        crop_method = MultiScaleCornerCrop(scales, opt['SAMPLE_SIZE'])
#    elif opt['DATASET']['CROP'] == 'center':
#        crop_method = MultiScaleCornerCrop(scales, opt['SAMPLE_SIZE'], crop_positions=['c'])
#    elif opt['DATASET']['CROP'] == 'object':
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
        
    temporal_transform =TemporalSampling(opt['SAMPLE_DURATION'])
        
    target_transform = ClassLabel()
    
    datatset = videoData(dataset_info, labels['NAMES'], crop_method, spatial_transform, temporal_transform,external_bbox=landmark_data)
 
    return datatset