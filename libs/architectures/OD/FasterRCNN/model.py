import torch
from torch import nn

from libs.architectures.videoCNN.models import resnet, pre_act_resnet, wide_resnet, resnext, densenet

#This is for a Faster R-CNN
def generate_model(opt):
    if not opt['TRAIN']:
        assert opt.mode in ['score', 'feature']
        if opt.mode == 'score':
            last_fc = True
        elif opt.mode == 'feature':
            last_fc = False
    else:
        last_fc = True        
    

    assert opt['MODEL']['TYPE'] in ['resnet', 'vgg', 'alexnet']

    if opt['MODEL']['TYPE'] == 'resnet':
        assert opt['MODEL']['DEPTH'] in [10, 18, 34, 50, 101, 152, 200]
        
        from libs.architectures.videoCNN.models.resnet import get_fine_tuning_parameters

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
    
    elif opt['MODEL']['TYPE'] == 'vgg16':
        assert opt['MODEL']['DEPTH'] in [16]

        if opt['MODEL']['DEPTH'] == 16:
            extractor, classifier = decom_vgg16() = wide_resnet.resnet50(num_classes=opt['MODEL']['NUM_CLASSES'], shortcut_type=opt['MODEL']['RESENT_SHORTCUT'], k=opt.wide_resnet_k,
                                         sample_size=opt['SAMPLE_SIZE'], sample_duration=opt['SAMPLE_DURATION'],
                                         last_fc=last_fc)
    elif opt['MODEL']['TYPE'] == 'alexnet':
    

            model = resnext.resnet50(num_classes=opt['MODEL']['NUM_CLASSES'], shortcut_type=opt['MODEL']['RESENT_SHORTCUT'], cardinality=opt.resnext_cardinality,
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
            else:
                model.module.fc = nn.Linear(model.module.fc.in_features,
                                            opt['MODEL']['NUM_FINE_TUNE_CLASSES'])
                model.module.fc = model.module.fc.cuda()

            parameters = get_fine_tuning_parameters(model, opt['MODEL']['FINE_TUNE_START_INDEX'])
            return model, parameters
    else:
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            assert opt.arch == pretrain['arch']

            model.load_state_dict(pretrain['state_dict'])

            if opt.model == 'densenet':
                model.classifier = nn.Linear(
                    model.classifier.in_features, opt.n_finetune_classes)
            else:
                model.fc = nn.Linear(model.fc.in_features,
                                            opt.n_finetune_classes)

            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters


    return model, model.parameters()
