from __future__ import  absolute_import
import torch as t
from torch import nn
from torchvision.models import vgg16, resnet101, alexnet
from libs.architectures.OD.Simple_Faster_RCNN.model.region_proposal_network import RegionProposalNetwork
from libs.architectures.OD.Simple_Faster_RCNN.model.faster_rcnn import FasterRCNN
from libs.architectures.OD.Simple_Faster_RCNN.model.roi_module import RoIPooling2D
from libs.architectures.OD.Simple_Faster_RCNN.utils import array_tool as at
from libs.architectures.OD.Simple_Faster_RCNN.utils.config import opt


def decom_vgg16():
    # the 30th layer of features is relu of conv5_3
    if opt.caffe_pretrain:
        model = vgg16(pretrained=False)
        if not opt.load_path:
            model.load_state_dict(t.load(opt.caffe_pretrain_path))
    else:
        model = vgg16(not opt.load_path)

    features = list(model.features)[:30]
    classifier = model.classifier

    classifier = list(classifier)
    del classifier[6]
    if not opt.use_drop:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    # freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier


def decom_resnet101():
    # the 30th layer of features is relu of conv5_3

    model = resnet101(pretrained=True)
    features = list(model.children())[:-1]      # delete the last fc layer.
    
     
    #Note we use the classifier componet of the VGG network
    model = vgg16(not opt.load_path)
    classifier = model.classifier
    classifier[0] = nn.Linear(25088*4, 4096)
    classifier = list(classifier)

    
    del classifier[6]
    if not opt.use_drop:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    # freeze initial layer before blocks
    for layer in features[:6]:
        for p in layer.parameters():
            p.requires_grad = False
            

    return nn.Sequential(*features), classifier

def decom_alexnet():
    # the 30th layer of features is relu of conv5_3

    model = alexnet(pretrained=True)
    features = list(model.features)[:-1]    # delete the last fc layer.
    classifier = model.classifier

    classifier = list(classifier)
    del classifier[6]
    if not opt.use_drop:
        del classifier[3]
        del classifier[0]
    classifier = nn.Sequential(*classifier)

    # freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier


class FasterRCNNModel(FasterRCNN):
    """Faster R-CNN based with options for the base model of VGG-16, Alexnet.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """

    #feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    def __init__(self,
                 n_fg_class=1,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32],
                 multi_task_cfg=None,
                 base_model = 'vgg16'
                 ):
             
        
        if base_model == 'vgg16':
            extractor, classifier = decom_vgg16()
            extrator_out_dim = 512
            feat_stride = 16
            roi_size = 7
        elif base_model == 'alexnet':
            extractor, classifier = decom_alexnet()
            extrator_out_dim = 256
            feat_stride = 16
            roi_size = 6
        elif base_model == 'resnet101':
            extractor, classifier = decom_resnet101()
            extrator_out_dim = 2048
            feat_stride = 16
            roi_size = 7
                
        rpn = RegionProposalNetwork(
            extrator_out_dim, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=feat_stride,
        )

        head = VGG16RoIHead(
            n_class=n_fg_class + 1,
            roi_size=roi_size,
            spatial_scale=(1. / feat_stride),
            classifier=classifier, 
            multi_task_cfg=multi_task_cfg
        )

        super(FasterRCNNModel, self).__init__(
            extractor,
            rpn,
            head, 
            multi_task_cfg,
        )


class VGG16RoIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.
    
    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier, multi_task_cfg):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)
        
        
        if multi_task_cfg is not None:
            self.multi_task =  True
            self.multi_task_layers =self.__contructMTLayers__(multi_task_cfg)
        else:
            self.multi_task =  False
            
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)
        
    def __contructMTLayers__(self, cfg):
    
        layers = list()
        for i in range(cfg.num_multi_lab):
            num_out = cfg.num_classes_mulit_lab[i]
            layers.append(nn.Linear(4096, num_out))
            normal_init(layers[i], 0, 0.01)
        return layers

    def forward(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        # in case roi_indices is  ndarray
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois =  xy_indices_and_rois.contiguous()
        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        
        if self.multi_task:
            mt_scores = self.__multi_forward__(fc7)
            return roi_cls_locs, roi_scores, mt_scores
            
        return roi_cls_locs, roi_scores
    
    def __multi_forward__(self, x):
        scores = list()      
        for f in self.multi_task_layers:
            f.cuda()
            scores.append(f(x))
    
        return scores
    
class FasterRCNNResnet(FasterRCNN):
    """Faster R-CNN based on Resnet-101.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """

    feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    def __init__(self,
                 n_fg_class=1,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ):
                 
        extractor, classifier = decom_resnet()

        rpn = RegionProposalNetwork(
            2048, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = VGG16RoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )

        super(FasterRCNNResnet, self).__init__(
            extractor,
            rpn,
            head,
        )




def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
