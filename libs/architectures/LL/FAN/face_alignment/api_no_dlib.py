# -*- coding: utf-8 -*-
"""
Created on Tue May 29 16:14:22 2018

@author: Gary
"""

from __future__ import print_function
import os
import numpy as np
import glob
import torch
import torch.nn as nn
from enum import Enum
from skimage import io
try:
    import urllib.request as request_file
except BaseException:
    import urllib as request_file

from .models import FAN, ResNetDepth
from .utils import *


class LandmarksType(Enum):
    _2D = 1
    _2halfD = 2
    _3D = 3


class NetworkSize(Enum):
    # TINY = 1
    # SMALL = 2
    # MEDIUM = 3
    LARGE = 4

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __int__(self):
        return self.value


class FaceAlignment:
    """Initialize the face alignment pipeline

    Args:
        landmarks_type (``LandmarksType`` object): an enum defining the type of predicted points.
        network_size (``NetworkSize`` object): an enum defining the size of the network (for the 2D and 2.5D points).
        enable_cuda (bool, optional): If True, all the computations will be done on a CUDA-enabled GPU (recommended).
        enable_cudnn (bool, optional): If True, cudnn library will be used in the benchmark mode
        flip_input (bool, optional): Increase the network accuracy by doing a second forward passed with
                                    the flipped version of the image
        use_cnn_face_detector (bool, optional): If True, dlib's CNN based face detector is used even if CUDA
                                                is disabled.

    Example:
        >>> FaceAlignment(NetworkSize.2D, flip_input=False)
    """

    def __init__(self, landmarks_type, network_size=NetworkSize.LARGE,
                 enable_cuda=True, enable_cudnn=True, flip_input=False,
                 use_cnn_face_detector=False):
        self.enable_cuda = enable_cuda
        self.use_cnn_face_detector = use_cnn_face_detector
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        base_path = os.path.join(appdata_dir('face_alignment'), "data")

        if not os.path.exists(base_path):
            os.makedirs(base_path)

        if enable_cudnn and self.enable_cuda:
            torch.backends.cudnn.benchmark = True

        # Initialise the face alignemnt networks
        self.face_alignemnt_net = FAN(int(network_size))
        if landmarks_type == LandmarksType._2D:
            network_name = '2DFAN-' + str(int(network_size)) + '.pth.tar'
        else:
            network_name = '3DFAN-' + str(int(network_size)) + '.pth.tar'
        fan_path = os.path.join(base_path, network_name)

        if not os.path.isfile(fan_path):
            print("Downloading the Face Alignment Network(FAN). Please wait...")

            request_file.urlretrieve(
                "https://www.adrianbulat.com/downloads/python-fan/" +
                network_name, os.path.join(fan_path))

        fan_weights = torch.load(
            fan_path,
            map_location=lambda storage,
            loc: storage)

        self.face_alignemnt_net.load_state_dict(fan_weights)

        if self.enable_cuda:
            self.face_alignemnt_net.cuda()
        self.face_alignemnt_net.eval()

        # Initialiase the depth prediciton network
        if landmarks_type == LandmarksType._3D:
            self.depth_prediciton_net = ResNetDepth()
            depth_model_path = os.path.join(base_path, 'depth.pth.tar')
            if not os.path.isfile(depth_model_path):
                print(
                    "Downloading the Face Alignment depth Network (FAN-D). Please wait...")

                request_file.urlretrieve(
                    "https://www.adrianbulat.com/downloads/python-fan/depth.pth.tar",
                    os.path.join(depth_model_path))

            depth_weights = torch.load(
                depth_model_path,
                map_location=lambda storage,
                loc: storage)
            depth_dict = {
                k.replace('module.', ''): v for k,
                v in depth_weights['state_dict'].items()}
            self.depth_prediciton_net.load_state_dict(depth_dict)

            if self.enable_cuda:
                self.depth_prediciton_net.cuda()
            self.depth_prediciton_net.eval()

    def get_landmarks(self, input_image, bboxes, all_faces=True):
        if isinstance(input_image, str):
            try:
                image = io.imread(input_image)
            except IOError:
                print("error opening file :: ", input_image)
                return None
        else:
            image = input_image


        if len(bboxes) > 0:
            if self.landmarks_type == LandmarksType._2D:     
                landmarks = np.zeros(shape=(len(bboxes),68,2))
            else:
                landmarks = np.zeros(shape=(len(bboxes),68,3))
            for j, d in enumerate(bboxes):
                if j > 1 and not all_faces:
                    break
                center = torch.FloatTensor(
                    [d[2] - (d[2] - d[0]) / 2.0, d[3] -(d[3] - d[1]) / 2.0])
                center[1] = center[1] - (d[3] - d[1]) * 0.12
                scale = (d[2] - d[0] + d[3] - d[1]) / 195.0
            
                
                inp = crop(image, center, scale)
                inp = torch.from_numpy(inp.transpose(
                    (2, 0, 1))).float().div(255.0).unsqueeze_(0)

                if self.enable_cuda:
                    inp = inp.cuda()

                out = self.face_alignemnt_net(
                    Variable(inp, volatile=True))[-1].data.cpu()
                if self.flip_input:
                    out += flip(self.face_alignemnt_net(flip(inp))
                                    [-1].data.cpu(), is_label=True)


                pts, pts_img = get_preds_fromhm(out, center, scale)
                pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)

                if self.landmarks_type == LandmarksType._3D:
                    heatmaps = np.zeros((68, 256, 256))
                    for i in range(68):
                        if pts[i, 0] > 0:
                            heatmaps[i] = draw_gaussian(heatmaps[i], pts[i], 2)
                    heatmaps = torch.from_numpy(
                        heatmaps).view(1, 68, 256, 256).float()
                    if self.enable_cuda:
                        heatmaps = heatmaps.cuda()
                    depth_pred = self.depth_prediciton_net(
                            torch.cat((inp, heatmaps), 1)).data.cpu().view(68, 1)
                    pts_img = torch.cat(
                        (pts_img, depth_pred * (1.0 / (256.0 / (200.0 * scale)))), 1)

                landmarks[j,:,:] = pts_img.numpy()
        else:
            print("Warning: No faces were detected.")
            return None

        return landmarks

    def process_folder(self, path, all_faces=False):
        types = ('*.jpg', '*.png')
        images_list = []
        for files in types:
            images_list.extend(glob.glob(os.path.join(path, files)))

        predictions = []
        for image_name in images_list:
            predictions.append((
                image_name, self.get_landmarks(image_name, all_faces)))

        return predictions

    def remove_models(self):
        base_path = os.path.join(appdata_dir('face_alignment'), "data")
        for data_model in os.listdir(base_path):
            file_path = os.path.join(base_path, data_model)
            try:
                if os.path.isfile(file_path):
                    print('Removing ' + data_model + ' ...')
                    os.unlink(file_path)
            except Exception as e:
                print(e)