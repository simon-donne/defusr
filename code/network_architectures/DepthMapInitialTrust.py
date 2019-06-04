"""
Predicting the good/bad classification of depth pixels, based solely on the current view.
This network returns the depth estimate and the trust both.
"""

import math
import numpy as np
import torch
from experiment_handler import ExperimentHandler
from network_architectures.network import Network, UNet


class DepthMapInitialTrust(Network):
    """
    Network to classify trust on the input image. It should return a good/bad rating for each pixel.

    Input: center color image and center depth map
    Output: trust image for the first camera
    """
    def __init__(self, F=8, depth_scale=1, scale_augmentation=2.0, file=None):
        super().__init__()
        gpu = torch.device('cuda')

        self.depth_scale = depth_scale
        self.scale_augmentation = scale_augmentation

        if file is not None:
            pt = ExperimentHandler.load_experiment_from_file(file).network
            # classification between good/bad pixels using a UNet
            self.process = pt.process
            self.sigmoid = pt.sigmoid
            return

        # classification between good/bad pixels using a UNet
        self.process = UNet(4, F, 1, depth=2, batchnorms=False).to(gpu)
        self.sigmoid = torch.nn.Sigmoid().to(gpu)
    
    def get_network_name(self):
        return "DepthMapInitialTrust"

    def forward(self, color, depth):
        depth_scale = self.depth_scale
        scale_augmentation = self.scale_augmentation
        if scale_augmentation > 1:
            B = color.shape[0]
            augmentation = scale_augmentation**torch.Tensor(np.random.rand(B,1,1,1)*2-1).cuda()
            depth_scale = depth_scale * augmentation
        scaled_depth = depth / depth_scale

        if color.max() > 1:
            color = color / 255.0

        network_input = torch.cat((color, scaled_depth), dim=1)
        output = self.process(network_input)
        trust = self.sigmoid(output)

        return (depth, trust)

