"""
Depth map refinement based on neighbouring views.

An attempt to handle depth map reprojection aliasing artifacts is made.
To do so, the first convolutional layer for the neighbouring depth maps
contains a scaling parameter, dividing by the number of valid pixels in
the kernel support range (taking care not to divide by zero), and the
bias is disabled for that one.

Additionally, we pass the center view depth map to each of the neighbours'
reprojections before calculating features -- hopefully giving more information.
Also included is the original color and the reprojected color.

We shy away from batchnorms because they help us lose the actual depth values.
They can be of interest in side branches, though.
"""

import math
import numpy as np
import os
import torch
from experiment_handler import ExperimentHandler
from network_architectures.network import Network, UNet
from experiments.continue_experiment import get_latest_instance
from MYTH.modules import DepthColorAngleReprojectionNeighbours, DepthReprojectionCompleteBound
from MYTH.modules import AverageReprojectionPooling, MaximumReprojectionPooling, VariationReprojectionPooling, MinimumAbsReprojectionPooling

class DepthMapSuccessiveRefinement(Network):
    """
    Network to refine an input depth map, guided by the color image by that camera and the neighbouring depth views.

    Input: (color images (B x N x C x H x W), depth maps (B x N x 1 x H x W), cameras (B x N x 3 x 4)) - tuple
    Output: refined depth map for the first cameras (B x 1 x H x W)

    The dmin/dmax/dstep parameters are only used for the CompleteBounds part of this network.
    Make sure they cover the entire groundtruth depth range!
    dstep is in some sense the thinnest structure support.
    """

    # one can set file to use pre-trained weights.
    def __init__(self, F=8, local_network=None, depth_scale=1, scale_augmentation=1.0, file=None, refinement_only=False, reset_trust=True):
        super().__init__()

        gpu = torch.device('cuda')

        if file is not None:
            pt = ExperimentHandler.load_experiment_from_file(file).network
            self.F = pt.F
            self.depth_scale = depth_scale
            self.scale_augmentation = scale_augmentation

            self.shared_feature_network = pt.shared_feature_network
            self.inpainting_network = pt.inpainting_network

            # the refinement part: a relatively local (small depth) network used as a residual on the input depth
            self.refinement_network = pt.refinement_network

            # classification between the two options: refinement and inpainting
            self.choice_network = pt.choice_network

            if reset_trust:
                self.trust_network = UNet(2*self.F + 17, 2*self.F, 1, depth=4, batchnorms=False).to(gpu)
            else:
                # the trust network: how much do we trust our proposed refinement?
                self.trust_network = pt.trust_network

            self.softmax = pt.softmax
            self.sigmoid = pt.sigmoid

            # for curriculum learning: when do we turn the classification on?
            self.do_classification = pt.do_classification

            # for curriculum learning: when do we turn the refinement on?
            self.do_refinement = pt.do_refinement

            try:
                self.refinement_only = pt.refinement_only
            except:
                self.refinement_only = False

            return

        self.scale_augmentation = scale_augmentation
        self.F = F
        self.depth_scale = depth_scale

        # just calculating some shared features for the various heads
        self.shared_feature_network = UNet(16, F, 2*F, depth=3, batchnorms=False).to(gpu)

        # the inpainting part: large depth, for huge spatial support so we can easily figure out exactly how to inpaint
        self.inpainting_network = UNet(2*F + 16, 2*F, 1, depth=5, batchnorms=False).to(gpu)

        # the refinement part: a relatively local (small depth) network used as a residual on the input depth
        self.refinement_network = UNet(2*F + 16, 2*F, 1, depth=1, batchnorms=False).to(gpu)

        # classification between the two options: refinement and inpainting
        self.choice_network = UNet(2*F + 18, 2*F, 2, depth=3, batchnorms=False).to(gpu)

        # the trust network: how much do we trust our proposed refinement?
        self.trust_network = UNet(2*F + 17, 2*F, 1, depth=4, batchnorms=False).to(gpu)

        self.softmax = torch.nn.Softmax2d().to(gpu)
        self.sigmoid = torch.nn.Sigmoid().to(gpu)

        # for curriculum learning: when do we turn the classification on?
        self.do_classification = False

        # for curriculum learning: when do we turn the refinement on?
        self.do_refinement = False

        # for ablations: only refinement
        self.refinement_only = refinement_only


    def get_network_name(self):
        return "DepthMapSuccessiveRefinement"


    def forward(self, colors, depths, trust, cameras):
        B = colors.shape[0]
        N = colors.shape[1]
        C = colors.shape[2]
        H = colors.shape[3]
        W = colors.shape[4]

        if colors.max() > 1:
            colors = colors / 255.0

        depth_scale = self.depth_scale
        scale_augmentation = self.scale_augmentation
        if scale_augmentation > 1:
            B = colors.shape[0]

            scales = torch.Tensor(scale_augmentation**(np.random.rand(B)*2-1)).cuda()
            depth_augmentation = scales.reshape(B,1,1,1,1)
            camera_augmentation = scales.reshape(B,1,1,1)
            ctr_depth_augmentation = scales.reshape(B,1,1,1)
        else:
            depth_augmentation = 1
            camera_augmentation = 1
            ctr_depth_augmentation = 1

        depths = depths / depth_scale * depth_augmentation
        cameras = cameras / depth_scale * camera_augmentation
        
        trusted_depths = depths * (trust > 0.75).float() * (depths > 0).float()

        reprojected_depths, reprojected_colors, _ = DepthColorAngleReprojectionNeighbours.apply(trusted_depths, colors, cameras, 1.0)

        ctr_depths = trusted_depths[:, 0]
        ctr_colors = colors[:, 0]

        depth_bounds = reprojected_depths.clone().zero_()
        depth_bounds = DepthReprojectionCompleteBound.apply(trusted_depths, depth_bounds, cameras, 1.0, 0.1, 10.0, 0.01)

        # here, the threshold should be a bit larger than the bound quantization
        cull_mask = (reprojected_depths <= depth_bounds + 0.05).float()
        reprojected_depths = reprojected_depths * cull_mask

        # we have a fancy layer of our own for the aggregation, automatically also getting respective features
        avg_bound, avg_depth, avg_colors = AverageReprojectionPooling.apply(depth_bounds, reprojected_depths, reprojected_colors)
        max_bound, max_depth, max_colors = MaximumReprojectionPooling.apply(depth_bounds, reprojected_depths, reprojected_colors, cull_mask)

        # the min_ pooling of the absolute depth residual w.r.t. the center hypothesis
        residual_bounds = depth_bounds - ctr_depths[:,None,:,:,:]
        residual_depth = reprojected_depths - ctr_depths[:,None,:,:,:]
        residual_depth = residual_depth * cull_mask
        min_residual_bound, min_residual_depth, min_residual_colors = MinimumAbsReprojectionPooling.apply(residual_bounds, residual_depth, reprojected_colors, cull_mask)
        min_residual_bound = min_residual_bound + ctr_depths
        min_residual_depth = min_residual_depth + ctr_depths

        network_input = [
            ctr_depths, ctr_colors, 
            max_depth, max_colors, 
            avg_depth, avg_colors,
            min_residual_depth, min_residual_colors
        ]
        network_input = torch.cat(network_input, dim=1)

        shared_features = self.shared_feature_network(network_input)
        shared_features = torch.cat((shared_features, network_input), dim=1)

        # the depth refinement
        refinement_only = hasattr(self, 'refinement_only') and self.refinement_only
        if self.do_classification and not refinement_only:
            inpainted_depth = self.inpainting_network(shared_features)
            refined_depth = ctr_depths + self.refinement_network(shared_features)
            choices_depths = torch.cat((refined_depth, inpainted_depth), dim=1)
            choice_features = torch.cat((shared_features, choices_depths), dim=1)
            choices = self.choice_network(choice_features)
            choices = self.softmax(choices)

            result_depth = (choices_depths * choices).sum(dim=1, keepdim=True)
        elif refinement_only:
            result_depth = ctr_depths + self.refinement_network(shared_features)
        else:
            result_depth = (self.inpainting_network(shared_features) + ctr_depths + self.refinement_network(shared_features)) / 2

        # the depth trust
        shared_features = torch.Tensor(shared_features.cpu().data).cuda()
        shared_features.requires_grad = True
        trust_features = torch.cat((shared_features, result_depth), dim=1)
        trust_features = trust_features.detach()
        refined_trust = self.trust_network(trust_features)
        refined_trust = self.sigmoid(refined_trust)

        return (result_depth * depth_scale / ctr_depth_augmentation, refined_trust)
