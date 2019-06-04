"""
Creating the visibility masks for all of the scenes.
"""

import os
from datasets.DTU import DTUAdapter
from datasets.unreal_DTU import UnrealDTUAdapter
from datasets.flying_things import FlyingThingsAdapter
import numpy as np
from utils.ply import load_pointcloud_ply, save_pointcloud_ply, load_pointcloud_ply_gipuma
from utils.depth_maps import depth_map_to_point_cloud
from utils.file_system import ensure_dir
from utils.debug import save_gray_tensor, save_color_tensor
from utils.timer import Timer
from experiment_handler import ExperimentHandler
import cv2
from utils.depth_map_visualization import color_depth_map, color_map_errors
import MYTH
import torch

from tqdm import tqdm

count_threshold = 2
sparsity = 0.5
threshold = 2.0

# NOTE: the threadcount is hard-set to 1024 because it is used in some shared-memory allocations inside the sparsity kernel

def compile_sparsity_count_kernel():
    def sparsity_count_gpu(points, sparsity1, sparsity2):
        N = points.shape[1]
        counts1 = np.zeros((N,))
        counts2 = np.zeros((N,))
        if N == 0:
            return counts1, counts2
        points = torch.Tensor(points).cuda()
        counts1 = torch.Tensor(counts1).cuda()
        counts2 = torch.Tensor(counts2).cuda()

        MYTH.bruteforce_sparsity_count_gpu(points, counts1, counts2, N, sparsity1, sparsity2)

        torch.cuda.synchronize()

        counts1 = counts1.cpu().numpy()
        counts2 = counts2.cpu().numpy()

        return counts1, counts2
    return sparsity_count_gpu

def compile_distance_kernel():
    def distance_gpu(points_from, points_to):
        N = points_from.shape[1]
        M = points_to.shape[1]
        dists = np.zeros((N,))
        if N == 0:
            return dists
        if M == 0:
            dists.fill(np.inf)
            return dists

        points_from = torch.Tensor(points_from).cuda()
        points_to = torch.Tensor(points_to).cuda()
        dists = torch.Tensor(dists).cuda()

        MYTH.bruteforce_distance_gpu(points_from, points_to, dists, N, M)

        torch.cuda.synchronize()

        dists = np.sqrt(dists.cpu().numpy())

        return dists
    return distance_gpu

sparsity_count = compile_sparsity_count_kernel()
cloud_distance = compile_distance_kernel()

def filter_cloud(locs, cols, bb=None, visibility=None, voxel_size=None, count_threshold=0, actual_sparsity=None, actual_threshold=None, actual_count_threshold=None):
    if bb is not None:
        inside_bounds = np.all((locs >= bb[0][:,None]) * (locs < bb[1][:,None]), axis=0)
        locs = locs[:,inside_bounds]
        cols = cols[:,inside_bounds]
        
    if voxel_size is not None and visibility is not None:
        voxel_gridsize = np.array(visibility.shape).astype(np.int32).reshape(-1,1)

        # bounding box filtering
        voxs = ((locs - bb[0].reshape(-1,1)) / voxel_size).astype(np.int32)

        # visibility filtering
        visibility_lin = visibility.reshape(-1)
        voxs_lin = voxs[0] * voxel_gridsize[1] * voxel_gridsize[2] + voxs[1] * voxel_gridsize[2] + voxs[2]
        voxs_vis = visibility_lin[voxs_lin].reshape(-1) > 0
        locs = locs[:,voxs_vis]
        cols = cols[:,voxs_vis]

    # density filtering: making sure that we have roughly the same spatial density everywhere
    # we only do this probabilistically, for speed -- give each point a survival chance, then sample
    if actual_sparsity is None:
        actual_sparsity = sparsity
    if actual_threshold is None:
        actual_threshold = threshold
    if actual_count_threshold is None:
        actual_count_threshold = count_threshold

    counts_sparsity, counts_threshold = sparsity_count(locs, actual_sparsity, actual_threshold)
    survival = (np.random.rand(counts_sparsity.shape[0]) < 1/counts_sparsity) * (counts_threshold >= actual_count_threshold)
    locs = locs[:,survival]
    cols = cols[:,survival]

    return locs, cols
