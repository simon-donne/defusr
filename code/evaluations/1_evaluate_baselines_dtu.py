from experiment_handler import ExperimentHandler
from utils.chamfer_evaluation import filter_cloud
from utils.chamfer_evaluation import cloud_distance
from utils.depth_maps import depth_map_to_point_cloud
from utils.ply import load_pointcloud_ply_gipuma, load_pointcloud_ply
import numpy as np
import sys
import os
import pickle
from tqdm import tqdm

from local_config import base_data_folder, base_evaluation_output_path

gt_cloud_format = os.path.join(
    base_data_folder,
    "dtu/Points/stl/stl%03d_total.ply"
)
pkl_format = os.path.join(
    base_evaluation_output_path,
    "dtu_baseline_%s_element_%s.pkl"
)

cloud_evaluation_thresholds = [2.0, ]

colmap_format = os.path.join(     
    base_data_folder,
    "dtu/colmap/geometric/depth/Depth/0.25/scan%d/rect_%03d_points.npy"
)
colmap_cloud = os.path.join(     
    base_data_folder,
    "dtu/Points/colmap/0.25/scan%d_total.ply"
)
mvsnet_format_depth = os.path.join(     
    base_data_folder,
    "dtu/MVSNet/Depth/0.25/scan%d/rect_%03d_points.npy"
)
mvsnet_format_trust = os.path.join(     
    base_data_folder,
    "dtu/MVSNet/Depth/0.25/scan%d/rect_%03d_points.trust.npy"
)
mvsnet_cloud = os.path.join(     
    base_data_folder,
    "dtu/Points/MVSNet/0.25/scan%d_total.ply"
)
gt_format = os.path.join(     
    base_data_folder,
    "dtu/Depth/0.25/scan%d/rect_%03d_points.npy"
)

from datasets.DTU import DTUAdapter

adapter = DTUAdapter(im_scale=0.25)

elements = adapter.split['test']

from random import shuffle
# we shuffle so that, if multiple instances are running, we get a semi-elegant
# parallelization (instances may still start the same element at the same time)
shuffle(elements)

def evaluate_colmap(element):
    pkl_filename = pkl_format % ("colmap", element)
    if os.path.exists(pkl_filename):
        print("COLMAP already evaluated on %s" % element)
        return

    center_views = range(49)
    print("Getting all image-space data and outputs")
    sys.stdout.flush()
    cameras = adapter.get_element_cameras(element).numpy()
    images = adapter.get_element_images(element).numpy() / 255

    # get the separate view clouds
    depths = []
    gt_depths = []
    for view in center_views:
        depths.append(np.load(colmap_format % (
            element, view
        )))
        gt_depths.append(np.load(gt_format % (
            element, view
        )))
    
    print("Casting the clouds into space")
    sys.stdout.flush()
    cloud_locs = []
    cloud_cols = []
    for view in center_views:
        cloud = depth_map_to_point_cloud(images[view], depths[view], cameras[view]).reshape(1,-1)
        cloud_locs.append(
            np.concatenate((
                cloud['x'],
                cloud['y'],
                cloud['z'],
            ),axis=0)
        )
        cloud_cols.append(
            np.concatenate((
                cloud['r'],
                cloud['g'],
                cloud['b'],
            ),axis=0)
        )
    
    print("Getting and filtering the gt cloud")
    sys.stdout.flush()
    gt_locs, gt_cols = load_pointcloud_ply_gipuma(gt_cloud_format % element)
    gt_clouds = {}
    for cloud_evaluation_threshold in cloud_evaluation_thresholds:
        gt_clouds[cloud_evaluation_threshold] = filter_cloud(gt_locs, gt_cols, actual_threshold=cloud_evaluation_threshold, actual_sparsity=cloud_evaluation_threshold/5)

    print("Evaluating the separate views")
    sys.stdout.flush()
    view_scores = {}
    for cloud_evaluation_threshold in cloud_evaluation_thresholds:
        view_scores[cloud_evaluation_threshold] = []

    for view in tqdm(center_views):
        # 1) chamfer distances for the per-view clouds: accuracy, completeness, percentage accurate, percentage complete
        for cloud_evaluation_threshold in cloud_evaluation_thresholds:
            sys.stdout.flush()
            locs = cloud_locs[view]
            cols = cloud_cols[view]
            locs, cols = filter_cloud(locs, cols, actual_threshold=cloud_evaluation_threshold, actual_sparsity=cloud_evaluation_threshold/5)

            accs = cloud_distance(locs, gt_locs)
            cmps = cloud_distance(gt_locs, locs)
            accs_mask = accs <= cloud_evaluation_threshold
            cmps_mask = cmps <= cloud_evaluation_threshold
            acc_sum = (accs * accs_mask).sum()
            acc_mask_sum = accs_mask.sum()
            cmp_sum = (cmps * cmps_mask).sum()
            cmp_mask_sum = cmps_mask.sum()

            scores = {}
            scores['cloud_acc_sum'] = acc_sum
            scores['cloud_acc_mask_sum'] = acc_mask_sum
            scores['cloud_cmp_sum'] = cmp_sum
            scores['cloud_cmp_mask_sum'] = cmp_mask_sum
            scores['cloud_acc_mask_count'] = accs_mask.size
            scores['cloud_cmp_mask_count'] = cmps_mask.size
            scores['cloud_acc_all_sum'] = accs.sum()
            scores['cloud_cmp_all_sum'] = cmps.sum()


            # 2) per-view L1 depth map evaluation: percentage of confident estimates that are correct, percentage of GT that is covered, average L1 distance on confidently covered GT depth
            errs = np.abs(depths[view] - gt_depths[view])
            confident = depths[view] > 0
            gt_available = gt_depths[view] > 0
            full_mask = confident * gt_available
            # from all confident pixels, how many are actually accurate (as far as we have GT) ?
            scores['depth_acc_ct'] = ((errs < cloud_evaluation_threshold) * full_mask).sum()
            scores['depth_acc_mask_sum'] = full_mask.sum()
            # from all known GT pixels, how many are adequately covered by a confident pixel ?
            scores['depth_cmp_ct'] = ((errs < cloud_evaluation_threshold) * full_mask).sum()
            scores['depth_cmp_mask_sum'] = gt_available.sum()
            scores['depth_err_sum'] = (errs * full_mask).sum()
            scores['depth_err_mask_sum'] = full_mask.sum()

            view_scores[cloud_evaluation_threshold].append(scores)

    print("Filtering and evaluating the full point cloud")
    sys.stdout.flush()
    # 3) chamfer distances for the final point cloud: accuracy, completeness, percentage accurate, percentage complete
    full_scores = {}
    full_locs, full_cols = load_pointcloud_ply_gipuma(colmap_cloud % element)
    for cloud_evaluation_threshold in tqdm(cloud_evaluation_thresholds):
        sys.stdout.flush()
        filtered_locs, filtered_cols = filter_cloud(full_locs, full_cols, actual_threshold=cloud_evaluation_threshold, actual_sparsity=cloud_evaluation_threshold/5)
        full_acc = cloud_distance(filtered_locs, gt_locs)
        full_cmp = cloud_distance(gt_locs, filtered_locs)
        full_acc_mask = full_acc <= cloud_evaluation_threshold
        full_cmp_mask = full_cmp <= cloud_evaluation_threshold
        full_avg_acc = (full_acc * full_acc_mask).sum() / max(1,full_acc_mask.sum())
        full_avg_cmp = (full_cmp * full_cmp_mask).sum() / max(1,full_cmp_mask.sum())
        full_pct_acc = full_acc_mask.mean()*100
        full_pct_cmp = full_cmp_mask.mean()*100
        
        full_scores[cloud_evaluation_threshold] = {
            'full_avg_acc': full_avg_acc,
            'full_avg_cmp': full_avg_cmp,
            'full_avg_acc_all': full_acc.mean(),
            'full_avg_cmp_all': full_cmp.mean(),
            'full_pct_acc': full_pct_acc,
            'full_pct_cmp': full_pct_cmp,
        }

        view_scores_h = view_scores[cloud_evaluation_threshold]

        # compile the average per-view values:
        avg_per_view_cloud_avg_acc = np.array([view['cloud_acc_sum'] for view in view_scores_h]).sum() / max(1, np.array([view['cloud_acc_mask_sum'] for view in view_scores_h]).sum())
        avg_per_view_cloud_avg_cmp = np.array([view['cloud_cmp_sum'] for view in view_scores_h]).sum() / max(1, np.array([view['cloud_cmp_mask_sum'] for view in view_scores_h]).sum())
        avg_per_view_cloud_avg_acc_all = np.array([view['cloud_acc_all_sum'] for view in view_scores_h]).sum() / max(1, np.array([view['cloud_acc_mask_count'] for view in view_scores_h]).sum())
        avg_per_view_cloud_avg_cmp_all = np.array([view['cloud_cmp_all_sum'] for view in view_scores_h]).sum() / max(1, np.array([view['cloud_cmp_mask_count'] for view in view_scores_h]).sum())
        avg_per_view_cloud_pct_acc = np.array([view['cloud_acc_mask_sum'] for view in view_scores_h]).sum() / max(1, np.array([view['cloud_acc_mask_count'] for view in view_scores_h]).sum()) * 100
        avg_per_view_cloud_pct_cmp = np.array([view['cloud_cmp_mask_sum'] for view in view_scores_h]).sum() / max(1, np.array([view['cloud_cmp_mask_count'] for view in view_scores_h]).sum()) * 100
        avg_per_view_depth_pct_acc = np.array([view['depth_acc_ct'] for view in view_scores_h]).sum() / max(1, np.array([view['depth_acc_mask_sum'] for view in view_scores_h]).sum()) * 100
        avg_per_view_depth_pct_cmp = np.array([view['depth_cmp_ct'] for view in view_scores_h]).sum() / max(1, np.array([view['depth_cmp_mask_sum'] for view in view_scores_h]).sum()) * 100
        avg_per_view_depth_avg_err = np.array([view['depth_err_sum'] for view in view_scores_h]).sum() / max(1, np.array([view['depth_err_mask_sum'] for view in view_scores_h]).sum())

        full_scores[cloud_evaluation_threshold]['avg_per_view_cloud_avg_acc'] = avg_per_view_cloud_avg_acc
        full_scores[cloud_evaluation_threshold]['avg_per_view_cloud_avg_cmp'] = avg_per_view_cloud_avg_cmp
        full_scores[cloud_evaluation_threshold]['avg_per_view_cloud_avg_acc_all'] = avg_per_view_cloud_avg_acc_all
        full_scores[cloud_evaluation_threshold]['avg_per_view_cloud_avg_cmp_all'] = avg_per_view_cloud_avg_cmp_all
        full_scores[cloud_evaluation_threshold]['avg_per_view_cloud_pct_acc'] = avg_per_view_cloud_pct_acc
        full_scores[cloud_evaluation_threshold]['avg_per_view_cloud_pct_cmp'] = avg_per_view_cloud_pct_cmp
        full_scores[cloud_evaluation_threshold]['avg_per_view_depth_pct_acc'] = avg_per_view_depth_pct_acc
        full_scores[cloud_evaluation_threshold]['avg_per_view_depth_pct_cmp'] = avg_per_view_depth_pct_cmp
        full_scores[cloud_evaluation_threshold]['avg_per_view_depth_avg_err'] = avg_per_view_depth_avg_err

    print("Dumping scores to file")
    sys.stdout.flush()
    with open(pkl_filename, 'wb') as f:
        pickle.dump(full_scores, f)
        pickle.dump(view_scores, f) # because I may want to access these -- might as well save them

    print("Finished with element %d" % element)
    sys.stdout.flush()

def evaluate_mvsnet(element):
    pkl_filename = pkl_format % ("mvsnet", element)

    if os.path.exists(pkl_filename):
        print("MVSNet already evaluated on %s" % element)
        return

    center_views = range(49)
    print("Getting all image-space data and outputs")
    sys.stdout.flush()
    cameras = adapter.get_element_cameras(element).numpy()
    images = adapter.get_element_images(element).numpy() / 255

    # get the separate view clouds
    depths = []
    gt_depths = []
    for view in center_views:
        depth = np.load(mvsnet_format_depth % (
            element, view
        ))
        trust = np.load(mvsnet_format_trust % (
            element, view
        ))
        depths.append(depth * (trust >= 0.8))
        gt_depths.append(np.load(gt_format % (
            element, view
        )))
    
    print("Casting the clouds into space")
    sys.stdout.flush()
    cloud_locs = []
    cloud_cols = []
    for view in center_views:
        cloud = depth_map_to_point_cloud(images[view], depths[view], cameras[view]).reshape(1,-1)
        cloud_locs.append(
            np.concatenate((
                cloud['x'],
                cloud['y'],
                cloud['z'],
            ),axis=0)
        )
        cloud_cols.append(
            np.concatenate((
                cloud['r'],
                cloud['g'],
                cloud['b'],
            ),axis=0)
        )
    
    print("Getting and filtering the gt cloud")
    sys.stdout.flush()
    gt_locs, gt_cols = load_pointcloud_ply_gipuma(gt_cloud_format % element)
    gt_clouds = {}
    for cloud_evaluation_threshold in cloud_evaluation_thresholds:
        gt_clouds[cloud_evaluation_threshold] = filter_cloud(gt_locs, gt_cols, actual_threshold=cloud_evaluation_threshold, actual_sparsity=cloud_evaluation_threshold/5)

    print("Evaluating the separate views")
    sys.stdout.flush()
    view_scores = {}
    for cloud_evaluation_threshold in cloud_evaluation_thresholds:
        view_scores[cloud_evaluation_threshold] = []

    for view in tqdm(center_views):
        # 1) chamfer distances for the per-view clouds: accuracy, completeness, percentage accurate, percentage complete
        for cloud_evaluation_threshold in cloud_evaluation_thresholds:
            sys.stdout.flush()
            locs = cloud_locs[view]
            cols = cloud_cols[view]
            locs, cols = filter_cloud(locs, cols, actual_threshold=cloud_evaluation_threshold, actual_sparsity=cloud_evaluation_threshold/5)

            accs = cloud_distance(locs, gt_locs)
            cmps = cloud_distance(gt_locs, locs)
            accs_mask = accs <= cloud_evaluation_threshold
            cmps_mask = cmps <= cloud_evaluation_threshold
            acc_sum = (accs * accs_mask).sum()
            acc_mask_sum = accs_mask.sum()
            cmp_sum = (cmps * cmps_mask).sum()
            cmp_mask_sum = cmps_mask.sum()

            scores = {}
            scores['cloud_acc_sum'] = acc_sum
            scores['cloud_acc_mask_sum'] = acc_mask_sum
            scores['cloud_cmp_sum'] = cmp_sum
            scores['cloud_cmp_mask_sum'] = cmp_mask_sum
            scores['cloud_acc_mask_count'] = accs_mask.size
            scores['cloud_cmp_mask_count'] = cmps_mask.size
            scores['cloud_acc_all_sum'] = accs.sum()
            scores['cloud_cmp_all_sum'] = cmps.sum()

            # 2) per-view L1 depth map evaluation: percentage of confident estimates that are correct, percentage of GT that is covered, average L1 distance on confidently covered GT depth
            errs = np.abs(depths[view] - gt_depths[view])
            confident = depths[view] > 0
            gt_available = gt_depths[view] > 0
            full_mask = confident * gt_available
            # from all confident pixels, how many are actually accurate (as far as we have GT) ?
            scores['depth_acc_ct'] = ((errs < cloud_evaluation_threshold) * full_mask).sum()
            scores['depth_acc_mask_sum'] = full_mask.sum()
            # from all known GT pixels, how many are adequately covered by a confident pixel ?
            scores['depth_cmp_ct'] = ((errs < cloud_evaluation_threshold) * full_mask).sum()
            scores['depth_cmp_mask_sum'] = gt_available.sum()
            scores['depth_err_sum'] = (errs * full_mask).sum()
            scores['depth_err_mask_sum'] = full_mask.sum()

            view_scores[cloud_evaluation_threshold].append(scores)

    print("Filtering and evaluating the full point cloud")
    sys.stdout.flush()
    # 3) chamfer distances for the final point cloud: accuracy, completeness, percentage accurate, percentage complete
    full_scores = {}
    full_locs, full_cols = load_pointcloud_ply(mvsnet_cloud % element)
    for cloud_evaluation_threshold in tqdm(cloud_evaluation_thresholds):
        sys.stdout.flush()
        filtered_locs, filtered_cols = filter_cloud(full_locs, full_cols, actual_threshold=cloud_evaluation_threshold, actual_sparsity=cloud_evaluation_threshold/5)
        full_acc = cloud_distance(filtered_locs, gt_locs)
        full_cmp = cloud_distance(gt_locs, filtered_locs)
        full_acc_mask = full_acc <= cloud_evaluation_threshold
        full_cmp_mask = full_cmp <= cloud_evaluation_threshold
        full_avg_acc = (full_acc * full_acc_mask).sum() / max(1,full_acc_mask.sum())
        full_avg_cmp = (full_cmp * full_cmp_mask).sum() / max(1,full_cmp_mask.sum())
        full_pct_acc = full_acc_mask.mean()*100
        full_pct_cmp = full_cmp_mask.mean()*100
        
        full_scores[cloud_evaluation_threshold] = {
            'full_avg_acc': full_avg_acc,
            'full_avg_cmp': full_avg_cmp,
            'full_avg_acc_all': full_acc.mean(),
            'full_avg_cmp_all': full_cmp.mean(),
            'full_pct_acc': full_pct_acc,
            'full_pct_cmp': full_pct_cmp,
        }

        view_scores_h = view_scores[cloud_evaluation_threshold]

        # compile the average per-view values:
        avg_per_view_cloud_avg_acc = np.array([view['cloud_acc_sum'] for view in view_scores_h]).sum() / max(1, np.array([view['cloud_acc_mask_sum'] for view in view_scores_h]).sum())
        avg_per_view_cloud_avg_cmp = np.array([view['cloud_cmp_sum'] for view in view_scores_h]).sum() / max(1, np.array([view['cloud_cmp_mask_sum'] for view in view_scores_h]).sum())
        avg_per_view_cloud_avg_acc_all = np.array([view['cloud_acc_all_sum'] for view in view_scores_h]).sum() / max(1, np.array([view['cloud_acc_mask_count'] for view in view_scores_h]).sum())
        avg_per_view_cloud_avg_cmp_all = np.array([view['cloud_cmp_all_sum'] for view in view_scores_h]).sum() / max(1, np.array([view['cloud_cmp_mask_count'] for view in view_scores_h]).sum())
        avg_per_view_cloud_pct_acc = np.array([view['cloud_acc_mask_sum'] for view in view_scores_h]).sum() / max(1, np.array([view['cloud_acc_mask_count'] for view in view_scores_h]).sum()) * 100
        avg_per_view_cloud_pct_cmp = np.array([view['cloud_cmp_mask_sum'] for view in view_scores_h]).sum() / max(1, np.array([view['cloud_cmp_mask_count'] for view in view_scores_h]).sum()) * 100
        avg_per_view_depth_pct_acc = np.array([view['depth_acc_ct'] for view in view_scores_h]).sum() / max(1, np.array([view['depth_acc_mask_sum'] for view in view_scores_h]).sum()) * 100
        avg_per_view_depth_pct_cmp = np.array([view['depth_cmp_ct'] for view in view_scores_h]).sum() / max(1, np.array([view['depth_cmp_mask_sum'] for view in view_scores_h]).sum()) * 100
        avg_per_view_depth_avg_err = np.array([view['depth_err_sum'] for view in view_scores_h]).sum() / max(1, np.array([view['depth_err_mask_sum'] for view in view_scores_h]).sum())

        full_scores[cloud_evaluation_threshold]['avg_per_view_cloud_avg_acc'] = avg_per_view_cloud_avg_acc
        full_scores[cloud_evaluation_threshold]['avg_per_view_cloud_avg_cmp'] = avg_per_view_cloud_avg_cmp
        full_scores[cloud_evaluation_threshold]['avg_per_view_cloud_avg_acc_all'] = avg_per_view_cloud_avg_acc_all
        full_scores[cloud_evaluation_threshold]['avg_per_view_cloud_avg_cmp_all'] = avg_per_view_cloud_avg_cmp_all
        full_scores[cloud_evaluation_threshold]['avg_per_view_cloud_pct_acc'] = avg_per_view_cloud_pct_acc
        full_scores[cloud_evaluation_threshold]['avg_per_view_cloud_pct_cmp'] = avg_per_view_cloud_pct_cmp
        full_scores[cloud_evaluation_threshold]['avg_per_view_depth_pct_acc'] = avg_per_view_depth_pct_acc
        full_scores[cloud_evaluation_threshold]['avg_per_view_depth_pct_cmp'] = avg_per_view_depth_pct_cmp
        full_scores[cloud_evaluation_threshold]['avg_per_view_depth_avg_err'] = avg_per_view_depth_avg_err

    print("Dumping scores to file")
    sys.stdout.flush()
    with open(pkl_filename, 'wb') as f:
        pickle.dump(full_scores, f)
        pickle.dump(view_scores, f) # because I may want to access these -- might as well save them

    print("Finished with element %d" % element)
    sys.stdout.flush()

for element in elements:
    evaluate_mvsnet(element)
    evaluate_colmap(element)
