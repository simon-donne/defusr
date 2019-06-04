from experiment_handler import ExperimentHandler
from utils.chamfer_evaluation import filter_cloud, cloud_distance
from utils.depth_maps import depth_map_to_point_cloud
from utils.ply import load_pointcloud_ply_gipuma
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
    "dtu_experiment_%s_element_%s.pkl"
)

trust_thresholds = [0.50, ]
cloud_evaluation_thresholds = [2.0, ]

def evaluate_experiment(handler, base_experiment_name, element, index=None):
    data_fcn = handler._config['data_loader_options']['data_function']
    center_views = range(handler._data_loader.adapter.nr_views)

    pkl_filename = pkl_format % (
        base_experiment_name.replace('/', '-'),
        element
    )
    if os.path.exists(pkl_filename):
        print("Pickle %s already exists. Skipping." % pkl_filename)
        sys.stdout.flush()
        return
    if index is not None:
        print("Index %d is missing" % index)
        return

    print("Getting all image-space data and outputs")
    sys.stdout.flush()
    cameras = handler._data_loader.adapter.get_element_cameras(element).numpy()
    images = handler._data_loader.adapter.get_element_images(element).numpy() / 255

    # get the separate view clouds
    depths = []
    trusts = []
    gt_depths = []
    for view in center_views:
        data = data_fcn(handler._data_loader, element, center_view=view)
        network_input = [x.cuda() for x in data[0]]
        out = handler.network(*network_input)
        
        depths.append(out[0].detach().cpu().numpy().squeeze())
        trusts.append(out[1].detach().cpu().numpy().squeeze())
        gt_depths.append(data[1][1].detach().cpu().numpy().squeeze())
    
    print("Casting the clouds into space")
    sys.stdout.flush()
    cloud_locs = {}
    cloud_cols = {}
    for trust_threshold in trust_thresholds:
        cloud_locs[trust_threshold] = []
        cloud_cols[trust_threshold] = []
    for view in center_views:
        for trust_threshold in trust_thresholds:
            trusted_depth = depths[view] * (trusts[view] > trust_threshold)
            cloud = depth_map_to_point_cloud(images[view], trusted_depth, cameras[view]).reshape(1,-1)
            cloud_locs[trust_threshold].append(
                np.concatenate((
                    cloud['x'],
                    cloud['y'],
                    cloud['z'],
                ),axis=0)
            )
            cloud_cols[trust_threshold].append(
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
    for trust_threshold in trust_thresholds:
        view_scores[trust_threshold] = {}
        for cloud_evaluation_threshold in cloud_evaluation_thresholds:
            view_scores[trust_threshold][cloud_evaluation_threshold] = []

    for view in tqdm(center_views):
        # 1) chamfer distances for the per-view clouds: accuracy, completeness, percentage accurate, percentage complete
        for trust_threshold in trust_thresholds:
            for cloud_evaluation_threshold in cloud_evaluation_thresholds:
                sys.stdout.flush()
                locs = cloud_locs[trust_threshold][view]
                cols = cloud_cols[trust_threshold][view]
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
                confident = trusts[view] > trust_threshold
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

                view_scores[trust_threshold][cloud_evaluation_threshold].append(scores)

    print("Filtering and evaluating the full point cloud")
    sys.stdout.flush()
    # 3) chamfer distances for the final point cloud: accuracy, completeness, percentage accurate, percentage complete
    full_scores = {}
    for trust_threshold in tqdm(trust_thresholds):
        full_scores[trust_threshold] = {}
        full_locs = np.concatenate(cloud_locs[trust_threshold], axis=1)
        full_cols = np.concatenate(cloud_cols[trust_threshold], axis=1)
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
            
            full_scores[trust_threshold][cloud_evaluation_threshold] = {
                'full_avg_acc': full_avg_acc,
                'full_avg_cmp': full_avg_cmp,
                'full_avg_acc_all': full_acc.mean(),
                'full_avg_cmp_all': full_cmp.mean(),
                'full_pct_acc': full_pct_acc,
                'full_pct_cmp': full_pct_cmp,
            }

            view_scores_h = view_scores[trust_threshold][cloud_evaluation_threshold]

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

            full_scores[trust_threshold][cloud_evaluation_threshold]['avg_per_view_cloud_avg_acc'] = avg_per_view_cloud_avg_acc
            full_scores[trust_threshold][cloud_evaluation_threshold]['avg_per_view_cloud_avg_cmp'] = avg_per_view_cloud_avg_cmp
            full_scores[trust_threshold][cloud_evaluation_threshold]['avg_per_view_cloud_avg_acc_all'] = avg_per_view_cloud_avg_acc_all
            full_scores[trust_threshold][cloud_evaluation_threshold]['avg_per_view_cloud_avg_cmp_all'] = avg_per_view_cloud_avg_cmp_all
            full_scores[trust_threshold][cloud_evaluation_threshold]['avg_per_view_cloud_pct_acc'] = avg_per_view_cloud_pct_acc
            full_scores[trust_threshold][cloud_evaluation_threshold]['avg_per_view_cloud_pct_cmp'] = avg_per_view_cloud_pct_cmp
            full_scores[trust_threshold][cloud_evaluation_threshold]['avg_per_view_depth_pct_acc'] = avg_per_view_depth_pct_acc
            full_scores[trust_threshold][cloud_evaluation_threshold]['avg_per_view_depth_pct_cmp'] = avg_per_view_depth_pct_cmp
            full_scores[trust_threshold][cloud_evaluation_threshold]['avg_per_view_depth_avg_err'] = avg_per_view_depth_avg_err

    print("Dumping scores to file")
    sys.stdout.flush()
    with open(pkl_filename, 'wb') as f:
        pickle.dump(full_scores, f)
        pickle.dump(view_scores, f) # because I may want to access these -- might as well save them

    print("Finished with element %d" % element)
    sys.stdout.flush()
