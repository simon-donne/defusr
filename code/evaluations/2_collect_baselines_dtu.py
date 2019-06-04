import itertools
import pickle
import numpy as np
import os

confidence_threshold = 2.0

from datasets.DTU import DTUAdapter
elements = DTUAdapter().split['test']

from local_config import base_evaluation_output_path
pkl_format = os.path.join(
    base_evaluation_output_path,
    "dtu_baseline_%s_element_%s.pkl"
)

baselines = ['colmap', 'mvsnet']
to_collect = [
    'full_pct_acc', 'full_pct_cmp',
    'full_avg_acc', 'full_avg_cmp',
    'full_avg_acc_all', 'full_avg_cmp_all',
    'avg_per_view_cloud_pct_acc', 'avg_per_view_cloud_pct_cmp',
    'avg_per_view_cloud_avg_acc', 'avg_per_view_cloud_avg_cmp',
    'avg_per_view_cloud_avg_acc_all', 'avg_per_view_cloud_avg_cmp_all',
    'avg_per_view_depth_pct_acc', 'avg_per_view_depth_pct_cmp',
]

for baseline in baselines:
    scores = None
    valid_elements = 0
    for element in elements:
        pkl_file = pkl_format % (
            baseline,
            element
        )
        try:
            with open(pkl_file, "rb") as pkl_file:
                full_scores = pickle.load(pkl_file)
        except FileNotFoundError:
            continue
        
        valid_elements += 1
        cloud_thresholds = list(full_scores.keys())

        if scores is None:
            scores = {}
            for data in to_collect:
                scores[data] = {}
                for cloud_threshold in cloud_thresholds:
                    scores[data][cloud_threshold] = 0

        for data in to_collect:
            for cloud_threshold in cloud_thresholds:
                scores[data][cloud_threshold] = scores[data][cloud_threshold] + full_scores[cloud_threshold][data]
    
    for cloud_threshold in cloud_thresholds:
        for data in to_collect:
            value = scores[data][cloud_threshold]
            value = value / valid_elements

            print("%s - cloud_threshold %.1f - %s: %f" % (
                baseline,
                cloud_threshold,
                data,
                value
            ))

print("Found %d/%d expected elements" % (valid_elements, len(elements)))