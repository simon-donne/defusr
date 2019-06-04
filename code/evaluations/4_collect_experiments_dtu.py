from experiment_subset import experiment_list_DTU as experiment_list
from experiments.continue_experiment import get_latest_instance
import itertools
import pickle
import numpy as np
import os

confidence_threshold = 2.0
trust_threshold = 0.5

from datasets.DTU import DTUAdapter
elements = DTUAdapter().split['test']

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

from local_config import base_data_folder, base_evaluation_output_path, base_network_output_path

pkl_format = os.path.join(
    base_evaluation_output_path, 
    "dtu_experiment_%s_element_%s.pkl"
)

for base_experiment in experiment_list:
    scores = None
    counts = None

    valid_elements = 0

    for element in elements:
        try:
            experiment_filename = get_latest_instance(os.path.join(base_network_output_path, base_experiment))
        except ValueError:
            print("Error getting updated instance for %s" % experiment_filename)
            continue
        
        pkl_file = pkl_format % (
            base_experiment,
            element
        )
        
        try:
            with open(pkl_file, "rb") as pkl_file:
                full_scores = pickle.load(pkl_file)
        except FileNotFoundError:
            continue

        valid_elements += 1
        
        cloud_thresholds = list(full_scores[trust_threshold].keys())

        if scores is None:
            scores = {}
            counts = {}
            for data in to_collect:
                scores[data] = {}
                counts[data] = {}
                for cloud_threshold in cloud_thresholds:
                    scores[data][cloud_threshold] = 0
                    counts[data][cloud_threshold] = 0

        for data in to_collect:
            for cloud_threshold in cloud_thresholds:
                scores[data][cloud_threshold] = scores[data][cloud_threshold] + full_scores[trust_threshold][cloud_threshold][data]
                counts[data][cloud_threshold] = counts[data][cloud_threshold] + 1
    
    for cloud_threshold in cloud_thresholds:
        for data in to_collect:
            value = scores[data][cloud_threshold] / counts[data][cloud_threshold]

            print("%s - %s: %f" % (
                base_experiment,
                data,
                value
            ))

print("Found %d/%d expected elements" % (valid_elements, len(elements)))