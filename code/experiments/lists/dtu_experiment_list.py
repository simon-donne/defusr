import torch
from utils.ddf_logging import Logger
from datasets.DTU import DTUAdapter

from local_config import base_network_output_path
import os

experiment_list = {}

experiment_list["DTU_colmap_local_depthtrust"] = {
    'experiment_framework': "local_depthtrust",
    'experiment_settings': {
        'it_size': 200,  # how many epochs per iteration ?
        'nr_its': 100,  # how many iterations?
    },
    'optimizer_options': {
        'lr': 1e-3,
    },
    'optimizer_lr_milestones': {
        20000*0.4: 1e-3/2,
        20000*0.6: 1e-3/4,
        20000*0.8: 1e-3/8,
        20000*0.9: 1e-3/16,
    },
    'data_adapter_options': {
        'depth_map_prefix': "colmap/photometric/depth/",
    },
    'data_loader_options': {
        'split_limits': {'train': None, 'test': None, 'val': None},
    },
    'loss_function_options': {
        'threshold': 5.0,
        'DTU_filter': True,
    },
    'network_options': {
        'F': 16,
        'scale_augmentation': 2.0,
        'depth_scale': 100,
    },
    'data_adapter': DTUAdapter,
}

experiment_list["DTU_colmap_depth_refine_run1"] = {
    'experiment_framework': "initial_refine",
    'experiment_settings': {
        'it_size': 100,
        'nr_its': 100,
        'classification_fraction': 0.3,
        'refinement_fraction': 0.4,
        'trust_fraction': 0.5,
    },
    'optimizer_options': {
        'lr': 1e-4,
    },
    'optimizer_lr_milestones': {
        10000*0.4: 1e-4/2,
        10000*0.6: 1e-4/4,
        10000*0.8: 1e-4/8,
        10000*0.9: 1e-4/16,
    },
    'data_adapter_options': {
        'depth_map_prefix': "colmap/photometric/depth/",
        '_neighbour_selection': "mixed",
    },
    'data_loader_options': {
        'minibatch_size': 2,
        'nr_neighbours': 12,
        'split_limits': {'train': None, 'test': None, 'val': None},
    },
    'loss_function_options': {
        'threshold': 5.0,
        'vmin': 5.0,
        'vmax': 2000.0,
        'limit': 2000.0,
        'DTU_filter': True,
        'do_trust': False,
    },
    'network_options': {
        'local_network': os.path.join(base_network_output_path, 'DTU_colmap_local_depthtrust/20190111-1016-bae88c/experiment_state_epoch_20000.pkl'),
        'scale_augmentation': 2.0,
        'depth_scale': 100,
        'F': 16,
        'reset_trust': True,
    },
    'data_adapter': DTUAdapter,
}

experiment_list["DTU_colmap_depth_refine_run2"] = {
    'experiment_framework': "successive_refine",
    'experiment_settings': {
        'it_size': 100,
        'nr_its': 100,
        'classification_fraction': 0.3,
        'refinement_fraction': 0.4,
        'trust_fraction': 0.5,
    },
    'optimizer_options': {
        'lr': 1e-4,
    },
    'optimizer_lr_milestones': {
        10000*0.4: 1e-4/2,
        10000*0.6: 1e-4/4,
        10000*0.8: 1e-4/8,
        10000*0.9: 1e-4/16,
    },
    'data_adapter_options': {
        'depth_map_prefix': "DTU_colmap_depth_refine_run1/depth/",
        '_neighbour_selection': "mixed",
    },
    'data_loader_options': {
        'minibatch_size': 2,
        'nr_neighbours': 12,
        'split_limits': {'train': None, 'test': None, 'val': None},
    },
    'loss_function_options': {
        'threshold': 5.0,
        'vmin': 5.0,
        'vmax': 2000.0,
        'limit': 2000.0,
        'DTU_filter': True,
        'do_trust': False,
    },
    'network_options': {
        'file': os.path.join(base_network_output_path, 'DTU_colmap_depth_refine_run1/20190120-2200-bae88c/experiment_state_epoch_10000.pkl'),
        'scale_augmentation': 2.0,
        'depth_scale': 100,
        'F': 16,
    },
    'data_adapter': DTUAdapter,
}

experiment_list["DTU_colmap_depth_refine_run3"] = {
    'experiment_framework': "successive_refine",
    'experiment_settings': {
        'it_size': 100,
        'nr_its': 50,
        'classification_fraction': 0.3,
        'refinement_fraction': 0.4,
        'trust_fraction': 0.5,
    },
    'optimizer_options': {
        'lr': 1e-4,
    },
    'optimizer_lr_milestones': {
        5000*0.4: 1e-4/2,
        5000*0.6: 1e-4/4,
        5000*0.8: 1e-4/8,
        5000*0.9: 1e-4/16,
    },
    'data_adapter_options': {
        'depth_map_prefix': "DTU_colmap_depth_refine_run2/depth/",
        '_neighbour_selection': "mixed",
    },
    'data_loader_options': {
        'minibatch_size': 2,
        'nr_neighbours': 12,
        'split_limits': {'train': None, 'test': None, 'val': None},
    },
    'loss_function_options': {
        'threshold': 5.0,
        'vmin': 5.0,
        'vmax': 2000.0,
        'limit': 2000.0,
        'DTU_filter': True,
        'do_trust': False,
    },
    'network_options': {
        'file': os.path.join(base_network_output_path, 'DTU_colmap_depth_refine_run2/20190126-1304-ebe210/experiment_state_epoch_10000.pkl'),
        'scale_augmentation': 2.0,
        'depth_scale': 100,
        'F': 16,
    },
    'data_adapter': DTUAdapter,
}

experiment_list["DTU_mvsnet_local_depthtrust"] = {
    'experiment_framework': "local_depthtrust",
    'experiment_settings': {
        'it_size': 200,  # how many epochs per iteration ?
        'nr_its': 100,  # how many iterations?
    },
    'optimizer_options': {
        'lr': 1e-3,
    },
    'optimizer_lr_milestones': {
        20000*0.4: 1e-3/2,
        20000*0.6: 1e-3/4,
        20000*0.8: 1e-3/8,
        20000*0.9: 1e-3/16,
    },
    'data_adapter_options': {
        'depth_map_prefix': "mvsnet/depth/",
    },
    'data_loader_options': {
        'split_limits': {'train': None, 'test': None, 'val': None},
    },
    'loss_function_options': {
        'threshold': 5.0,
        'DTU_filter': True,
    },
    'network_options': {
        'F': 16,
        'scale_augmentation': 2.0,
        'depth_scale': 100,
    },
    'data_adapter': DTUAdapter,
}

experiment_list["DTU_mvsnet_depth_refine_run1"] = {
    'experiment_framework': "initial_refine",
    'experiment_settings': {
        'it_size': 100,
        'nr_its': 100,
        'classification_fraction': 0.3,
        'refinement_fraction': 0.4,
        'trust_fraction': 0.5,
    },
    'optimizer_options': {
        'lr': 1e-4,
    },
    'optimizer_lr_milestones': {
        10000*0.4: 1e-4/2,
        10000*0.6: 1e-4/4,
        10000*0.8: 1e-4/8,
        10000*0.9: 1e-4/16,
    },
    'data_adapter_options': {
        'depth_map_prefix': "MVSNet/",
        '_neighbour_selection': "mixed",
    },
    'data_loader_options': {
        'minibatch_size': 2,
        'nr_neighbours': 12,
        'split_limits': {'train': None, 'test': None, 'val': None},
    },
    'loss_function_options': {
        'threshold': 5.0,
        'vmin': 5.0,
        'vmax': 2000.0,
        'limit': 2000.0,
        'DTU_filter': True,
        'do_trust': False,
    },
    'network_options': {
        'local_network': os.path.join(base_network_output_path, 'DTU_mvsnet_local_depthtrust/20190111-1016-bae88c/experiment_state_epoch_20000.pkl'),
        'scale_augmentation': 2.0,
        'depth_scale': 100,
        'F': 16,
        'reset_trust': True,
    },
    'data_adapter': DTUAdapter,
}

experiment_list["DTU_mvsnet_depth_refine_run2"] = {
    'experiment_framework': "successive_refine",
    'experiment_settings': {
        'it_size': 100,
        'nr_its': 100,
        'classification_fraction': 0.3,
        'refinement_fraction': 0.4,
        'trust_fraction': 0.5,
    },
    'optimizer_options': {
        'lr': 1e-4,
    },
    'optimizer_lr_milestones': {
        10000*0.4: 1e-4/2,
        10000*0.6: 1e-4/4,
        10000*0.8: 1e-4/8,
        10000*0.9: 1e-4/16,
    },
    'data_adapter_options': {
        'depth_map_prefix': "DTU_mvsnet_depth_refine_run1/depth/",
        '_neighbour_selection': "mixed",
    },
    'data_loader_options': {
        'minibatch_size': 2,
        'nr_neighbours': 12,
        'split_limits': {'train': None, 'test': None, 'val': None},
    },
    'loss_function_options': {
        'threshold': 5.0,
        'vmin': 5.0,
        'vmax': 2000.0,
        'limit': 2000.0,
        'DTU_filter': True,
        'do_trust': False,
    },
    'network_options': {
        'file': os.path.join(base_network_output_path, 'DTU_mvsnet_depth_refine_run1/20190120-2223-bae88c/experiment_state_epoch_10000.pkl'),
        'scale_augmentation': 2.0,
        'depth_scale': 100,
        'F': 16,
    },
    'data_adapter': DTUAdapter,
}

experiment_list["DTU_mvsnet_depth_refine_run3"] = {
    'experiment_framework': "successive_refine",
    'experiment_settings': {
        'it_size': 100,
        'nr_its': 50,
        'classification_fraction': 0.3,
        'refinement_fraction': 0.4,
        'trust_fraction': 0.5,
    },
    'optimizer_options': {
        'lr': 1e-4,
    },
    'optimizer_lr_milestones': {
        5000*0.4: 1e-4/2,
        5000*0.6: 1e-4/4,
        5000*0.8: 1e-4/8,
        5000*0.9: 1e-4/16,
    },
    'data_adapter_options': {
        'depth_map_prefix': "DTU_mvsnet_depth_refine_run2/depth/",
        '_neighbour_selection': "mixed",
    },
    'data_loader_options': {
        'minibatch_size': 2,
        'nr_neighbours': 12,
        'split_limits': {'train': None, 'test': None, 'val': None},
    },
    'loss_function_options': {
        'threshold': 5.0,
        'vmin': 5.0,
        'vmax': 2000.0,
        'limit': 2000.0,
        'DTU_filter': True,
        'do_trust': False,
    },
    'network_options': {
        'file': os.path.join(base_network_output_path, 'DTU_mvsnet_depth_refine_run2/20190126-1304-ebe210/experiment_state_epoch_10000.pkl'),
        'scale_augmentation': 2.0,
        'depth_scale': 100,
        'F': 16,
    },
    'data_adapter': DTUAdapter,
}
