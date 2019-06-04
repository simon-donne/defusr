"""
The settings for all possible experiments in one big dictionary.
Common settings (base output folder and similar things) are in base_config.
"""

import torch
from utils.ddf_logging import Logger

from local_config import base_network_output_path

base_config = {
    # general settings
    'output_folder': base_network_output_path,
    'log_to_file': True,
    'console_verbosity': Logger.MESSAGE_INFO,
    'experiment_settings': {
        # how many iterations before we shutdown ourselves?
        # for short-job cluster systems
        'checkpoint_its': 0,
        'plot_losses': True,
    },

    # data loader
    'data_adapter_options': {
        'im_scale': 0.25,
        'ensure_multiple': 1,
    },
    'data_loader_options': {
        'split_limits': {'train': None, 'test': None, 'val': None},
        'minibatch_size': 8,
        'caching': True,
        'gpu_caching': False,
        'center_crop_size': None,
    },

    # training
    'optimizer': torch.optim.Adam,
}

experiment_list = {}

from experiments.lists.dtu_experiment_list import experiment_list as dtu_experiment_list
experiment_list.update(dtu_experiment_list)