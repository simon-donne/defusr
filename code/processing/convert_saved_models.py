"""
Script to convert saved networks from the original framework to the public one.
Mainly hard casting to the now-correct type, as well as fixing the folders.
"""

import os
import pickle
from glob import glob

input_folder = "/home/sdonne/Desktop/defusr/models_orig"
from local_config import base_network_output_path as output_folder
from experiment_handler import ExperimentHandler
from collections import defaultdict

class MYTHUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'network_architectures.DepthMapRefinement_v2_no_center_scaling_successive':
            module = 'network_architectures.DepthMapSuccessiveRefinement'
        elif module == 'network_architectures.DepthMapRefinement_v2_no_center_scaling':
            module = 'network_architectures.DepthMapInitialRefinement'
        elif module == 'network_architectures.DepthTrustLocal_v3':
            module = 'network_architectures.DepthMapInitialTrust'
        elif module == 'experiments.local_depthtrust':
            module = 'experiments.initial_depthtrust'
        if name == 'DepthMapRefinement_v2_successive':
            name = 'DepthMapSuccessiveRefinement'
        elif name == 'DepthMapRefinement_v2':
            name = 'DepthMapInitialRefinement'
        elif name == 'DepthTrustLocal_v3':
            name = 'DepthMapInitialTrust'
        
        return super().find_class(module, name)

models = sorted(glob(os.path.join(input_folder, "*/*/experiment_state_*.pkl")))

for model in models:
    handler = ExperimentHandler.load_experiment_from_file(model, folder_override=input_folder , unpickler=MYTHUnpickler)
    handler._config['output_folder'] = output_folder
    handler.output_path = os.path.join(output_folder, handler.get_experiment_identifier())
    handler._logger.log_path.replace(input_folder, output_folder)
    # we also clear the optimization state, to save on space (~factor 3)
    handler.optimizer.state = defaultdict(dict)
    handler.save_state_to_file()

    # quick test: load this new one
    handler = ExperimentHandler.load_experiment_from_file(model.replace(input_folder, output_folder))
    
    print("processed %s" % (model))