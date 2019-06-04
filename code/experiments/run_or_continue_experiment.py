"""
Start the experiment, or continue it if it already exists
For a cluster where jobs are restarted on a regular basis
"""

import sys
import importlib.util
from experiment_handler import ExperimentHandler
from experiments.run_experiment import get_config
from experiments.continue_experiment import get_latest_instance, continue_experiment_pkl
from utils.ddf_logging import Logger
import copy

def run_or_continue(experiment_name):
    config = get_config(experiment_name=experiment_name)

    # check whether there are already instances
    try:
        instance_pkl = get_latest_instance(config['output_folder']+config['experiment_name'] + '/')
        print("Continuing experiment %s" % instance_pkl)
        continue_experiment_pkl(instance_pkl)
    except:
        experiment = ExperimentHandler(config)
        print("Starting new instance of experiment %s" % config['experiment_name'])
        experiment.run()
    print("Experiment concluded.")

if __name__ == "__main__":
    run_or_continue(experiment_name=sys.argv[1])
