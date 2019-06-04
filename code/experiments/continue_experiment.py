"""
Continues a partially-run experiment from a given point.
If no specific experiment instance is given (only the experiment name),
automatically finds the latest experiment and the latest epoch, and
continues from there.
"""

import glob
import os
from experiment_handler import ExperimentHandler
from experiments.run_experiment import get_config
from utils.ddf_logging import Logger
import sys

def get_latest_instance_base(basename):
    config = get_config(experiment_name=basename)
    return get_latest_instance(config['output_folder']+config['experiment_name'] + '/')

def get_latest_instance(folder):
    date_folder = None
    for date_folder in sorted(glob.glob(os.path.join(folder,"2*/"))): #2*:  2018-something
        pass
    if date_folder is None:
        raise ValueError("This experiment does not have any instances yet!")
    pkl_file = None
    for pkl_file in sorted(glob.glob(os.path.join(date_folder, "experiment_state_epoch_*.pkl"))):
        pass
    if pkl_file is None:
        raise ValueError("This experiment does not have any instances yet!")
    return pkl_file


def continue_experiment(folder, instance=None):
    if instance is None:
        instance = get_latest_instance(folder)
    experiment = ExperimentHandler.load_experiment_from_file(os.path.join(folder, instance))

    # # just for debugging now
    # with experiment:
    #     experiment._logger.print("[ WARNING | DEBUG ] single-element train/test sets", Logger.MESSAGE_WARNING )
    # loader = experiment._data_loader
    
    # loader.split_limits['train'] = 1
    # loader.split_limits['test'] = 1

    experiment.run()

def continue_experiment_pkl(filename):
    experiment = ExperimentHandler.load_experiment_from_file(filename)

    # # just for debugging now
    # with experiment:
    #     experiment._logger.print("[ WARNING | DEBUG ] single-element train/test sets", Logger.MESSAGE_WARNING )
    # loader = experiment._data_loader
    
    # loader.split_limits['train'] = 1
    # loader.split_limits['test'] = 1

    experiment.run()


def main():
    if len(sys.argv) < 2:
        print("WARNING> pass an experiment to continue")
    else:
        continue_experiment_pkl(sys.argv[1])

if __name__ == "__main__":
    main()
