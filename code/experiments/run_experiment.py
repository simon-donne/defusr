"""
Module responsible for actually running experiments.
Pass the experiment config file as a command-line argument, or hardcode it.
"""

import sys
import importlib.util
from experiment_handler import ExperimentHandler
from experiments.lists.all_experiments_list import base_config, experiment_list
from utils.ddf_logging import Logger
import copy

def main():
    "Loads the config file in sys.argv[1], and runs the experiment."
    config = get_config()
    experiment = ExperimentHandler(config)

    # for tmux sessions: set the pane name
    print("\033]2;%s\033\\" % config['experiment_name'])

    experiment.run()
    print("Experiment concluded.")


def update_experiment_dictionary(target, source):
    for setting in source:
        value = source[setting]
        if target.__contains__(setting) and isinstance(value, dict):
            update_experiment_dictionary(target[setting], value)
        else:
            target[setting] = value


def get_config(experiment_name=None):
    """
    Imports a given config file.
    Should be a module name of the form 'experiments.<name>'.
    """
    if experiment_name is None:
        if len(sys.argv) == 2:
            experiment_name = sys.argv[1]
        else:
            raise UserWarning("Correct usage: python run_experiment.py experiment_name")

    try:
        experiment_settings = experiment_list[experiment_name]
    except KeyError:
        raise UserWarning("The experiment '%s' is unknown." % experiment_name)

    experiment_settings = experiment_settings.copy()
    framework_file = "experiments." + experiment_settings.pop('experiment_framework')
    framework_module = importlib.import_module(framework_file)

    config = copy.deepcopy(base_config)

    # fill in the modified settings for this experiment
    update_experiment_dictionary(config, experiment_settings)

    # fill in the relevant functions
    config['experiment_name'] = experiment_name
    config['network'] = framework_module.network
    config['setup'] = framework_module.setup
    config['experiment'] = framework_module.experiment
    config['loss_function'] = framework_module.LossWrapper
    config['data_loader_options']['data_function'] = framework_module.data_function

    return config

if __name__ == "__main__":
    main()
