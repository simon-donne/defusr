from evaluations._evaluate_experiment_dtu import evaluate_experiment
from datasets.DTU import DTUAdapter as Adapter

from experiment_subset import experiment_list_DTU as experiment_list
from experiments.continue_experiment import get_latest_instance_base
from experiment_handler import ExperimentHandler
import sys
import itertools

elements = Adapter().split['test']

combos = list(itertools.product(experiment_list, elements))

if len(sys.argv) > 1:
    idx = int(sys.argv[1])
    indices = [idx, ]
    combos = combos[idx:idx+1]
else:
    from random import shuffle
    indices = list(range(len(combos)))
    shuffle(indices)
    combos = [combos[number] for number in indices]

for number, combo in enumerate(combos):
    base_experiment = combo[0]
    element = combo[1]
    print("Doing %d/%d: %s -- %d" % (indices[number], len(indices), base_experiment, element))

    try:
        experiment_filename = get_latest_instance_base(base_experiment)
    except ValueError:
        print("Error getting updated instance for %s" % base_experiment)
        continue

    handler = ExperimentHandler.load_experiment_from_file(experiment_filename)
    # turn off scale augmentation if it is turned on
    handler.network.scale_augmentation = 1

    evaluate_experiment(handler, base_experiment, element)
