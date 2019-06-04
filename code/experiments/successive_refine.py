"""
"""

from network_architectures.DepthMapSuccessiveRefinement import DepthMapSuccessiveRefinement
import experiments.initial_refine as initial_refine_framework

network = DepthMapSuccessiveRefinement
setup = initial_refine_framework.setup
experiment = initial_refine_framework.experiment
LossWrapper = initial_refine_framework.LossWrapper

def data_function(self, element, **kwargs):
    input, output = self.data_function_successive_refinement_full(element, **kwargs)
    output = (input[0][:,0], output[0])
    return input, output
