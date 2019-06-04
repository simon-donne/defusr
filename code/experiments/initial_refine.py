
"""
"""

import os
import sys
import pickle
import torch
import cv2
import time
import numpy as np
from collections import deque
from datasets.DTU import DTUAdapter
from network_architectures.DepthMapInitialRefinement import DepthMapInitialRefinement
from utils.ddf_logging import Logger
from utils.timer import Timer
from losses.depth_gt import DepthGTLoss
from utils.debug import *

network = DepthMapInitialRefinement

class LossWrapper(torch.nn.Module):
    """
    Evaluates how good the trust classification is.
    """
    def __init__(self, threshold, crop_fraction=0.0, vmin=100, vmax=1000, limit=1000, do_trust=False, DTU_filter=False):
        super().__init__()
        gpu = torch.device('cuda')
        self.threshold = threshold
        self.BCEloss = torch.nn.BCELoss()
        self.gt_loss = DepthGTLoss(crop_fraction, vmin, vmax, limit)
        self.do_trust = do_trust
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3,stride=1,padding=1).to(gpu)
        self.maxpool_L = torch.nn.MaxPool2d(kernel_size=9,stride=1,padding=4).to(gpu)
        self.DTU_filter = DTU_filter

    def open(self, data):
        return self.maxpool(-self.maxpool(-data))

    def close_L(self, data):
        return -self.maxpool_L(-self.maxpool_L(data))

    def preprocess(self, data):
        # return data
        return self.open(data)

    def forward(self, input, color, depth_gt):
        depth_estimate = input[0]
        trust_estimate = input[1]

        gt_loss = self.gt_loss(depth_estimate, depth_gt)

        if self.do_trust:
            trust_gt = (depth_estimate - depth_gt).abs().lt(self.threshold).float()
            trust_gt = self.preprocess(trust_gt)

            ignore_mask = (trust_estimate > 0.95) * (trust_gt == 1) + (trust_estimate < 0.05) * (trust_gt == 0)

            if self.DTU_filter:
                gray_scale = color.mean(dim=1, keepdim=True)
                gt_well_defined = (((gray_scale <= 250) * (gray_scale >=24) + (depth_gt > 0)) > 0).float()
                gt_well_defined = self.close_L(gt_well_defined)
                gt_unknown = gt_well_defined * (depth_gt <= 0).float()
                gt_well_defined = gt_well_defined - gt_unknown

                # in DTU there are three cases for the groundtruth and the implied supervision:
                # [well_defined] ground truth available -- should be predicted correctly
                # [gt_unknown] -- should not be penalized
                # [gt_empty_space] -- should predict untrusted

                # when the groundtruth is one, we clamp below 0.95
                # when the groundtruth is zero we clamp above 0.05
                # in this way, we don't hopelessly lose pixels we are currently very sure about to other changes
                # in all areas where the gt is not well defined because of inherent issues, set to 'don't care'
                ignore_mask = (ignore_mask + (gt_unknown > 0)) > 0

                # in all areas where the gt is not well defined, set the target to 'do not trust'
                trust_gt = trust_gt * gt_well_defined

            ignore_mask = ignore_mask.float()
            trust_gt = trust_gt * (1-ignore_mask)
            trust_estimate = trust_estimate * (1-ignore_mask)

            trust_loss = self.BCEloss(trust_estimate, trust_gt)
        else:
            trust_loss = gt_loss.new_full(gt_loss.size(), 2./3)

        return torch.cat((trust_loss[None], gt_loss[None]))


def setup(handler):
    """
    Run every time before an experiment run is started.
    """
    handler._logger.print("Force loading the entire cache: images, depths and cameras")
    loader = handler._data_loader
    adapter = loader.adapter

    elements = [
        *adapter.split['train'][:loader.split_limits['train']],
        *adapter.split['test'][:loader.split_limits['test']],
    ]

    nr_views = adapter.nr_views
    with Timer(message="Cache loading", logger=handler._logger):
        for el_idx, element in enumerate(elements):
            loader.cache.get(adapter.get_element_cameras, (element, ))
            if el_idx % max(1,len(elements)//100) == 0:
                handler._logger.print("Element %d/%d" % (el_idx, len(elements)))
            for view in range(nr_views):
                loader.cache.get(adapter.get_single_image, (element, view))
                loader.cache.get(adapter.get_single_depth_map, (element, view, False))
                loader.cache.get(adapter.get_single_depth_map, (element, view, True))


def experiment(handler):
    current_epochs = handler.get('epochs_trained')
    settings = handler.get('experiment_settings')

    # handler.network.do_classification = False
    # handler.network.do_refinement = False
    # handler.loss_fcn.do_trust = False

    # the initial test run... Not strictly necessary, but nicer graphs
    if current_epochs == 0:
        handler.network.eval()
        handler.test()

    current_it = current_epochs // settings['it_size']

    timings = deque(maxlen=1)

    goal_its = settings['nr_its']
    if settings['checkpoint_its'] > 0:
        goal_its = min(current_it + settings['checkpoint_its'], settings['nr_its'])

    classification_its = settings['classification_fraction'] * settings['nr_its']
    refinement_its = settings['refinement_fraction'] * settings['nr_its']
    trust_its = settings['trust_fraction'] * settings['nr_its']

    first_time = time.time()
    for _i in range(current_it, goal_its):
        start_time = time.time()

        if _i >= classification_its and not handler.network.do_classification:
            handler.network.do_classification = True
            handler._logger.print("Turning on the classification head")
        if _i >= refinement_its and not handler.network.do_refinement:
            handler.network.do_refinement = True
            handler._logger.print("Turning on the refinement head")
        if _i >= trust_its and not handler.loss_fcn.do_trust:
            handler.loss_fcn.do_trust = True
            handler._logger.print("Turning on the trust loss")

        handler.network.train()
        handler.train(nr_epochs=settings['it_size'], direction=1)
        handler.network.eval()
        handler.test()
        handler.save_state_to_file()
        if settings['plot_losses']:
            handler.plot_losses()

        min_frac = 0
        def save_images(prefix, image, estimate_in, estimate_out, trust, gt):
            threshold = handler.get('loss_function_options')['threshold']
            gt_mask = gt > 0
            gt_validity = np.abs(estimate_in-gt)
            gt_validity = color_error_image(gt_validity, scale=threshold, mask=gt_mask)
            fn_fmt = handler.output_path + prefix + "_epoch%d_error_input.png" % handler.get('epochs_trained')
            cv2.imwrite(fn_fmt, gt_validity)
            gt_validity = np.abs(estimate_out-gt)
            gt_validity = color_error_image(gt_validity, scale=threshold, mask=gt_mask)
            fn_fmt = handler.output_path + prefix + "_epoch%d_error_output.png" % handler.get('epochs_trained')
            cv2.imwrite(fn_fmt, gt_validity)
            gt_validity = np.abs(estimate_out-gt) < threshold
            gt_validity = color_error_image(np.power(2.0,4 - gt_validity*8), mask=gt_mask)
            fn_fmt = handler.output_path + prefix + "_epoch%d_validity_gt.png" % handler.get('epochs_trained')
            cv2.imwrite(fn_fmt, gt_validity)
            fn_fmt = handler.output_path + prefix + "_epoch%d_input_image.png" % handler.get('epochs_trained')
            cv2.imwrite(fn_fmt, image)
            fn_fmt = handler.output_path + prefix + "_epoch%d_depth_estimate_in.png" % handler.get('epochs_trained')
            scale = gt.max().item()
            estimate_in = color_depth_map(estimate_in - scale*min_frac, scale=scale)
            cv2.imwrite(fn_fmt, estimate_in)
            fn_fmt = handler.output_path + prefix + "_epoch%d_depth_estimate_out.png" % handler.get('epochs_trained')
            estimate_out = color_depth_map(estimate_out - scale*min_frac, scale=scale)
            cv2.imwrite(fn_fmt, estimate_out)
            fn_fmt = handler.output_path + prefix + "_epoch%d_depth_gt.png" % handler.get('epochs_trained')
            gt = color_depth_map(gt - scale*min_frac, scale=scale)
            cv2.imwrite(fn_fmt, gt)
            fn_fmt = handler.output_path + prefix + "_epoch%d_validity_estimate.png" % handler.get('epochs_trained')
            trust = color_error_image(np.power(2.0, 4 - trust*8))
            cv2.imwrite(fn_fmt, trust)


        test_input_image = handler.last_test_in[0][0][0].detach().cpu().numpy().transpose((1,2,0))
        test_input_estimate = handler.last_test_in[1][0][0][0].detach().cpu().numpy()
        test_output_estimate = handler.last_test_out[0][0][0].detach().cpu().numpy()
        test_trust = handler.last_test_out[1][0][0].detach().cpu().numpy()
        test_gt = handler.last_test_gt[1][0][0].detach().cpu().numpy()
        save_images("test", test_input_image, test_input_estimate, test_output_estimate, test_trust, test_gt)

        train_input_image = handler.last_train_in[0][0][0].detach().cpu().numpy().transpose((1,2,0))
        train_input_estimate = handler.last_train_in[1][0][0][0].detach().cpu().numpy()
        train_output_estimate = handler.last_train_out[0][0][0].detach().cpu().numpy()
        train_trust = handler.last_train_out[1][0][0].detach().cpu().numpy()
        train_gt = handler.last_train_gt[1][0][0].detach().cpu().numpy()
        save_images("train", train_input_image, train_input_estimate, train_output_estimate, train_trust, train_gt)

        loop_time = time.time() - start_time
        timings.append(loop_time)
        handler._logger.print(
            "Projected time left: %d s" % int(np.mean(np.array(timings)[-5:]) * (goal_its - 1 - _i)),
            message_type=Logger.MESSAGE_INFO
        )
    last_time = time.time()
    handler._logger.print(
        "Took %d s for %d iterations (%d epochs)" % (last_time - first_time, goal_its - current_it, (goal_its - current_it) * settings['it_size']),
        message_type=Logger.MESSAGE_INFO
    )
    if goal_its == settings['nr_its']:
        sys.exit(0)
    else:
        handler._logger.print(
            "Premature exit for cluster efficiency."
        )
        sys.exit(3)


def data_function(self, element, **kwargs):
    input, output = self.data_function_refinement_full(element, **kwargs)
    output = (input[0][:,0], output[0])
    return input, output

