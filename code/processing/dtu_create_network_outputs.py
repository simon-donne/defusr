import random
import torch
import cv2
import os
from tqdm import tqdm
import numpy as np
from itertools import chain
from experiment_handler import ExperimentHandler
from utils.ddf_logging import Logger
from utils.file_system import ensure_dir
from experiments.continue_experiment import get_latest_instance
from utils.depth_map_visualization import color_depth_map, color_error_image

from local_config import base_network_output_path as base_folder
from experiment_subset import experiment_list_DTU as ablations

GPU = torch.device('cuda')

def main():
    with Logger("", log_to_file=False) as debug_log:
        handlers = {}
        debug_log.print("Loading experiments...")
        for ablation in ablations:
            pkl_file = get_latest_instance(os.path.join(base_folder, ablation))
            handlers[ablation] = ExperimentHandler.load_experiment_from_file(pkl_file)
            handlers[ablation].network.scale_augmentation = 1.0
        debug_log.print("Experiments loaded.")

        # we'll assume for now that basic parameters are shared
        handler = handlers[ablations[0]]
        center_views = handler._data_loader.adapter.valid_centerviews
        elements = sorted(handler._data_loader.adapter._all_elements())

        random.shuffle(elements)

        skip_element = False

        with torch.no_grad():
            for idx, element in enumerate(tqdm(elements, miniters=1)):
                print(element)
                for ablation in ablations:
                    if skip_element:
                        skip_element=False
                        break
                    print(f"  {ablation}")
                    data_fcn = handlers[ablation]._config['data_loader_options']['data_function']
                    for center_view in center_views:
                        output_folder = os.path.join(handlers[ablation]._data_loader.adapter.datapath, ablation, "depth", "Depth", "%.2f" % handlers[ablation]._data_loader.adapter.im_scale, "scan%d" % element)
                        out_name = os.path.join(output_folder, "rect_%03d_points" % (center_view))
                        if(os.path.exists(out_name+".trust.png")):
                            continue
                        data = data_fcn(handlers[ablation]._data_loader, element, center_view=center_view)
                        in_data = [x.cuda() for x in data[0]]
                        try:
                            ensure_dir(output_folder)
                        except FileExistsError:
                            skip_element = True
                            print("Skipping due to directory creation collision")
                            break

                        out = handlers[ablation].network(*in_data)
                        refined_depth = out[0].detach().cpu().numpy().squeeze()
                        depth_trust = out[1].detach().cpu().numpy().squeeze()
                        np.save(
                            out_name + ".npy",
                            refined_depth
                        )
                        cv2.imwrite(
                            out_name + ".png",
                            color_depth_map(refined_depth)
                        )
                        np.save(
                            out_name + ".trust.npy",
                            depth_trust
                        )
                        cv2.imwrite(
                            out_name + ".trust.png",
                            color_error_image(np.power(2.0, 4 - depth_trust*8))
                        )
                    handlers[ablation]._data_loader.cache.clear()
                print("...finished")

if __name__ == "__main__":
    main()