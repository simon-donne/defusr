from itertools import chain
import torch
import numpy as np
import glob
import os
import MYTH
from datasets.dataset_adapter import DatasetAdapter

from local_config import base_data_folder
import os

class FlyingThingsAdapter(DatasetAdapter):
    """Adapter for the synthetic Flying Things dataset."""

    base_datapath = os.path.join(base_data_folder, 'flying_things_MVS/')
    im_width = 1920
    im_height = 1080
    nr_views = 10
    im_scale = 0.25

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        input_scale = self.im_scale
        if input_scale <= 0.25:
            input_scale = 0.25
        elif input_scale <= 0.50: 
            input_scale = 0.50
        self.camera_scaling = self.im_scale / input_scale
        self.datapath = FlyingThingsAdapter.base_datapath.replace('flying_things_MVS', 'flying_things_MVS_%.02f' % input_scale)
        self._set_default_splits()


    def _set_default_splits(self):
        all_elements = self._all_elements()
        testing_size = int(len(all_elements)*0.02)
        val_size = int(len(all_elements)*0.28)
        self.split['test'] = all_elements[:testing_size]
        if(val_size > 0):
            self.split['val'] = all_elements[-val_size:]
            self.split['train'] = all_elements[testing_size:-val_size]
        else:
            self.split['train'] = all_elements[testing_size:]


    def _all_elements(self):
        return sorted(glob.glob(self.datapath + "*"))


    def get_dataset_name(self):
        return "FlyingThingsMVS"


    @staticmethod
    def get_scene_seed(element):
        return element.split('_')[-1]


    def get_image_path(self, element, view):
        image_path = os.path.join(element, 
            "scene_%s_frame_%03d.png" % (
                self.get_scene_seed(element),
                view + 1
            )
        )
        return image_path


    def get_depth_map_path(self, element, view, gt=True):
        depth_map_path = os.path.join(element, 
            "%sscene_%s_frame_%03d_depth.npy" % (
                "" if gt else self.depth_map_prefix,
                self.get_scene_seed(element),
                view + 1
            )
        )
        return depth_map_path


    def get_normal_map_path(self, element, view, gt=True):
        raise NotImplementedError("Normals not implemented for Flying Things MVS")


    def get_element_cameras(self, element):
        cameras = []
        scaler = np.array([[self.camera_scaling, 0, 0], [0, self.camera_scaling, 0], [0, 0, 1]])
        for view in range(self.nr_views):
            camera_filename = os.path.join(element, 
                "scene_%s_frame_%03d.png.P" % (
                    self.get_scene_seed(element),
                    view + 1
                )
            )
            camera_matrix = np.matmul(scaler, np.loadtxt(camera_filename))
            cameras.append(torch.from_numpy(camera_matrix.astype(np.float32)).unsqueeze(0))
        return torch.cat(cameras, 0)


    def get_element_worldtf(self, element):
        world_transform = torch.from_numpy(np.eye(4).astype(np.float32))
        return world_transform

    valid_centerviews = range(0, nr_views)

    def get_view_neighbours(self, cameras, center_view, nr_neighbours):
        if nr_neighbours == 0:
            return []

        cameras = cameras.cuda()
        B = cameras.shape[0]
        camlocs = cameras.new_empty(B, 3, 1)
        invKRs = cameras.new_empty(B, 3, 3)
        MYTH.InvertCams_gpu(cameras, invKRs, camlocs)
        
        distances = (camlocs - camlocs[center_view:center_view+1,:,:]).pow(2).sum(dim=1).sum(dim=1)
        distances = [d.item() for d in distances]
        
        orders = sorted(range(len(distances)), key=distances.__getitem__)
        if nr_neighbours >= len(distances):
            return orders
        if self._neighbour_selection == "closest":
            return orders[1:1+nr_neighbours]
        elif self._neighbour_selection == "furthest":
            return orders[-nr_neighbours:]
        elif self._neighbour_selection == "mixed":
            return orders[1:1+nr_neighbours//2] + orders[-(nr_neighbours - nr_neighbours//2):]
        else:
            raise ValueError("Unsupported neighbourhood selection approach '%s'" % self._neighbour_selection)
