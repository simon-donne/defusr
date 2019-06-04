from itertools import chain
import torch
import numpy as np
from datasets.dataset_adapter import DatasetAdapter

from local_config import base_data_folder
import os

class DTUAdapter(DatasetAdapter):
    """Adapter for the DTU MVS dataset."""

    datapath = os.path.join(base_data_folder, 'dtu/')
    im_width = 1600
    im_height = 1200
    im_scale = 0.25
    nr_views = 49
    K = torch.from_numpy(np.array([
        [2892.843725329502400, 0, 824.425157504919530],
        [0, 2882.249450476587300, 605.187152104484080],
        [0, 0, 1]
    ]).astype(np.float32))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_default_splits()


    def _set_default_splits(self):

        self.split['test'] = [
            3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117
        ]
        self.split['val'] = [
            1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118,
        ]
        self.split['train'] = [
        ]
        self._complete_splits()

    @staticmethod
    def _all_elements():
        return chain(range(1, 78), range(82, 129))


    def _complete_splits(self):
        """Put any missing elements into the training set."""
        test_idx = 0
        val_idx = 0
        for i in self._all_elements():
            while test_idx < len(self.split['test']) and self.split['test'][test_idx] < i:
                test_idx += 1
            while val_idx < len(self.split['val']) and self.split['val'][val_idx] < i:
                val_idx += 1
            test_equal = test_idx < len(self.split['test']) and self.split['test'][test_idx] == i
            val_equal = val_idx < len(self.split['val']) and self.split['val'][val_idx] == i
            if not (test_equal or val_equal):
                self.split['train'].append(i)


    def get_dataset_name(self):
        return "DTU"


    def _get_element_folder(self, element):
        """Returns the subfolder for this element."""
        return "scan%d" % element


    def _get_image_scale_subfolder(self):
        """Returns the subfolder for the images, depending on the image scale."""
        if self.im_scale <= 0.25:
            return "Rectified_rescaled/0.25/"
        else:
            return "Rectified/"


    def get_image_path(self, element, view):
        image_path = "%s/%s/%s/rect_%03d_max.png" % (
            self.datapath,
            self._get_image_scale_subfolder(),
            self._get_element_folder(element),
            view+1
        )
        return image_path


    def _get_normal_map_scale_subfolder(self):
        """Returns the subfolder for the depth maps, depending on the image scale."""
        if self.im_scale <= 0.25:
            return "Normals/0.25/"
        else: 
            return "Normals/"


    def get_normal_map_path(self, element, view):
        normal_map_path = "%s/%s/%s/rect_%03d.npy" % (
            self.datapath,
            self._get_normal_map_scale_subfolder(),
            self._get_element_folder(element),
            view
        )
        return normal_map_path


    def _get_depth_map_scale_subfolder(self):
        """Returns the subfolder for the depth maps, depending on the image scale."""
        if self.im_scale <= 0.25:
            return "Depth/0.25/"
        else: 
            return "Depth/"


    def get_depth_map_path(self, element, view, gt=True):
        depth_map_path = "%s/%s%s/%s/rect_%03d_points.npy" % (
            self.datapath,
            "planed_" if gt else self.depth_map_prefix,
            self._get_depth_map_scale_subfolder(),
            self._get_element_folder(element),
            view
        )
        return depth_map_path


    def get_element_cameras(self, element):
        cameras = []
        scaler = np.array([[self.im_scale, 0, 0], [0, self.im_scale, 0], [0, 0, 1]])
        for i in range(self.nr_views):
            camera_filename = "%s/Calibration/cal18/pos_%03d.txt" % (
                self.datapath,
                i+1
            )
            camera_matrix = np.matmul(scaler, np.loadtxt(camera_filename))
            cameras.append(torch.from_numpy(camera_matrix.astype(np.float32)).unsqueeze(0))
        return torch.cat(cameras, 0)


    def get_element_worldtf(self, element):
        bb_path = "%s/ObsMask_NPY/BB%d_10.npy" % (
            self.datapath,
            element
        )
        bb = np.load(bb_path).astype(np.float32)
        extent = np.max(bb[1, :] - bb[0, :])
        bottom_left = (bb[0, :] + bb[1, :]) / 2 - extent / 2

        world_transform = np.array([
            [extent, 0, 0, bottom_left[0]],
            [0, extent, 0, bottom_left[1]],
            [0, 0, extent, bottom_left[2]],
            [0, 0, 0, 1],
        ])
        world_transform = torch.from_numpy(world_transform.astype(np.float32))
        return world_transform

    valid_centerviews = range(0, nr_views)

    def get_view_neighbours(self,cameras,center_view,nr_neighbours):
        if nr_neighbours == 0:
            return []
        clocs = []
        for i in range(cameras.shape[0]):
            invKR = torch.inverse(cameras[i][:3,:3])
            cloc = - torch.matmul(invKR,cameras[i][:3,3])
            clocs.append(cloc)
        cloc = clocs[center_view]
        distances = []
        for i in range(len(clocs)):
            distances.append(torch.norm(clocs[i] - cloc).item())
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
