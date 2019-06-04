
from datasets.DTU import DTUAdapter
import torch

from local_config import base_data_folder
import os

class UnrealDTUAdapter(DTUAdapter):
    """Adapter for a homebrew Unreal Engine version of the DTU MVS dataset."""

    datapath = os.path.join(base_data_folder, 'unrealDTU/')
    nr_views = 49

    def _set_default_splits(self):
        self.split['train'] = []
        self.split['test'] = [8,16,24,32,40,48,56,64]
        self.split['val'] = []
        self._complete_splits()

    @staticmethod
    def _all_elements():
        return range(1, 69)


    def get_dataset_name(self):
        return "uDTU"

    def _get_image_scale_subfolder(self):
        """Returns the subfolder for the images, depending on the image scale."""
        if self.im_scale <= 0.25:
            if self.im_scale <= 0.125:
                return "Rectified_rescaled/0.125/"
            else:
                return "Rectified_rescaled/0.25/"
        else:
            return "Rectified/"

    def _get_depth_map_scale_subfolder(self):
        """Returns the subfolder for the depth maps, depending on the image scale."""
        if self.im_scale <= 0.25:
            if self.im_scale <= 0.125:
                return "Depth/0.125/"
            else:
                return "Depth/0.25/"
        else: 
            return "Depth/"

    def get_depth_map_path(self, element, view, gt=True):
        depth_map_path = "%s/%s%s/%s/rect_%03d_points.npy" % (
            self.datapath,
            "" if gt else self.depth_map_prefix,
            self._get_depth_map_scale_subfolder(),
            self._get_element_folder(element),
            view
        )
        return depth_map_path

    def _get_normal_map_scale_subfolder(self):
        """Returns the subfolder for the normal maps, depending on the image scale."""
        if self.im_scale <= 0.25:
            if self.im_scale <= 0.125:
                return "Normals/0.125/"
            else:
                return "Normals/0.25/"
        else: 
            return "Normals/"

    def get_element_worldtf(self, element):
        world_transform = torch.eye(4, 4)
        for i in range(3):
            world_transform[i, i] = 70
        world_transform[0, 3] = -35
        world_transform[1, 3] = -35
        world_transform[2, 3] = -10
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
