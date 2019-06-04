
from abc import ABC, abstractmethod
import numpy as np
import torch
import cv2

class DatasetAdapter(ABC):
    """
    Abstract class for dataset adapters.
    Contains the logic for actually loading the raw data from file,
    and for creating the relevant datastructures.
    """
    
    datapath = ''
    "The base folder for this dataset's files."

    im_width = 0
    "The original image width (some datasets support multiple; this is the highest)."

    im_height = 0
    "The original image height (some datasets support multiple; this is the highest)."

    im_scale = 0
    "The scale at which images are returned by the adapter."

    nr_views = 0
    "The number of views per scene."

    split = {'train': [], 'test': [], 'val': []}
    "The data split elements."

    ensure_multiple = 1
    """
    Ensure that the size of images are a multiple of this value.
    Will crop off the bottom right, so that camera matrices don't change.
    """

    depth_map_prefix = ""
    """
    Prepended to all Depth map folder names. Can be used to switch depth map inputs.
    """

    _neighbour_selection = "closest"
    """
    How to select neighbours for a given view.
    One of either "closest", "furthest", "mixed"
    """

    def __init__(self,**kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

    @abstractmethod
    def get_dataset_name(self):
        """
        Returns an informative names for the underlying data.
        """
        pass


    @abstractmethod
    def get_image_path(self, element, view):
        """
        Get the location of the image file for a given view of a given element (string).
        The views should be zero-indexed.

        Arguments:
        element -- the element index (integer)
        view -- the camera index (integer)
        """
        pass


    def get_single_image(self, element, view):
        """
        Get a single image from the specified dataset element (3 x H x W torch tensor).

        Arguments:
        element -- the element index (integer)
        view -- the camera index (integer)
        """
        image_path = self.get_image_path(element, view)
        img = cv2.imread(image_path)
        current_scale = img.shape[1] / (self.im_width * self.im_scale)
        target_multiple = int(self.ensure_multiple * current_scale)
        new_width = int(img.shape[1] / target_multiple) * target_multiple
        new_height = int(img.shape[0] / target_multiple) * target_multiple
        img = img[:new_height, :new_width, :]
        if self.im_width * self.im_scale < img.shape[1]:
            img = cv2.resize(
                img,
                (int(self.im_width * self.im_scale), int(self.im_height * self.im_scale)),
                interpolation=cv2.INTER_LANCZOS4
            )
        # change it to the C x H x W format required
        img = np.transpose(img, axes=(2, 0, 1))
        return torch.from_numpy(img.astype(np.float32))


    @abstractmethod
    def get_normal_map_path(self, element, view):
        """
        Get the location of the normal info file for a given view of a given element (string).
        The views should be zero-indexed.

        Arguments:
        element -- the element index (integer)
        view -- the camera index (integer)
        """
        pass
        

    def get_single_normal_map(self, element, view):
        """
        Get a single normal map from the specified dataset element (3 x H x W torch tensor).

        Arguments:
        element -- the element index (integer)
        view -- the camera index (integer)
        """
        normal_map_path = self.get_normal_map_path(element, view)
        normal_map = np.load(normal_map_path)

        current_scale = normal_map.shape[1] / (self.im_width * self.im_scale)
        target_multiple = int(self.ensure_multiple * current_scale)
        new_width = int(normal_map.shape[1] / target_multiple) * target_multiple
        new_height = int(normal_map.shape[0] / target_multiple) * target_multiple
        normal_map = normal_map[:new_height, :new_width, :]

        if self.im_width * self.im_scale < normal_map.shape[2]:
            normal_map = cv2.resize(
                normal_map,
                (int(self.im_width * self.im_scale), int(self.im_height * self.im_scale)),
                interpolation=cv2.INTER_NEAREST
            )
        normal_map = np.transpose(normal_map, axes=(2, 0, 1))

        return torch.from_numpy(normal_map.astype(np.float32))


    def get_element_images(self, element):
        """
        Get all images for the specified dataset element (N x 3 x H x W torch tensor).

        Arguments:
        element -- the element index (integer)
        """
        views = []
        for view in range(self.nr_views):
            views.append(self.get_single_image(element, view).unsqueeze(0))
        return torch.cat(views, 0)


    @abstractmethod
    def get_depth_map_path(self, element, view, gt=True):
        """
        Get the location of the depth map for a given view of a given element (string).
        The views should be zero-indexed.
        Should prepend depth_map_prefix to the depth subfolder.

        Arguments:
        element -- the element index (integer)
        view -- the camera index (integer)
        """
        pass


    def get_single_depth_map(self, element, view, gt=True):
        """
        Get a single depth map from the specified dataset element (1 x H x W torch tensor).

        Arguments:
        element -- the element index (integer)
        view -- the camera index (integer)
        """
        depth_map_path = self.get_depth_map_path(element, view, gt)
        depth_map = np.expand_dims(np.load(depth_map_path),2)

        current_scale = depth_map.shape[1] / (self.im_width * self.im_scale)
        target_multiple = int(self.ensure_multiple * current_scale)
        new_width = int(depth_map.shape[1] / target_multiple) * target_multiple
        new_height = int(depth_map.shape[0] / target_multiple) * target_multiple
        depth_map = depth_map[:new_height, :new_width, :]

        if self.im_width * self.im_scale < depth_map.shape[1]:
            depth_map = cv2.resize(
                depth_map,
                (int(self.im_width * self.im_scale), int(self.im_height * self.im_scale)),
                interpolation=cv2.INTER_NEAREST
            )[None]
        else:
            depth_map = np.transpose(depth_map, axes=(2, 0, 1))
        return torch.from_numpy(depth_map.astype(np.float32))


    def get_single_depth_map_and_trust(self, element, view, gt=True):
        """
        Get a single depth map from the specified dataset element (1 x H x W torch tensor).

        Arguments:
        element -- the element index (integer)
        view -- the camera index (integer)
        """
        depth_map_path = self.get_depth_map_path(element, view, gt)
        depth_map = np.expand_dims(np.load(depth_map_path),2)
        trust_path = depth_map_path.replace('.npy', '.trust.npy')
        trust = np.expand_dims(np.load(trust_path),2)

        current_scale = depth_map.shape[1] / (self.im_width * self.im_scale)
        target_multiple = int(self.ensure_multiple * current_scale)
        new_width = int(depth_map.shape[1] / target_multiple) * target_multiple
        new_height = int(depth_map.shape[0] / target_multiple) * target_multiple
        depth_map = depth_map[:new_height, :new_width, :]
        trust = trust[:new_height, :new_width, :]

        if self.im_width * self.im_scale < depth_map.shape[1]:
            depth_map = cv2.resize(
                depth_map,
                (int(self.im_width * self.im_scale), int(self.im_height * self.im_scale)),
                interpolation=cv2.INTER_NEAREST
            )[None]
            trust = cv2.resize(
                trust,
                (int(self.im_width * self.im_scale), int(self.im_height * self.im_scale)),
                interpolation=cv2.INTER_NEAREST
            )[None]
        else:
            depth_map = np.transpose(depth_map, axes=(2, 0, 1))
            trust = np.transpose(trust, axes=(2, 0, 1))

        return (torch.from_numpy(depth_map.astype(np.float32)), torch.from_numpy(trust.astype(np.float32)))


    def set_depth_map_provenance(self, prefix):
        """
        Sets the prefix for the Depth folder name.
        """
        self.depth_map_prefix = prefix


    def get_element_depth_maps(self, element, gt=True):
        """
        Get all depth maps for the specified dataset element (N x 1 x H x W torch tensor).

        Arguments:
        element -- the element index (integer)
        """
        views = []
        for view in range(self.nr_views):
            views.append(self.get_single_depth_map(element, view, gt=gt).unsqueeze(0))
        return torch.cat(views, 0)


    @abstractmethod
    def get_element_cameras(self, element):
        """
        Get all camera matrices for the specified dataset element (N x 3 x 4 torch tensor).

        Arguments:
        element -- the element index (integer)
        """
        pass


    @abstractmethod
    def get_element_worldtf(self, element):
        """
        Get the world transformation for the specified dataset element (4 x 4 torch tensor).

        Arguments:
        element -- the element index (integer)
        """
        pass


    def set_neighbour_selection(self, approach):
        """
        Set how neighbours are selected

        Arguments: 
        approach -- the new approach (string, one of "closest", "furthest", "mixed")
        """
        if approach not in ["closest", "furthest", "mixed"]:
            raise ValueError("Valid values for neighbour selection: 'closest', 'furthest', 'mixed'.")
        self._neighbour_selection = approach


    @abstractmethod
    def get_view_neighbours(self, cameras, center_view, nr_neighbours):
        """
        Get the closest neighbours for the given view.

        Arguments:
        cameras -- the camera matrices (N x 3 x 4 torch tensor)
        center_view -- the selected view (integer)
        nr_neighbours -- the number of neighbours (integer)
        """
        pass
