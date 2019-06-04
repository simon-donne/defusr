
import random
import math
import torch
import numpy as np
from utils.cache import Cache
from utils.ddf_logging import Timer
from MYTH.camera_utils import extract_patch_cython
import experiment_handler

GPU = torch.device('cuda')

class DataLoader:
    """
    Class responsible for combining raw dataset data into the form consumed by the networks.
    It is also responsible for sending them to the GPU.
    Make sure the data functions are thread safe!

    Constructor arguments:
        Adapter -- the dataset adapter serving the raw data
        minibatch_size -- the number of elements per minibatch
        data_function -- data_function for network input/output. See attribute docstring.
        [split_limits] -- optional limit for the elements in the train, test and val sets.
        [noshuffle=False] -- if set, epoch elements are not shuffled
        [caching=True] -- whether or not to cache data function calls
        [gpu_caching=True] -- whether or not to cache data function calls in gpu memory
    """

    def __init__(self, adapter, **kwargs):
        self.adapter = adapter
        "The dataset adapter serving the raw data."

        self.queue = []
        "A queue for the elements still to be served this epoch."

        self.split_limits = {'train': None, 'test': None, 'val': None}
        "Optional limits for the number of elements in train, test, and val sets."

        self.noshuffle = False
        "If set, epoch elements are not shuffled."

        self.current_minibatch = None
        "The current minibatch index in the epoch."

        self.current_phase = None
        "The current phase (train, test or val)."

        self.minibatch_size = None
        "The amount of elements per minibatch."

        self.cache = Cache(
            enabled=kwargs.pop('caching', True),
            gpu=kwargs.pop('gpu_caching', True)
        )
        "The cache used by the data function calls. By default, caches everything in GPU memory."

        self.data_function = None
        """
        Function that serves the input and target data for a given minibatch element from a given adapter.
        The minibatch dimension should already be added - they are concatenated along the first dimension.
        
        This function should handle any desired caching itself, using the passed cache.
        Input: adapter, element [, cache]
        Output: (input, target) tuple
        Both input and target should be a tuple
        """

        self._logger = None
        "Logger to handle output."

        self.center_crop_size = None
        "Used by the patch-based data servers to crop the center view."

        self.refinement_experiment = None

        self.nr_neighbours = 4

        self.restricted_nr_views = 1
        "Used by some data loader functions"

        self.__dict__.update(kwargs)

        if self.refinement_experiment is not None:
            self.refinement_experiment = experiment_handler.ExperimentHandler.load_experiment_from_file(self.refinement_experiment)


    def get_nr_neighbours(self):
        "Silly function to be backwards compatible to previous versions that didn't have this field"
        try:
            return self.nr_neighbours
        except Exception:
            return 4

    def set_logger(self, logger):
        self._logger = logger


    def get_data_name(self):
        "Returns an informative name for the underlying data."
        return self.adapter.get_dataset_name()


    def initialize_phase(self, phase):
        """Initialize the epoch element list, shuffle it, and reset state."""
        self.queue = self.adapter.split[phase].copy()
        if self.split_limits[phase] is not None:
            self.queue = self.queue[:self.split_limits[phase]]
        if not self.noshuffle and phase == "train":
            random.shuffle(self.queue)
        self.current_minibatch = -1
        self.current_phase = phase


    def get_minibatch_count(self):
        """Returns the total number of minibatches this epoch."""
        return math.ceil(len(self.queue)/self.minibatch_size)


    def get_epoch_size(self):
        """Returns the total number of elements this epoch."""
        return len(self.queue)


    def get_minibatch_size(self):
        """Returns the number of elements this minibatch."""
        return min(self.minibatch_size, len(self.queue) - self.minibatch_size * self.current_minibatch)


    def __next__(self):
        """Returns the next minibatch in the current epoch. Generator."""
        self.current_minibatch += 1
        if self.current_minibatch >= self.get_minibatch_count():
            raise StopIteration()
        # at this point, we should make sure they are on the GPU
        # if they were before, this is a cheap operation
        (data_in, target) = self.get_minibatch_data(self.data_function)
        data_in = [x.to(GPU) for x in data_in]
        target = [x.to(GPU) for x in target]
        return data_in, target


    def __iter__(self):
        return self


    def get_minibatch_elements(self):
        """Returns the elements in the current minibatch."""
        elements = []
        for i in range(self.get_minibatch_size()):
            elements.append(self.queue[i + self.minibatch_size * self.current_minibatch])
        return elements


    def get_minibatch_data(self, data_function):
        """Load the requested data for all elements of the current minibatch."""
        minibatch_elements = self.get_minibatch_elements()
        minibatch_datas_in = []
        minibatch_datas_gt = []
        for element in minibatch_elements:
            data_args = (self, element)
            element_data = data_function(*data_args)
            minibatch_datas_in.append(element_data[0])
            minibatch_datas_gt.append(element_data[1])
            nr_in = len(element_data[0])
            nr_gt = len(element_data[1])
        
        minibatch_data_in = []
        minibatch_data_gt = []
        for idx_in in range(nr_in):
            subdata = [x[idx_in] for x in minibatch_datas_in]
            minibatch_data_in.append(torch.cat(subdata, dim=0))
        for idx_gt in range(nr_gt):
            subdata = [x[idx_gt] for x in minibatch_datas_gt]
            minibatch_data_gt.append(torch.cat(subdata, dim=0))
        
        return (minibatch_data_in, minibatch_data_gt)


    def data_function_cameras_worldtf(self, element):
        """
        Example data function for getting an element's cameras as input and world transform as output.
        Low-dimensional data for framework sanity checks.
        """
        return (
            (self.cache.get(self.adapter.get_element_cameras, (element,)).unsqueeze(0),),
            (self.cache.get(self.adapter.get_element_worldtf, (element,)).unsqueeze(0),)
        )


    def data_function_neighbours_and_depth(self, element):
        """
        Example data function serving the central view and its neighbours (+ cameras),
        and its depth map as the target
        """
        # decide on a central view
        cameras = self.cache.get(self.adapter.get_element_cameras, (element, ))
        neighbours = None
        while neighbours is None:
            center_view = random.randint(0, self.adapter.nr_views-1)
            neighbours = self.adapter.get_view_neighbours(cameras, center_view, self.get_nr_neighbours())

        # get the central depth map
        center_image = self.cache.get(self.adapter.get_single_image, (element, center_view))
        center_depth_map = self.cache.get(self.adapter.get_single_depth_map, (element, center_view, True))

        neighbour_images = [center_image]
        neighbour_cameras = [cameras[center_view]]
        for neighbour in neighbours:
            neighbour_images.append(
                self.cache.get(self.adapter.get_single_image, (element, neighbour))
            )
            neighbour_cameras.append(
                cameras[neighbour]
            )
        neighbour_images = [torch.unsqueeze(x, dim=0) for x in neighbour_images]
        neighbour_cameras = [torch.unsqueeze(x, dim=0) for x in neighbour_cameras]

        views = torch.cat(neighbour_images, dim=0)
        cameras = torch.cat(neighbour_cameras, dim=0)

        return ((views.unsqueeze(0), cameras.unsqueeze(0)), (center_depth_map.unsqueeze(0),))


    def data_function_color_depth_and_gt(self, element, center_view=None):
        """
        Example data function serving the central view and its estimated depth,
        and the gt depth as the target.
        """
        # decide on a central view
        if center_view is None:
            center_view = random.randint(0, self.adapter.nr_views-1)

        # get the central image, estimate and GT depth map
        center_image = self.cache.get(self.adapter.get_single_image, (element, center_view))
        center_estimate = self.cache.get(self.adapter.get_single_depth_map, (element, center_view, False))
        center_depth_map = self.cache.get(self.adapter.get_single_depth_map, (element, center_view, True))

        return ((center_image[None], center_estimate[None]), (center_depth_map[None],))

    def data_function_all_color_depth_and_gt(self, element):
        """
        Example data function serving the central view and its estimated depth, for all views,
        and the gt depth as the target.
        """
        # get the central image, estimate and GT depth map
        center_images = self.cache.get(self.adapter.get_element_images, (element, ))
        center_estimates = self.cache.get(self.adapter.get_element_depth_maps, (element, False)) 
        center_depth_maps = self.cache.get(self.adapter.get_element_depth_maps, (element, True)) 

        if self.restricted_nr_views != 0:
            center_images = center_images[:self.restricted_nr_views]
            center_estimates = center_estimates[:self.restricted_nr_views]
            center_depth_maps = center_depth_maps[:self.restricted_nr_views]

        return ((center_images, center_estimates), (center_depth_maps,))


    def data_function_patched_color_depth_and_gt(self, element, center_view=None):
        """
        Example data function serving the central view and its estimated depth,
        and the gt depth as the target.
        """
        # decide on a central view
        center_view = random.randint(0, self.adapter.nr_views-1)

        # get the central image, estimate and GT depth map
        center_image = self.cache.get(self.adapter.get_single_image, (element, center_view))
        center_estimate = self.cache.get(self.adapter.get_single_depth_map, (element, center_view, False))
        center_depth_map = self.cache.get(self.adapter.get_single_depth_map, (element, center_view, True))

        if self.center_crop_size is not None:
            Cc = self.center_crop_size
            x0 = np.random.randint(0, center_image.shape[2] - Cc)
            y0 = np.random.randint(0, center_image.shape[1] - Cc)
            center_image = center_image[:, y0:y0+Cc, x0:x0+Cc]
            center_estimate = center_estimate[:, y0:y0+Cc, x0:x0+Cc]
            center_depth_map = center_depth_map[:, y0:y0+Cc, x0:x0+Cc]

        return ((center_image[None], center_estimate[None]), (center_depth_map[None],))


    def data_function_color_depth_and_color_gt(self, element):
        """
        Example data function serving the central view and its estimated depth,
        and the gt depth as the target.
        """
        # decide on a central view
        center_view = random.randint(0, self.adapter.nr_views-1)

        # get the central image, estimate and GT depth map
        center_image = self.cache.get(self.adapter.get_single_image, (element, center_view))
        center_estimate = self.cache.get(self.adapter.get_single_depth_map, (element, center_view, False))
        center_depth_map = self.cache.get(self.adapter.get_single_depth_map, (element, center_view, True))

        return ((center_image[None], center_estimate[None]), (center_image[None], center_depth_map[None],))


    def data_function_neighbours_and_depth_and_normals(self, element):
        """
        Example data function serving the central view and its neighbours (+ cameras),
        and its depth map as the target, as well as its normal map
        """
        # decide on a central view
        cameras = self.cache.get(self.adapter.get_element_cameras, (element, ))
        neighbours = None
        while neighbours is None:
            center_view = random.randint(0, self.adapter.nr_views-1)
            neighbours = self.adapter.get_view_neighbours(cameras, center_view, self.get_nr_neighbours())

        # get the central depth map
        center_image = self.cache.get(self.adapter.get_single_image, (element, center_view))
        center_depth_map = self.cache.get(self.adapter.get_single_depth_map, (element, center_view, True))
        center_normal_map = self.cache.get(self.adapter.get_single_normal_map, (element, center_view))

        neighbour_images = [center_image]
        neighbour_cameras = [cameras[center_view]]
        for neighbour in neighbours:
            neighbour_images.append(
                self.cache.get(self.adapter.get_single_image, (element, neighbour))
            )
            neighbour_cameras.append(
                cameras[neighbour]
            )
        neighbour_images = [torch.unsqueeze(x, dim=0) for x in neighbour_images]
        neighbour_cameras = [torch.unsqueeze(x, dim=0) for x in neighbour_cameras]

        views = torch.cat(neighbour_images, dim=0)
        cameras = torch.cat(neighbour_cameras, dim=0)

        return ((views.unsqueeze(0), cameras.unsqueeze(0)), (center_depth_map.unsqueeze(0),center_normal_map.unsqueeze(0),))


    def data_function_depth_volumes_and_gt(self, element):
        """
        Example data function serving all of the depth map estimates (+ cameras),
        and a GT depth map as the target. Order of the viewset is random (camera matrices adjusted)
        """
        
        all_cameras = self.cache.get(self.adapter.get_element_cameras, (element, ))
        viewset = list(self.adapter.valid_centerviews)
        random.shuffle(viewset)
        estimates = []
        cameras = []
        
        for view in viewset:
            estimate = self.cache.get(self.adapter.get_single_depth_map, (element, view, False))
            estimates.append(torch.unsqueeze(estimate, dim=0))
            cameras.append(torch.unsqueeze(all_cameras[view], dim=0))

        gt_depth = self.cache.get(self.adapter.get_single_depth_map, (element, viewset[0], True))

        views = torch.cat(estimates, dim=0).unsqueeze(0)
        cameras = torch.cat(cameras, dim=0).unsqueeze(0)

        return ((views, cameras), (gt_depth.unsqueeze(0),))


    def data_function_refinement(self, element, center_view=None):
        """
        Example data function serving central color, neighbouring depth map estimates (+ cameras),
        and a GT depth map as the target. Order of the viewset is random (camera matrices adjusted).
        """
        
        all_cameras = self.cache.get(self.adapter.get_element_cameras, (element, ))
        if center_view is None:
            viewset = list(self.adapter.valid_centerviews)
            random.shuffle(viewset)
            center_view = viewset[0]
        neighbours = self.adapter.get_view_neighbours(all_cameras, center_view, self.get_nr_neighbours())
        
        center_color = self.cache.get(self.adapter.get_single_image, (element, center_view))[None]
        estimates = [self.cache.get(self.adapter.get_single_depth_map, (element, center_view, False))[None]]
        cameras = [all_cameras[center_view][None]]
        
        for view in neighbours:
            estimate = self.cache.get(self.adapter.get_single_depth_map, (element, view, False))
            estimates.append(estimate[None])
            cameras.append(all_cameras[view][None])

        gt_depth = self.cache.get(self.adapter.get_single_depth_map, (element, center_view, True))[None]

        center_color = center_color.unsqueeze(0)
        depths = torch.cat(estimates, dim=0).unsqueeze(0)
        cameras = torch.cat(cameras, dim=0).unsqueeze(0)

        return ((center_color, depths, cameras), (gt_depth,))


    def data_function_refinement_full(self, element, center_view=None):
        """
        Example data function serving color, neighbouring depth map estimates (+ cameras),
        and a GT depth map as the target. Order of the viewset is random (camera matrices adjusted).
        """
        
        all_cameras = self.cache.get(self.adapter.get_element_cameras, (element, ))
        if center_view is None:
            viewset = list(self.adapter.valid_centerviews)
            random.shuffle(viewset)
            center_view = viewset[0]
        neighbours = self.adapter.get_view_neighbours(all_cameras, center_view, self.get_nr_neighbours())
        
        colors = [self.cache.get(self.adapter.get_single_image, (element, center_view))[None]]
        estimates = [self.cache.get(self.adapter.get_single_depth_map, (element, center_view, False))[None]]
        cameras = [all_cameras[center_view][None]]
        
        gt_depth = self.cache.get(self.adapter.get_single_depth_map, (element, center_view, True))[None]

        for view in neighbours:
            color = self.cache.get(self.adapter.get_single_image, (element, view))
            colors.append(color[None])
            estimate = self.cache.get(self.adapter.get_single_depth_map, (element, view, False))
            estimates.append(estimate[None])
            cameras.append(all_cameras[view][None])

        colors = torch.cat(colors, dim=0).unsqueeze(0)
        depths = torch.cat(estimates, dim=0).unsqueeze(0)
        cameras = torch.cat(cameras, dim=0).unsqueeze(0)

        return ((colors, depths, cameras), (gt_depth,))


    def data_function_refinement_patched(self, element, center_view=None):
        """
        Example data function serving color, neighbouring depth map estimates (+ cameras),
        and a GT depth map as the target. Order of the viewset is random (camera matrices adjusted).
        """
        
        all_cameras = self.cache.get(self.adapter.get_element_cameras, (element, ))
        if center_view is None:
            viewset = list(self.adapter.valid_centerviews)
            random.shuffle(viewset)
            center_view = viewset[0]
        neighbours = self.adapter.get_view_neighbours(all_cameras, center_view, self.get_nr_neighbours())
        
        color_center = self.cache.get(self.adapter.get_single_image, (element, center_view))[None]
        estimate_center = self.cache.get(self.adapter.get_single_depth_map, (element, center_view, False))[None]
        camera_center = all_cameras[center_view][None]

        colors = []
        estimates = []
        cameras = []
        
        gt_center = self.cache.get(self.adapter.get_single_depth_map, (element, center_view, True))[None]

        if self.center_crop_size is not None:
            Cc = self.center_crop_size
            x0 = np.random.randint(0, color_center.shape[3] - Cc)
            y0 = np.random.randint(0, color_center.shape[2] - Cc)
            color_center = color_center[:, :, y0:y0+Cc, x0:x0+Cc]
            estimate_center = estimate_center[:, :, y0:y0+Cc, x0:x0+Cc]
            gt_center = gt_center[:, :, y0:y0+Cc, x0:x0+Cc]

            # also move the optical center for the camera
            opticalcenter_shifter = torch.Tensor(np.array([
                [1, 0, -x0],
                [0, 1, -y0],
                [0, 0, 1]
            ]))[None]
            opticalcenter_shifter = opticalcenter_shifter.to(camera_center.device)
            camera_center = torch.matmul(opticalcenter_shifter, camera_center)

        for view in neighbours:
            color = self.cache.get(self.adapter.get_single_image, (element, view))
            colors.append(color[None])
            estimate = self.cache.get(self.adapter.get_single_depth_map, (element, view, False))
            estimates.append(estimate[None])
            cameras.append(all_cameras[view][None])

        colors = torch.cat(colors, dim=0).unsqueeze(0)
        estimates = torch.cat(estimates, dim=0).unsqueeze(0)
        cameras = torch.cat(cameras, dim=0).unsqueeze(0)
        
        return ((color_center, colors, estimate_center, estimates, camera_center, cameras), (gt_center,))


    def data_function_successive_refinement_full(self, element, center_view=None):
        """
        Example data function serving color, neighbouring depth map estimates (+ cameras),
        and a GT depth map as the target. Order of the viewset is random (camera matrices adjusted).
        """
        
        all_cameras = self.cache.get(self.adapter.get_element_cameras, (element, ))
        if center_view is None:
            viewset = list(self.adapter.valid_centerviews)
            random.shuffle(viewset)
            center_view = viewset[0]
        neighbours = self.adapter.get_view_neighbours(all_cameras, center_view, self.get_nr_neighbours())
        
        colors = [self.cache.get(self.adapter.get_single_image, (element, center_view))[None]]
        estimate, trust = self.cache.get(self.adapter.get_single_depth_map_and_trust, (element, center_view, False))
        estimates = [estimate[None]]
        trusts = [trust[None]]
        cameras = [all_cameras[center_view][None]]
        
        gt_depth = self.cache.get(self.adapter.get_single_depth_map, (element, center_view, True))[None]

        for view in neighbours:
            color = self.cache.get(self.adapter.get_single_image, (element, view))
            colors.append(color[None])
            estimate, trust = self.cache.get(self.adapter.get_single_depth_map_and_trust, (element, view, False))
            estimates.append(estimate[None])
            trusts.append(trust[None])
            cameras.append(all_cameras[view][None])

        colors = torch.cat(colors, dim=0).unsqueeze(0)
        depths = torch.cat(estimates, dim=0).unsqueeze(0)
        trusts = torch.cat(trusts, dim=0).unsqueeze(0)
        cameras = torch.cat(cameras, dim=0).unsqueeze(0)

        return ((colors, depths, trusts, cameras), (gt_depth,))


    def data_function_successive_refinement_patched(self, element, center_view=None):
        """
        This performs pre-refinement of the depth maps
        """

        all_cameras = self.cache.get(self.adapter.get_element_cameras, (element, ))
        if center_view is None:
            viewset = list(self.adapter.valid_centerviews)
            random.shuffle(viewset)
            center_view = viewset[0]
        neighbours = self.adapter.get_view_neighbours(all_cameras, center_view, self.get_nr_neighbours())
        
        color_center = self.cache.get(self.adapter.get_single_image, (element, center_view))[None]
        estimate_center, trust_center = self.cache.get(self.adapter.get_single_depth_map_and_trust, (element, center_view, False))
        estimate_center = estimate_center[None]
        trust_center = trust_center[None]
        camera_center = all_cameras[center_view][None]
        
        gt_center = self.cache.get(self.adapter.get_single_depth_map, (element, center_view, True))[None]

        if self.center_crop_size is not None:
            Cc = self.center_crop_size
            x0 = np.random.randint(0, color_center.shape[3] - Cc)
            y0 = np.random.randint(0, color_center.shape[2] - Cc)
            color_center = color_center[:, :, y0:y0+Cc, x0:x0+Cc]
            estimate_center = estimate_center[:, :, y0:y0+Cc, x0:x0+Cc]
            trust_center = trust_center[:, :, y0:y0+Cc, x0:x0+Cc]
            gt_center = gt_center[:, :, y0:y0+Cc, x0:x0+Cc]

            # also move the optical center for the camera
            opticalcenter_shifter = torch.Tensor(np.array([
                [1, 0, -x0],
                [0, 1, -y0],
                [0, 0, 1]
            ]))[None]
            opticalcenter_shifter = opticalcenter_shifter.to(camera_center.device)
            camera_center = torch.matmul(opticalcenter_shifter, camera_center)

        colors = []
        estimates = []
        trusts = []
        cameras = []

        for view in neighbours:
            color = self.cache.get(self.adapter.get_single_image, (element, view))
            colors.append(color[None])
            estimate, trust = self.cache.get(self.adapter.get_single_depth_map_and_trust, (element, view, False))
            estimates.append(estimate[None])
            trusts.append(trust[None])
            cameras.append(all_cameras[view][None])

        colors = torch.cat(colors, dim=0).unsqueeze(0)
        depths = torch.cat(estimates, dim=0).unsqueeze(0)
        trusts = torch.cat(trusts, dim=0).unsqueeze(0)
        cameras = torch.cat(cameras, dim=0).unsqueeze(0)

        return ((color_center, colors, estimate_center, depths, trust_center, trusts, camera_center, cameras), (gt_center,))


    def data_function_in_memory_successive_refinement_patched(self, element, center_view=None):
        """
        This performs pre-refinement of the depth maps
        """
        
        pr_loader = self.refinement_experiment._data_loader
        pr_loader.center_crop_size = None
        pr_datafn = self.refinement_experiment.get('data_loader_options')['data_function']
        pr_network = self.refinement_experiment.network

        all_cameras = self.cache.get(self.adapter.get_element_cameras, (element, ))
        if center_view is None:
            viewset = list(self.adapter.valid_centerviews)
            random.shuffle(viewset)
            center_view = viewset[0]
        neighbours = self.adapter.get_view_neighbours(all_cameras, center_view, self.get_nr_neighbours())
        
        color_center = self.cache.get(self.adapter.get_single_image, (element, center_view))[None]
        ctr_input = [x.cuda() for x in pr_datafn(pr_loader, element, center_view=center_view)[0]]
        with torch.no_grad():
            estimate_center, trust_center = pr_network(*ctr_input)
        camera_center = all_cameras[center_view][None]

        colors = []
        estimates = []
        trusts = []
        cameras = []
        
        gt_center = self.cache.get(self.adapter.get_single_depth_map, (element, center_view, True))[None]

        if self.center_crop_size is not None:
            Cc = self.center_crop_size
            x0 = np.random.randint(0, color_center.shape[3] - Cc)
            y0 = np.random.randint(0, color_center.shape[2] - Cc)
            color_center = color_center[:, :, y0:y0+Cc, x0:x0+Cc]
            estimate_center = estimate_center[:, :, y0:y0+Cc, x0:x0+Cc]
            trust_center = trust_center[:, :, y0:y0+Cc, x0:x0+Cc]
            gt_center = gt_center[:, :, y0:y0+Cc, x0:x0+Cc]

            # also move the optical center for the camera
            opticalcenter_shifter = torch.Tensor(np.array([
                [1, 0, -x0],
                [0, 1, -y0],
                [0, 0, 1]
            ]))[None]
            opticalcenter_shifter = opticalcenter_shifter.to(camera_center.device)
            camera_center = torch.matmul(opticalcenter_shifter, camera_center)

        for view in neighbours:
            color = self.cache.get(self.adapter.get_single_image, (element, view))
            colors.append(color[None])
            nbr_input = [x.cuda() for x in pr_datafn(pr_loader, element, center_view=view)[0]]
            with torch.no_grad():
                estimate, trust = pr_network(*nbr_input)
            estimates.append(estimate)
            trusts.append(trust)
            cameras.append(all_cameras[view][None])

        colors = torch.cat(colors, dim=0).unsqueeze(0)
        estimates = torch.cat(estimates, dim=0).unsqueeze(0)
        trusts = torch.cat(trusts, dim=0).unsqueeze(0)
        cameras = torch.cat(cameras, dim=0).unsqueeze(0)
        
        return ((color_center, colors, estimate_center, estimates, trust_center, trusts, camera_center, cameras), (gt_center,))
