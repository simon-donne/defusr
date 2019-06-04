# -*- coding: utf-8 -*-

import os
import socket

g_DEBUG = False                # if FALSE: redirects the terminal output of all rendering threads to /dev/null

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
g_blender_executable_path = "/is/rg/avg/sdonne/installs/blender-2.79b/blender"
g_shapenet_root_folder = "/is/rg/avg/datasets/ShapeNetCore.v2/"
g_blank_blend_file_path = "/is/sg/sdonne/Desktop/research/differentiable_depth_fusion/code/data/flying_things/_blank.blend"
g_tmp_data_folder = "/tmp/sdonne/"
g_backgrounds_file = "/is/rg/avg/sdonne/data/flying_backgrounds/list.txt"
g_background_list = open(g_backgrounds_file).readlines()

# ------------------------------------------------------------
# RENDER FOR CNN PIPELINE
# ------------------------------------------------------------
# g_shape_synset_name_pairs = [('02691156', 'aeroplane'),
#                             ('02834778', 'bicycle'),
#                             ('02858304', 'boat'),
#                             ('02876657', 'bottle'),
#                             ('02924116', 'bus'),
#                             ('02958343', 'car'),
#                             ('03001627', 'chair'),
#                             ('04379243', 'diningtable'),
#                             ('03790512', 'motorbike'),
#                             ('04256520', 'sofa'),
#                             ('04468005', 'train'),
#                             ('03211117', 'tvmonitor')]
g_shape_synset_name_pairs = [
    ('02691156', ''),
    ('02747177', ''),
    ('02773838', ''),
    ('02801938', ''),
    ('02808440', ''),
    ('02818832', ''),
    ('02828884', ''),
    ('02843684', ''),
    ('02871439', ''),
    ('02876657', ''),
    ('02880940', ''),
    ('02924116', ''),
    ('02933112', ''),
    ('02942699', ''),
    ('02946921', ''),
    ('02954340', ''),
    ('02958343', ''),
    ('02992529', ''),
    ('03001627', ''),
    ('03046257', ''),
    ('03085013', ''),
    ('03207941', ''),
    ('03211117', ''),
    ('03261776', ''),
    ('03325088', ''),
    ('03337140', ''),
    ('03467517', ''),
    ('03513137', ''),
    ('03593526', ''),
    ('03624134', ''),
    ('03636649', ''),
    ('03642806', ''),
    ('03691459', ''),
    ('03710193', ''),
    ('03759954', ''),
    ('03761084', ''),
    ('03790512', ''),
    ('03797390', ''),
    ('03928116', ''),
    ('03938244', ''),
    ('03948459', ''),
    ('03991062', ''),
    ('04004475', ''),
    ('04074963', ''),
    ('04090263', ''),
    ('04099429', ''),
    ('04225987', ''),
    ('04256520', ''),
    ('04330267', ''),
    ('04379243', ''),
    ('04401088', ''),
    ('04460130', ''),
    ('04468005', ''),
    ('04530566', ''),
    ('04554684', ''),
]

g_shape_synsets = [x[0] for x in g_shape_synset_name_pairs]
g_shape_names = [x[1] for x in g_shape_synset_name_pairs]

g_syn_rendering_thread_num = 4

# Rendering is computational demanding. you may want to consider using multiple servers.
# g_hostname_synset_idx_map = {'<server1-hostname>': [0,1],
#                             '<server2-hostname>': [2,3,4],
#                             '<server3-hostname>': [5,6,7],
#                             '<server4-hostname>':[8,9],
#                             '<server5-hostname>':[10,11]}
g_hostname_synset_idx_map = {socket.gethostname(): range(12)}

# render model
g_samples = 1  # number of examples
g_sequence_length = 10  # sequence length of one example
g_categories = 55  # number of object categories (max 55)
g_render_resolution_percentage = 50  # precentage of full resolution (1920x1080) 
g_render_samples = 50  # render samples (higher reduces noise but also much slower)
g_remove_exrFile = True  # delete the exrFile after extracting images, flow and depth
g_specularity = False    # turn on or off reflections

# camera configuration
g_syn_cam_dist_lowbound = 3  # minimum camera distance from origin
g_syn_cam_dist_highbound = 4  # maximum camera distance from origin

g_syn_scale_background = 2.0  # scaling factor for background image to fit after projection

# number of objects
g_syn_objects_lowbound = 10 # minimum number of objects in the scene
g_syn_objects_highbound = 20  # maximum number of objects in the scene

# object orientation and motion
g_syn_obj_initial_rotation_lowbound = [-1, -1, -1]  # min initial objects rotation
g_syn_obj_initial_rotation_highbound = [1, 1, 1]  # max initial objects rotation
g_syn_obj_rotation_lowbound = [-0.0, -0.0, -0.0]  # min change of objects rotation
g_syn_obj_rotation_highbound = [0.0, 0.0, 0.0]  # max change of objects rotation
g_syn_obj_initial_translation_lowbound = [-0.1, -0.1, -0.1]  # min initial of objects translation
g_syn_obj_initial_translation_highbound = [0.1, 0.1, 0.1]  # max initial of objects translation
g_syn_obj_translation_lowbound = [-0, -0, -0]  # min change of objects translation
g_syn_obj_translation_highbound = [0, 0, 0]  # max change of objects translation

# background orientation and motion
g_syn_bkg_dist = g_syn_cam_dist_highbound + 1
g_syn_bkg_orientation_lowbound = [-15, -15, -15]  # min initial background orientation
g_syn_bkg_orientation_highbound = [15, 15, 15]  # max initial background orientation
g_syn_bkg_initial_rotation_lowbound = [-3, -3, -3]  # min initial background rotation
g_syn_bkg_initial_rotation_highbound = [3, 3, 3]  # max initial background rotation
g_syn_bkg_rotation_lowbound = [-0, -0, -0]  # min change of background rotation
g_syn_bkg_rotation_highbound = [0, 0, 0]  # max change of background rotation
g_syn_bkg_initial_translation_lowbound = [-2, -2, -2]  # min initial of background translation
g_syn_bkg_initial_translation_highbound = [2, 2, 2]  # max initial of background translation
g_syn_bkg_translation_lowbound = [-0, -0, -0]  # min change of background translation
g_syn_bkg_translation_highbound = [0, 0, 0]  # max change of background translation

# render model lights
g_syn_light_num_lowbound = 5  # min number of lights
g_syn_light_num_highbound = 8  # max number of lights
g_syn_light_dist_lowbound = 5  # min distance of lights from origin
g_syn_light_dist_highbound = 7  # min distance of lights from origin
g_syn_light_azimuth_degree_lowbound = -45  # min azimuth of lights
g_syn_light_azimuth_degree_highbound = 45  # max azimuth of lights
g_syn_light_elevation_degree_lowbound = -45  # min elevation of lights
g_syn_light_elevation_degree_highbound = 45  # min elevation of lights
g_syn_light_energy_mean = 0.25  # mean of light energy (brightness)
g_syn_light_energy_std = 0.2  # mean of light energy (brightness)
g_syn_light_environment_energy_lowbound = 0.25  # mean of environment energy (brightness)
g_syn_light_environment_energy_highbound = 0.25  # mean of environment energy (brightness)
