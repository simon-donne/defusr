# -*- coding: utf-8 -*-
'''
RENDER_ALL_SHAPES
'''

import os
import sys
import random as ra
import socket

from data.flying_things._global_variables_MVS import g_categories, g_background_list, g_shape_synsets, g_shape_names
from data.flying_things import _render_helper as render_helper

if __name__ == '__main__':
    seed = -1
    g_syn_images_folder = '/scratch/sdonne/data/flying_things_MVS/'
    if len(sys.argv) > 1:
        g_syn_images_folder = sys.argv[1]
    if len(sys.argv) > 2:
        seed = int(sys.argv[2])

    if not os.path.exists(g_syn_images_folder):
        os.mkdir(g_syn_images_folder)

    # load background images and transformations
    backg_list = g_background_list

    # load models and view parameters
    all_shape_lists = [None] * g_categories
    filtered = 0
    for idx in range(g_categories):
        synset = g_shape_synsets[idx]
        # print('%d: %s, %s\n' % (idx, synset, g_shape_names[idx]))
        all_shape_lists[idx] = render_helper.load_one_category_shape_list(synset)
        filtered += (len(all_shape_lists[idx]))

    # render
    print('After texture filtering %d objects left!' % filtered)
    render_helper.render_samples(g_syn_images_folder, all_shape_lists, backg_list, seed)
