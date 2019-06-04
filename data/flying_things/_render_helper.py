# -*- coding: utf-8 -*-

import os
import sys
import shutil
import random as ra
import tempfile
import datetime
import math
from functools import partial
from multiprocessing.dummy import Pool
from subprocess import call

import shlex
import numpy as np
import cv2
import OpenEXR
import Imath

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from ._global_variables_MVS import *

def depth_map_radial_to_planar(depth, K):
    """
    Transform depth maps from radial depth (euclidean distance to the camera center)
    to planar depth (orthogonal distance to the image plane + 1)
    """
    H = depth.shape[0]
    W = depth.shape[1]

    y = np.arange(0, H) + 0.5
    x = np.arange(0, W) + 0.5
    xv, yv = np.meshgrid(x, y)

    xs = xv.reshape(1,-1)
    ys = yv.reshape(1,-1)
    zs = np.ones(xs.shape)
    ds = depth.reshape(-1)

    locs_3d = np.concatenate((xs,ys,zs), axis=0)
    locs_3d = np.matmul(np.linalg.inv(K), locs_3d)
    nrms = np.linalg.norm(locs_3d, axis=0).reshape(1,-1)
    locs_3d = locs_3d / nrms
    locs_3d = locs_3d * ds

    zs = locs_3d[2]
    planar_depth = zs.reshape(depth.shape)
    return planar_depth

'''
@input:
    shape_synset e.g. '03001627' (each category has a synset)
@output:
    a list of (synset, md5, obj_filename, view_num) for each shape of that synset category
    where synset is the input synset, md5 is md5 of one shape in the synset category,
    obj_filename is the obj file of the shape, view_num is the number of images to render for that shape
'''
def load_one_category_shape_list(shape_synset):
    # return a list of (synset, md5, numofviews) tuples
    shape_md5_list = os.listdir(os.path.join(g_shapenet_root_folder, shape_synset))
    shape_list = []
    for shape_md5 in shape_md5_list:
        path = os.path.join(g_shapenet_root_folder, shape_synset, shape_md5, 'images/')
        if(os.path.isdir(path)):
            shape_list.append((shape_synset, shape_md5, os.path.join(g_shapenet_root_folder, shape_synset, shape_md5, 'models/model_normalized.obj')))
    # shape_list = [(shape_synset, shape_md5, os.path.join(g_shapenet_root_folder, shape_synset, shape_md5, 'model.obj')) for shape_md5 in shape_md5_list]
    return shape_list

'''
@input:
    shape_list and view_params as output of load_one_category_shape_list/views
@output:
    save rendered images to g_syn_images_folder/<synset>/<md5>/xxx.png
'''
def render_samples(g_syn_images_folder, all_shape_lists, backg_list, preset_seed):
    if not os.path.exists(os.path.join(g_tmp_data_folder, 'tmp/')):
        os.makedirs(os.path.join(g_tmp_data_folder, 'tmp/'))
    tmp_dirname_models = tempfile.mkdtemp(dir=os.path.join(g_tmp_data_folder, 'tmp/'), prefix='tmp_models_')
    if not os.path.exists(tmp_dirname_models):
        os.makedirs(tmp_dirname_models)

    samples = g_samples

    print('Generating rendering commands...')
    commands = []
    # render each sample
    for s in range(0, samples):
        # set a known seed
        if preset_seed == -1:
            seed = ra.randrange(4294967295)
        else:
            seed = preset_seed + s
        
        rng = ra.Random(seed)
        
        # use random number of objects
        obj_num = rng.randint(g_syn_objects_lowbound, g_syn_objects_highbound)
        
        folder_name = 'scene_%d_%d' % (obj_num, seed)
        
        if not os.path.isdir(os.path.join(g_syn_images_folder, folder_name)):
            # write model names to file
            tmp_models = tempfile.NamedTemporaryFile(dir=tmp_dirname_models, delete=False, mode="wt")
            for i in range(0, obj_num):
                cidx = rng.randint(0, g_categories - 1)
                while len(all_shape_lists[cidx]) == 0:
                    cidx = rng.randint(0, g_categories - 1)
                
                # draw random shape
                midx = rng.randint(0, len(all_shape_lists[cidx]) - 1)
                # print(all_shape_lists[cidx][midx])
                
                shape_synset, shape_md5, shape_file = all_shape_lists[cidx][midx]
                tmp_string = '%s\n' % (shape_file)
                tmp_models.write(tmp_string)
            tmp_models.close()
            
            # draw background randomly
            paramId = rng.randint(0, len(backg_list) - 1)
            backg_file = backg_list[paramId].strip('\n')
            
            if(g_DEBUG): 
                ### RUN COMMAND WITH TERMINAL OUTPUT
                command = '%s %s --background --python %s -- %s %s %d %s %d' % (g_blender_executable_path, g_blank_blend_file_path, os.path.join(BASE_DIR, 'render_model_views.py'), tmp_models.name, backg_file, g_sequence_length, os.path.join(g_syn_images_folder, folder_name), seed)
            else:     
                ### RUN COMMAND WITH REDIRECTED TERMINAL OUTPUT (-> NO OUTPUT)
                command = '%s %s --background --python %s -- %s %s %d %s %d > /dev/null 2>&1' % (g_blender_executable_path, g_blank_blend_file_path, os.path.join(BASE_DIR, 'render_model_views.py'), tmp_models.name, backg_file, g_sequence_length, os.path.join(g_syn_images_folder, folder_name), seed)

            commands.append(command)
        
            print('Created command %d' % (s))
        else:
            print('Folder ', os.path.join(g_syn_images_folder, folder_name), ' already exists, next try!')

    print('Rendering, it takes long time...')
    report_step = 100
    
    if not os.path.exists(os.path.join(g_syn_images_folder)):
        os.mkdir(os.path.join(g_syn_images_folder))
        
    pool = Pool(g_syn_rendering_thread_num)
    results = pool.imap(partial(call, shell=True), commands)
    for idx, return_code in enumerate(results):
        args = shlex.split(str.replace(commands[idx], ' > /dev/null 2>&1', ''))
        if idx % report_step == 0:
            print('[%s] Rendering command %d of %d' % (datetime.datetime.now().time(), idx, g_samples))
        if return_code != 0 or os.listdir(args[-2]) == []:
            print('Rendering command %d of %d (\"%s\") failed' % (idx, g_samples, commands[idx]))
        else:
            print(args)

            # extract depth from exr files!
            for s in range(1, g_sequence_length + 1):
                syn_image_file = 'scene_%03d_frame_%03d.exr' % (int(args[-1]), s)
                syn_depth_file = 'scene_%03d_frame_%03d_depth.exr' % (int(args[-1]), s)
                syn_png = str.replace(os.path.join(args[-2], syn_image_file), '.exr', '.png')
                depth_png_out = str.replace(os.path.join(args[-2], syn_depth_file), '.exr', '.png')
                print(syn_image_file)

                if(not os.path.isfile(os.path.join(args[-2], syn_depth_file))):
                    print('Cannot read %s' % syn_depth_file)
                    continue

                exrFile = OpenEXR.InputFile(os.path.join(args[-2], syn_depth_file))
                header = exrFile.header()
#                print(header)

                dw = header['dataWindow']
                pt = Imath.PixelType(Imath.PixelType.FLOAT)
                size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

                # get depth
                depth = np.fromstring(exrFile.channel('Image.V', pt), dtype=np.float32)
                depth.shape = (size[1], size[0])
                # transform from radial to planar
                K = np.loadtxt(syn_png + '.K')
                depth = depth_map_radial_to_planar(depth, K)
                depth[depth > 10] = 0
                
                d_img = 255 * depth / np.max(depth)
                d_img = d_img.astype('uint8')

                cv2.imwrite(depth_png_out, d_img)
                np.save(str.replace(depth_png_out, '.png', '.npy'), depth)

                # REMOVE THE .EXR FILE
                if g_remove_exrFile:
                    os.remove(os.path.join(args[-2], "Z%04d.png" % s))
                    os.remove(os.path.join(args[-2], syn_depth_file))
                    shutil.move(
                        os.path.join(args[-2], "Image%04d.png" % s),
                        syn_png
                    )
                    os.remove(os.path.join(args[-2], syn_image_file))

    shutil.rmtree(tmp_dirname_models)
