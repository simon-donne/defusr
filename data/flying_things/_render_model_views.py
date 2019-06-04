# -*- coding: utf-8 -*-
'''
RENDER_MODEL_VIEWS.py
brief:
    render projections of a 3D model from viewpoints specified by an input parameter file
usage:
    blender blank.blend --background --python render_model_views.py -- <shape_obj_filename> <shape_category_synset> <shape_model_md5> <shape_view_param_file> <syn_img_output_folder>

inputs:
       <shape_obj_filename>: .obj file of the 3D shape model
       <shape_category_synset>: synset string like '03001627' (chairs)
       <shape_model_md5>: md5 (as an ID) of the 3D shape model
       <shape_view_params_file>: txt file - each line is '<azimith angle> <elevation angle> <in-plane rotation angle> <distance>'
       <syn_img_output_folder>: output folder path for rendered images of this model

author: hao su, charles r. qi, yangyan li
'''

import os
import bpy
import addon_utils
from mathutils import *
from math import *
import sys
import math
import random
import shutil
import numpy as np

# enable plugins
addon_utils.enable("io_import_images_as_planes")

# Load rendering light parameters
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from data.flying_things._global_variables_MVS import *
light_num_lowbound = g_syn_light_num_lowbound
light_num_highbound = g_syn_light_num_highbound
light_dist_lowbound = g_syn_light_dist_lowbound
light_dist_highbound = g_syn_light_dist_highbound

# some helping functions
def quaternionFromYawPitchRoll(yaw, pitch, roll):
    c1 = math.cos(yaw / 2.0)
    c2 = math.cos(pitch / 2.0)
    c3 = math.cos(roll / 2.0)
    s1 = math.sin(yaw / 2.0)
    s2 = math.sin(pitch / 2.0)
    s3 = math.sin(roll / 2.0)
    q1 = c1 * c2 * c3 + s1 * s2 * s3
    q2 = c1 * c2 * s3 - s1 * s2 * c3
    q3 = c1 * s2 * c3 + s1 * c2 * s3
    q4 = s1 * c2 * c3 - c1 * s2 * s3
    return (q1, q2, q3, q4)

def quaternionProduct(qx, qy):
    a = qx[0]
    b = qx[1]
    c = qx[2]
    d = qx[3]
    e = qy[0]
    f = qy[1]
    g = qy[2]
    h = qy[3]
    q1 = a * e - b * f - c * g - d * h
    q2 = a * f + b * e + c * h - d * g
    q3 = a * g - b * h + c * e + d * f
    q4 = a * h + b * g - c * f + d * e
    return (q1, q2, q3, q4)

def obj_centened_camera_pos(dist, azimuth_deg, elevation_deg):
    phi = float(elevation_deg) / 180 * math.pi
    theta = float(azimuth_deg) / 180 * math.pi
    x = (dist * math.cos(theta) * math.cos(phi))
    y = (dist * math.sin(theta) * math.cos(phi))
    z = (dist * math.sin(phi))
    return (x, y, z)

# cycles_material_text_node.py Copyright (C) 2012, Silvio Falcinelli
def cycles_material_text():
    mats = bpy.data.materials
    for cmat in mats:
        # print(cmat.name)
        cmat.use_nodes = True
        TreeNodes = cmat.node_tree
        links = TreeNodes.links

        shader = ''
        for n in TreeNodes.nodes:

            if n.type == 'TEX_IMAGE' or n.type == 'RGBTOBW':
                TreeNodes.nodes.remove(n)

            if n.type == 'OUTPUT_MATERIAL':
                shout = n

            if n.type == 'BACKGROUND':
                shader = n
            if n.type == 'BSDF_DIFFUSE':
                shader = n
            if n.type == 'BSDF_GLOSSY':
                shader = n
            if n.type == 'BSDF_GLASS':
                shader = n
            if n.type == 'BSDF_TRANSLUCENT':
                shader = n
            if n.type == 'BSDF_TRANSPARENT':
                shader = n
            if n.type == 'BSDF_VELVET':
                shader = n
            if n.type == 'EMISSION':
                shader = n
            if n.type == 'HOLDOUT':
                shader = n

        if cmat.raytrace_mirror.use and cmat.raytrace_mirror.reflect_factor > 0.001:
            print("MIRROR")
            if shader:
                if not shader.type == 'BSDF_GLOSSY':
                    print("MAKE MIRROR SHADER NODE")
                    TreeNodes.nodes.remove(shader)
                    shader = TreeNodes.nodes.new('ShaderNodeBsdfGlossy')  # RGB node
                    shader.location = 0, 450
                    # print(shader.glossy)
                    links.new(shader.outputs[0], shout.inputs[0])

        if not shader:
            shader = TreeNodes.nodes.new('ShaderNodeBsdfDiffuse')  # RGB node
            shader.location = 0, 450

            shout = TreeNodes.nodes.new('ShaderNodeOutputMaterial')
            shout.location = 200, 400
            links.new(shader.outputs[0], shout.inputs[0])



        if shader:
            textures = cmat.texture_slots
            for tex in textures:

                if tex:
                    if tex.texture.type == 'IMAGE':

                        img = tex.texture.image
                        # print(img.name)
                        shtext = TreeNodes.nodes.new(type='ShaderNodeTexImage')

                        shtext.location = -200, 400

                        shtext.image = img

                        if tex.use_map_color_diffuse:
                            links.new(shtext.outputs[0], shader.inputs[0])

                        if tex.use_map_normal:
                            t = TreeNodes.nodes.new('ShaderNodeRGBToBW')
                            t.location = -0, 300
                            links.new(t.outputs[0], shout.inputs[2])
                            links.new(shtext.outputs[0], t.inputs[0])

# Input parameters
shapes_file = sys.argv[-5]
back_file = sys.argv[-4]
frames = int(sys.argv[-3])
syn_images_folder = sys.argv[-2]
seed = int(sys.argv[-1])
if not os.path.exists(syn_images_folder):
    os.mkdir(syn_images_folder)
    
shapes = [[x for x in line.strip().split(' ')] for line in open(shapes_file).readlines()]

objects = len(shapes)

# set random seed
rng = random.Random(seed)
np.random.seed(seed)

scn = bpy.context.scene
obj = bpy.ops.object

if not os.path.exists(syn_images_folder):
    os.makedirs(syn_images_folder)

# load background image
bpy.ops.import_image.to_plane(files=[{'name':back_file}], directory='.')
bpy.context.selected_objects[0].name = "Background"
bck_material = bpy.data.materials[0]

# load objects
for s in range(0,objects):
    shape_file = shapes[s][0]
    bpy.ops.import_scene.obj(filepath=shape_file)

    # Combine meshes of imported model
    for ob in bpy.context.scene.objects:
        if ob.type == 'MESH' and ob.name[:5] != 'Model' and ob.name[:10] != 'Background':
            ob.select = True
            bpy.context.scene.objects.active = ob
        else:
            ob.select = False

    bpy.ops.object.join()
    bpy.context.selected_objects[0].name = 'Model_%d' % (s)
    s = s + 1

### SET RENDERING
scn.render.engine = 'CYCLES'
scn.render.alpha_mode = 'TRANSPARENT'
world = bpy.data.worlds['World']
world.use_nodes = True
world.light_settings.use_ambient_occlusion = False

# changing these values does affect the render.
bg = world.node_tree.nodes['Background']
bg.inputs[0].default_value[:3] = (1.0, 1.0, 1.0)
bg.inputs[1].default_value = 1.0

scene = bpy.data.scenes["Scene"]
scene.cycles.samples = g_render_samples
scene.cycles.min_bounces = 1
scene.cycles.max_bounces = 1
scene.cycles.use_multiple_importance_sampling = True
scene.render.tile_x = 16
scene.render.tile_y = 16
scene.cycles.film_transparent = True
scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
scene.render.image_settings.color_mode = 'RGBA'
scene.render.use_motion_blur = False
scene.render.layers["RenderLayer"].use_pass_vector = 1
scene.render.layers["RenderLayer"].use_pass_z = 1

scene.render.resolution_percentage = g_render_resolution_percentage

# Save resolution
x_res = bpy.context.scene.render.resolution_x
y_res = bpy.context.scene.render.resolution_y

bpy.data.objects['Lamp'].data.energy = 0

cam = bpy.data.cameras['Camera']
camObj = bpy.data.objects['Camera']
backObj = bpy.data.objects['Background']

backObj.parent = camObj
backObj.parent = None

# ANIMATION
scn.frame_start = 1
scn.frame_end = 1 + frames

# set background emission property
bck_material.use_nodes = True
bck_material.node_tree.nodes.remove(bck_material.node_tree.nodes.get('Diffuse BSDF'))
bck_material_output = bck_material.node_tree.nodes.get('Material Output')
bck_emission = bck_material.node_tree.nodes.new('ShaderNodeEmission')
bck_emission.inputs['Strength'].default_value = 2 * g_syn_light_environment_energy_highbound
bck_material.node_tree.links.new(bck_material_output.inputs[0], bck_emission.outputs[0])
bck_material.use_shadeless = True
bck_material.use_nodes = True

# set lights
obj.select_all(action='TOGGLE')
if 'Lamp' in list(bpy.data.objects.keys()):
    bpy.data.objects['Lamp'].select = True  # remove default light
obj.delete()

# clear default lights
obj.select_by_type(type='LAMP')
obj.delete(use_global=False)

# set environment lighting
scn.world.light_settings.use_environment_light = True
scn.world.light_settings.environment_energy = np.random.uniform(g_syn_light_environment_energy_lowbound, g_syn_light_environment_energy_highbound)
scn.world.light_settings.environment_color = 'PLAIN'

# set point lights
for i in range(rng.randint(light_num_lowbound, light_num_highbound)):
    light_azimuth_deg = np.random.uniform(g_syn_light_azimuth_degree_lowbound, g_syn_light_azimuth_degree_highbound)
    light_elevation_deg = np.random.uniform(g_syn_light_elevation_degree_lowbound, g_syn_light_elevation_degree_highbound)
    light_dist = np.random.uniform(light_dist_lowbound, light_dist_highbound)
    lx, ly, lz = obj_centened_camera_pos(light_dist, light_azimuth_deg, light_elevation_deg)
    obj.lamp_add(type='POINT', view_align=False, location=(lx, ly, lz))
    bpy.data.objects['Point'].data.energy = np.random.normal(g_syn_light_energy_mean, g_syn_light_energy_std)
    bpy.data.objects['Point'].data.use_nodes = True

# camera location
rho = np.random.uniform(g_syn_cam_dist_lowbound, g_syn_cam_dist_highbound)
camObj.location[0] = rho
camObj.location[1] = 0
camObj.location[2] = 0
camObj.rotation_mode = 'QUATERNION'
camObj.rotation_quaternion[0] = 0.5
camObj.rotation_quaternion[1] = 0.5
camObj.rotation_quaternion[2] = 0.5
camObj.rotation_quaternion[3] = 0.5

camObj.keyframe_insert(data_path="location", frame=1, index=-1)
camObj.keyframe_insert(data_path="rotation_quaternion", frame=1, index=-1)

# set background position and orientation
bck_dist = g_syn_bkg_dist
bck_rotation = Vector((0,0,0))

# set distance from camera and scale
backObj.parent = camObj
scn.update()            # necessary otherwise not working;
backObj.location = (0, 0, -bck_dist)
backObj.scale = ((bck_dist + rho) * g_syn_scale_background / 2, (bck_dist + rho) * g_syn_scale_background / 2,1)
q1 = backObj.rotation_quaternion
q2 = quaternionFromYawPitchRoll(bck_rotation[0], bck_rotation[1], bck_rotation[2])
q = quaternionProduct(q2, q1)
backObj.rotation_mode = 'QUATERNION'
backObj.rotation_quaternion[0] = q[0]
backObj.rotation_quaternion[1] = q[1]
backObj.rotation_quaternion[2] = q[2]
backObj.rotation_quaternion[3] = q[3]
scn.update()            # necessary otherwise not working
backObj.parent = None
scn.update()            # necessary otherwise not working
backObj.matrix_world = camObj.matrix_world * backObj.matrix_local

# clear actions
for a in bpy.data.actions: a.user_clear()

# turn off reflections
if(not g_specularity):
    for m in bpy.data.materials:
        m.specular_intensity = 0.0

# random location and orientation for each object
for o in range(0, objects):
    o_name = 'Model_%d' % (o)
    ob = bpy.data.objects[o_name]

    # odist = np.random.uniform(-1.0, 0.35 * rho)    # spawn objects between camera and background plane
    # oy = (rho - odist) * math.tan(cam.angle_x/2.0)*0.85        # get max width
    # oz = (rho - odist) * math.tan(cam.angle_y/2.0)*0.85        # get max height
    odist = np.random.uniform(0.1, 0.2 * rho)    # spawn objects between camera and background plane
    oy = (rho - odist) * math.tan(cam.angle_x/2.0)        # get max width
    oz = (rho - odist) * math.tan(cam.angle_y/2.0)        # get max height
    ob.location = Vector((odist, np.random.uniform(-oy, oy), np.random.uniform(-oz, oz)))
    q = quaternionFromYawPitchRoll(np.random.uniform(0, 2*pi), np.random.uniform(0, 2*pi), np.random.uniform(0, 2*pi))     # random orientation
    ob.rotation_mode = 'QUATERNION'
    ob.rotation_quaternion[0] = q[0]
    ob.rotation_quaternion[1] = q[1]
    ob.rotation_quaternion[2] = q[2]
    ob.rotation_quaternion[3] = q[3]

# camera position in each frame: randomize
for f in range(-1, frames):
    # camObj.location[0] = np.random.uniform(0.9*rho,1.1*rho)
    # loc1 = np.sign(np.random.uniform(-1,1))*np.random.uniform(0,0.2*rho)
    # camObj.location[1] = loc1
    # loc2 = np.sign(np.random.uniform(-1,1))*np.random.uniform(0,0.2*rho)
    # camObj.location[2] = loc2
    camObj.location[0] = rho
    loc1 = np.sign(np.random.uniform(-1,1))*np.random.uniform()
    camObj.location[1] = loc1
    loc2 = np.sign(np.random.uniform(-1,1))*np.random.uniform()
    camObj.location[2] = loc2

    # try to make the change in the direction of where we want to go
    q = quaternionFromYawPitchRoll(
        math.pi/2 + loc1 * math.pi/24,
        math.pi/16 * np.random.uniform(-1,1),
        math.pi/2 - loc2 * math.pi/24,
    )
    camObj.rotation_quaternion[0] = q[0]
    camObj.rotation_quaternion[1] = q[1]
    camObj.rotation_quaternion[2] = q[2]
    camObj.rotation_quaternion[3] = q[3]

    camObj.keyframe_insert(data_path="location", frame=2+f, index=-1)
    camObj.keyframe_insert(data_path="rotation_quaternion", frame=2+f, index=-1)

cycles_material_text()


def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam * location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv*R_world2bcam
    T_world2cv = R_bcam2cv*T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
         ))
    return RT

def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal), 
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal), 
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm


    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew,    u_0),
        (    0  , alpha_v, v_0),
        (    0  , 0,        1 )))
    return K


def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K,RT


bpy.context.scene.render.use_compositing = True
bpy.context.scene.use_nodes = True
scene_tree = bpy.context.scene.node_tree
renderlayers_node = scene_tree.nodes.new('CompositorNodeRLayers')
outputfile_node = scene_tree.nodes.new('CompositorNodeOutputFile')
outputfile_node.format.file_format = 'PNG'

depthFileOutputEXR = scene_tree.nodes.new(type="CompositorNodeOutputFile")
depthFileOutputEXR.label = 'Depth Output EXR'
depthFileOutputEXR.format.file_format = 'OPEN_EXR_MULTILAYER'
depthFileOutputEXR.format.color_depth = '32'
depthFileOutputEXR.format.compression = 0
try:
    scene_tree.links.new(renderlayers_node.outputs['Depth'], depthFileOutputEXR.inputs[0])
except KeyError:
    scene_tree.links.new(renderlayers_node.outputs['Z'], depthFileOutputEXR.inputs[0])

for s in range(scn.frame_start, scn.frame_end):
    scn.frame_set(s)
    syn_image_file = './scene_%03d_frame_%03d.exr' % (seed, s)
    syn_depth_file = './scene_%03d_frame_%03d_depth.exr' % (seed, s)

    outputfile_node.base_path = syn_images_folder+"/"
    outputfile_node.file_slots.new('Z')
    scene_tree.links.new(renderlayers_node.outputs['Image'], outputfile_node.inputs['Image'])
    try:
        scene_tree.links.new(renderlayers_node.outputs['Depth'], outputfile_node.inputs['Z'])
    except KeyError:
        scene_tree.links.new(renderlayers_node.outputs['Z'], outputfile_node.inputs['Z'])
    bpy.context.scene.render.filepath = os.path.join(syn_images_folder, syn_image_file)

    depthFileOutputEXR.base_path = os.path.join(syn_images_folder, "depth_")
    depthFileOutputEXR.file_slots[0].path = os.path.join(syn_images_folder, syn_depth_file)

    bpy.ops.render.render(write_still=True)

    shutil.move(
        os.path.join(syn_images_folder, 'depth_%04d.exr' % s),
        os.path.join(syn_images_folder, syn_depth_file),
    )


    K,RT = get_3x4_P_matrix_from_blender(camObj)
    P = np.matmul(K,RT)
    np.savetxt(os.path.join(syn_images_folder, 'scene_%03d_frame_%03d.png.K' % (seed, s)), K)
    np.savetxt(os.path.join(syn_images_folder, 'scene_%03d_frame_%03d.png.P' % (seed, s)), P)
