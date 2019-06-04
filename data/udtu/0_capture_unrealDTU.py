"""
Note: this file should probably be run with python2 because StringIO
"""

import numpy as np
from unrealcv import client
from matplotlib import pyplot as plt
import time
import io
import sys
from scipy.misc import toimage
import os
import math
from tqdm import tqdm

base_folder = '/scratch/sdonne/data/unrealDTU/'
base_width = 1600
base_height = 1200
image_scale = 0.25
scale_folder = "Rectified/"
sub_scale_folder = ""
if image_scale != 1:
    scale_folder = "Rectified_rescaled/%s/"%(image_scale,)
    sub_scale_folder = "%s/"%(image_scale,)

rot0 = np.array((0.,-20.,-90.))
loc0 = np.array((0., 80., 50.))
# we know the intrinsic matrix: a perfect one!
baseK = np.array([[800,0,800],[0,800,600],[0,0,1]])
K = baseK.copy()
K[0:2,:] = K[0:2,:] * image_scale

def generate_camera_positions():
    # the distance to the look-at point (the origin)
    radius = np.linalg.norm(loc0)

    locations = []
    rotations = []

    ystep = 6.
    zstep = 6.
    ysteps = 6
    zsteps = [5,6,8,9,10,11]
    for oy in range(ysteps):
        dangle_y = (ysteps-1-oy) * ystep
        cloc_x = 0.
        cloc_y = radius * np.cos(np.arctan2(loc0[2],loc0[1]) + dangle_y/180.*np.pi)
        loc_z = radius * np.sin(np.arctan2(loc0[2],loc0[1]) + dangle_y/180.*np.pi)
        rot_x = rot0[0]
        rot_y = rot0[1] - dangle_y

        for oz in range(zsteps[oy]):
            order = 1. - 2.*np.mod(oy,2)
            dangle_z = order * (oz - (zsteps[oy] - 1) / 2) * zstep
            rot_z = rot0[2] + dangle_z
            
            loc_x = np.sin(-dangle_z/180.*np.pi)*cloc_y
            loc_y = np.cos(-dangle_z/180.*np.pi)*cloc_y
            
            locations.append((loc_x,loc_y,loc_z))
            rotations.append((rot_x,rot_y,rot_z))
    return locations, rotations


def position_to_transform(loc,rot):
    roll = math.radians(rot[0])
    pitch = math.radians(rot[1])
    yaw = math.radians(rot[2])
    c, s = np.cos(yaw), np.sin(yaw)
    RZ = np.matrix([[c,-s,0],[s,c,0],[0,0,1]])
    c, s = np.cos(pitch), np.sin(pitch)
    RY = np.matrix([[c,0,s],[0,1,0],[-s,0,c]])
    c, s = np.cos(roll), np.sin(roll)
    RX = np.matrix([[1,0,0],[0,c,-s],[0,s,c]])
    R = np.dot(RX,np.dot(RY,RZ))
    
    # to transform from the unreal engine camera reference system (x into the scene)
    # to the camera reference system we expect (z into the scenes)
    UE_transform = np.array([[0,-1,0],[0,0,-1],[1,0,0]])
    R = np.dot(UE_transform,R)
    
    camloc = np.array([loc[0],-loc[1],loc[2]]).reshape((3,1))
    T = -np.dot(R,camloc)

    RT = np.concatenate((R,T), axis=1)
    
    return RT
    
    
def invert_transform(RT):
    R = np.transpose(RT[0:3,0:3])
    T = - np.dot(R, RT[0:3,3:4])
    RTinv = np.concatenate((R, T), axis=1)
    return RTinv


def depth_to_point_cloud(plane_depth,intrinsic,color_image):
    H = plane_depth.shape[0]
    W = plane_depth.shape[1]
    j_c = intrinsic[0,2]
    i_c = intrinsic[1,2]
    Xs = np.dot(np.ones((H,1)),np.arange(0,W).reshape((1,W)))
    Ys = np.dot(np.arange(0,H).reshape((H,1)),np.ones((1,W)))
    valids = np.multiply(plane_depth > 0, plane_depth < 999)

    zs = plane_depth[valids]
    xs = np.multiply(zs,(Xs[valids] - j_c) / intrinsic[(0,0)])
    ys = np.multiply(zs,(Ys[valids] - i_c) / intrinsic[(0,0)])

    rs = color_image[...,0][valids]
    gs = color_image[...,1][valids]
    bs = color_image[...,2][valids]

    points = np.zeros(len(xs), dtype={'names': ('x', 'y', 'z', 'r', 'g', 'b'),
                              'formats': ('f4', 'f4', 'f4', 'u1', 'u1', 'u1')})

    points['x'] = xs
    points['y'] = ys
    points['z'] = zs
    
    points['r'] = rs
    points['g'] = gs
    points['b'] = bs
    return points


def apply_transform(points,RT):
    xin = np.expand_dims(points['x'],axis=0)
    yin = np.expand_dims(points['y'],axis=0)
    zin = np.expand_dims(points['z'],axis=0)
    win = np.ones(xin.shape)
    coords = np.concatenate((xin,yin,zin,win),axis=0)

    tcoords = np.dot(RT,coords)
    new_points = np.copy(points)
    new_points['x'] = tcoords[0]
    new_points['y'] = tcoords[1]
    new_points['z'] = tcoords[2]

    return new_points


def point_dists_to_plane_depth(point_dists,intrinsic):
    H = point_dists.shape[0]
    W = point_dists.shape[1]
    j_c = intrinsic[0,2]
    i_c = intrinsic[1,2]
    Xs = np.dot(np.ones((H,1)),np.arange(0,W).reshape((1,W)))
    Ys = np.dot(np.arange(0,H).reshape((H,1)),np.ones((1,W)))
    center_distance = np.power(Xs - j_c, 2) + np.power(Ys - i_c, 2)
    plane_depth = np.divide(point_dists,np.sqrt(1+center_distance / intrinsic[(0,0)]**2))
    return plane_depth


def set_resolution():
    client.request('vset /unrealcv/resolution %d %d' % (base_width * image_scale, base_height * image_scale))


def set_all_invisible(nr_objects):
    for i in range(1,nr_objects+1):
        client.request('vset /object/Object_%03d/visible 0' % (i,))


def set_visible_object(i):
    client.request('vset /object/Object_%03d/visible 1' % i)
    # we assume that the only possibly visible object is i - 1, if it exists
    if i > 1:
        client.request('vset /object/Object_%03d/visible 0' % (i-1,))
    # we need some time for UnrealEngine to load the correct 
    time.sleep(1.0)


def set_camera_position(loc, rot):
    client.request('vset /camera/0/location %f %f %f' % (loc[0],loc[1],loc[2]))
    client.request('vset /camera/0/rotation %f %f %f' % (rot[1],rot[2],rot[0]))
    time.sleep(0.1)


def save_frame(scan_folder, index, loc, rot):
    # color image
    color = np.load(io.BytesIO(client.request('vget /camera/0/lit npy')))
    image_path = base_folder+scale_folder+scan_folder+"/rect_%03d_max.png" % (index+1)
    toimage(color,cmin=0.0,cmax=255.0).save(image_path)
        
    # depth image/npy
    dists = np.load(io.BytesIO(client.request('vget /camera/0/depth npy')))
    #note: there is a scaling of 100 between the world units and the depth units (cm vs m)
    dists = dists * 100
    depth = point_dists_to_plane_depth(dists, K)
    np.save(base_folder+"Depth/"+sub_scale_folder+scan_folder+"/rect_%03d_points.npy" % index, depth)
#    np.savez_compressed(base_folder+"Depth/"+sub_scale_folder+scan_folder+"/rect_%03d_points.npz" % index, depth=depth)

    toimage(depth * 2, cmin = 0.0, cmax= 255.0).save(base_folder+"Depth/"+sub_scale_folder+scan_folder+"/rect_%03d_points.png" % index)
    
    # normal image/npy
    normals = (np.load(io.BytesIO(client.request('vget /camera/0/normal npy'))) - 128.)/128.
    # these normals are in world space. We want camera-space normals!
    RT = position_to_transform(loc, rot)
    normals = np.squeeze(np.matmul(np.asarray(RT[:3,:3]), np.expand_dims(normals,3)))
    np.save(base_folder+"Normals/"+sub_scale_folder+scan_folder+"/rect_%03d.npy" % index, normals)
#    np.savez_compressed(base_folder+"Normals/"+sub_scale_folder+scan_folder+"/rect_%03d.npz" % index, normals=normals)
    toimage(normals * 128. + 128., cmin = 0.0, cmax= 255.0).save(base_folder+"Normals/"+sub_scale_folder+scan_folder+"/rect_%03d.png" % index)
    
    # point cloud
    points = depth_to_point_cloud(depth,K,color)
    new_points = apply_transform(points,invert_transform(RT))
    
    #the extents of the table, idgaf about the background
    valids = np.abs(new_points['y']) < 150
    valids = np.multiply(np.abs(new_points['x']) <  100, valids)
    valids = np.multiply(np.abs(new_points['z']) <  100, valids)
    
    new_points = new_points[valids]
    
    return new_points


def write_ply(filename, points):
    f = open(filename, 'w')
    f.write("ply\n")
    f.write("format binary_little_endian 1.0\n")
    f.write("element vertex %d\n"%(points.shape[0],))
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("property uchar red\n")
    f.write("property uchar green\n")
    f.write("property uchar blue\n")
    f.write("end_header\n")
    points.tofile(f)
    f.close()
    print("")
    
    
def capture_camera_positions(scan_path,locations, rotations):
    ensure_dir(base_folder+"Depth/"+sub_scale_folder+"%s/" % scan_path)
    ensure_dir(base_folder+"Normals/"+sub_scale_folder+"%s/" % scan_path)
    ensure_dir(base_folder+scale_folder+"%s/" % scan_path)
    cloud = []
    for i in tqdm(range(0,len(locations)),desc=scan_path):
        set_camera_position(locations[i],rotations[i])
        points = save_frame(scan_path,i,locations[i],rotations[i])
        cloud.append(points)
    
    cloud = np.concatenate(cloud, axis=0)
    
    write_ply(base_folder+"Points/gt/%s/%s_total.ply" % (sub_scale_folder,scan_path), cloud)
    

def ensure_dir(newdir):
    if os.path.isdir(newdir):
        pass
    else:
        head, tail = os.path.split(newdir)
        if head and not os.path.isdir(head):
            ensure_dir(head)
        if tail:
            os.mkdir(newdir)


def main():
    total_nr_objects = 68
    max_nr_objects = 68
    locations, rotations = generate_camera_positions()
    
    ensure_dir(base_folder+"Points/gt/%s/"%(sub_scale_folder))
    ensure_dir(base_folder+"Calibration/cal18/")
    for c in range(len(locations)):
        RT = position_to_transform(locations[c],rotations[c])
        P = np.matmul(baseK,RT)
        output_file = base_folder+"Calibration/cal18/pos_%03d.txt" % (c+1,)
        f = open(output_file, 'w')
        f.write("%f %f %f %f\n" % (P[0,0],P[0,1],P[0,2],P[0,3]))
        f.write("%f %f %f %f\n" % (P[1,0],P[1,1],P[1,2],P[1,3]))
        f.write("%f %f %f %f\n" % (P[2,0],P[2,1],P[2,2],P[2,3]))
        f.close()
            
    try:
        if not client.connect():
            print("Could not connect to client! Exiting.")
            return
        set_all_invisible(total_nr_objects)
        set_resolution()
        for i in tqdm(range(1,max_nr_objects+1),desc="Scanning"):
            set_visible_object(i)
            scan_path = 'scan%d' % (i,)
            capture_camera_positions(scan_path,locations,rotations)
    finally:
        client.disconnect()
        

if __name__ == "__main__":
    main()
