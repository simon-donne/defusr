import numpy as np
import cv2
import sys


folder = sys.argv[1]
scene_path = sys.argv[2]
scene_seed = scene_path.split('_')[-1][:-1]
phase = sys.argv[3]
out_folder = sys.argv[4]

for idx in range(1,10+1):
    fn_depth = folder + "depth_maps/scene_%s_frame_%03d.png.%s.bin" % (scene_seed, idx, phase)
    with open(fn_depth, 'rb') as fh:
        width, height, channels = np.genfromtxt(fh, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fh.seek(0)
        num_delimiter = 0
        byte = fh.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fh.read(1)
        array = np.fromfile(fh, np.float32)
        array = array.reshape((width, height, channels), order="F")

    image = np.transpose(array, (1, 0, 2)).squeeze()
    np.save(out_folder+"depth/scene_%s_frame_%03d_depth.npy" % (scene_seed, idx), image)
    cv2.imwrite(out_folder+"depth/scene_%s_frame_%03d_depth.png" % (scene_seed, idx), image*30)

    # problem: COLMAP normals are in the camera coordinate system
    # we want them in the world coordinate system
    camera_P = np.loadtxt(scene_path + "scene_%s_frame_%03d.png.P" % (scene_seed, idx))
    camera_K = np.loadtxt(scene_path + "scene_%s_frame_%03d.png.K" % (scene_seed, idx))
    camera_R = np.matmul(np.linalg.inv(camera_K), camera_P[:,:3])
    fn_normals = folder + "normal_maps/scene_%s_frame_%03d.png.%s.bin" % (scene_seed, idx, phase)
    with open(fn_normals, 'rb') as fh:
        width, height, channels = np.genfromtxt(fh, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fh.seek(0)
        num_delimiter = 0
        byte = fh.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fh.read(1)
        array = np.fromfile(fh, np.float32)
        array = array.reshape((width, height, channels), order="F")

    image = np.transpose(array, (1, 0, 2)).squeeze()
    image = np.matmul(image.reshape(-1, channels), camera_R).reshape(height, width, channels)

    np.save(out_folder+"normals/scene_%s_frame_%03d_normals.npy" % (scene_seed, idx), image)
    cv2.imwrite(out_folder+"normals/scene_%s_frame_%03d_normals.png" % (scene_seed, idx), (image+1)*128)

