import numpy as np
import sys
import os

data_folder = sys.argv[1]
scene_seed = data_folder.split('_')[-1][:-1]
nr_cameras = 10
nr_neighbours = 9
image_scale = 0.5

# first: the intrinsic matrix
K = np.array([[2100*image_scale,0,960*image_scale],[0,2100*image_scale,540*image_scale],[0,0,1]])
with open(os.path.join(data_folder,"ft_cameras.txt"), 'wt') as fh:
    fh.write("# Camera list with one line of data per camera:\n")
    fh.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
    fh.write("# Number of cameras: 1\n")
    fh.write("1 PINHOLE ")
    intrs = np.array([
        int(1920 * image_scale),
        int(1080 * image_scale),
        int(2100 * image_scale),
        int(2100 * image_scale),
        int(960 * image_scale),
        int(540 * image_scale)
    ])
    intrs.tofile(fh, sep=" ", format="%s")

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def rotation_to_quaternion(R):
    x42 = 1 + R[0,0] - R[1,1] - R[2,2]
    if x42 > 1e-3:
        q = np.array([
            R[2,1] - R[1,2],
            x42,
            R[1,0] + R[0,1],
            R[2,0] + R[0,2],
        ])
    else:
        q = np.array([
            R[0,2] - R[2,0],
            R[0,1] + R[1,0],
            1 - R[0,0] + R[1,1] - R[2,2],
            R[2,1] + R[1,2],
        ])
    q = q / np.linalg.norm(q)
    return q

def quaternion_to_rotation(q):
    q = q / np.linalg.norm(q)
    R = np.array([
        [1 - 2*(q[2]**2 + q[3]**2), 2*(q[1]*q[2] - q[3]*q[0]), 2*(q[1]*q[3] + q[0]*q[2])],
        [2*(q[1]*q[2] + q[3]*q[0]), 1 - 2*(q[1]**2 + q[3]**2), 2*(q[2]*q[3] - q[0]*q[1])],
        [2*(q[1]*q[3] - q[2]*q[0]), 2*(q[2]*q[3] + q[1]*q[0]), 1 - 2*(q[1]**2 + q[2]**2)],
    ]);
    return R

# second: the extrinsic ones. These are annoying, swear to god.
ts = []
with open(os.path.join(data_folder,"ft_images.txt"), 'wt') as fh:
    fh.write("# Image list with two lines of data per image:\n")
    fh.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
    fh.write("#   POINTS2D[] as (X, Y, POINT3D_ID) -- empty because these are precomputed matrices\n")
    fh.write("# Number of images: 10, mean observations per image: 0\n")
    for cam_idx in range(1,nr_cameras+1):
        P = np.loadtxt(data_folder + "scene_%s_frame_%03d.png.P" % (scene_seed, cam_idx))
        Rt = np.matmul(np.linalg.inv(K),P)
        R = Rt[:,:3]
        t = Rt[:,3:]
        R[2:3,:] = np.cross(R[0:1,:], R[1:2,:])
        R[:,2] = np.cross(R[:,0], R[:,1])
        Q = rotmat2qvec(R)
        fh.write("%d " % (cam_idx-1))
        Q.tofile(fh, sep=" ", format="%s")
        fh.write(" ")
        t.tofile(fh, sep=" ", format="%s")
        fh.write(" 1 scene_%s_frame_%03d.png\n\n" % (scene_seed, cam_idx))
        ts.append(np.matmul(-R.transpose(),t))

# third: just a completely empty points3D file
with open(os.path.join(data_folder,"ft_points3D.txt"),'wt') as fh:
    fh.write("# nothing to see here")

# finally, create the neighbours file
with open(os.path.join(data_folder,"ft_patch-match.cfg"),'wt') as fh:
    ts = np.array(ts).squeeze()
    for cam_idx in range(0,nr_cameras):
        t = ts[cam_idx:cam_idx+1,:] # 1,3
        diffs = np.sum((ts - t)**2,axis=1)
        order = np.argsort(diffs)
        neighbours = order[1:nr_neighbours+1]
        fh.write("scene_%s_frame_%03d.png\n" % (scene_seed, cam_idx+1))
        format = "scene_%s_frame_%%03d.png" % scene_seed
        (neighbours+1).tofile(fh, sep=", ",format=format)
        fh.write("\n")

