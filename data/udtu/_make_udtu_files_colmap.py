import numpy as np
import sys
import os

output_path = sys.argv[1]

data_folder = "/is/rg/avg/sdonne/data/unrealDTU/"
nr_cameras = 49
nr_neighbours = 10
image_scale = 0.25

# first: the intrinsic matrix
K = np.array([
    [800,0,800],
    [0,800,600],
    [0,0,1]
])
with open(os.path.join(output_path, "udtu_cameras.txt"), 'wt') as fh:
    fh.write("# Camera list with one line of data per camera:\n")
    fh.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
    fh.write("# Number of cameras: 1\n")
    fh.write("1 PINHOLE ")
    intrs = np.array([
        int(1600 * image_scale),
        int(1200 * image_scale),
        int(K[0,0] * image_scale),
        int(K[1,1] * image_scale),
        int(K[0,2] * image_scale),
        int(K[1,2] * image_scale)
    ])
    intrs.tofile(fh, sep=" ", format="%s")

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

# i'm pretty sure these are still wrong. If I transpose the rotation matrices, I get basically the same output ...

# second: the extrinsic ones. These are annoying, swear to god.
ts = []
with open(os.path.join(output_path, "udtu_images.txt"), 'wt') as fh:
    fh.write("# Image list with two lines of data per image:\n")
    fh.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
    fh.write("#   POINTS2D[] as (X, Y, POINT3D_ID) -- empty because these are precomputed matrices\n")
    fh.write("# Number of images: 49, mean observations per image: 0\n")
    for cam_idx in range(1,nr_cameras+1):
        P = np.loadtxt(data_folder + "Calibration/cal18/pos_%03d.txt" % cam_idx)
        Rt = np.matmul(np.linalg.inv(K),P)
        R = Rt[:,:3]
        t = Rt[:,3:]
        R[2:3,:] = np.cross(R[0:1,:], R[1:2,:])
        R[:,2] = np.cross(R[:,0], R[:,1])
        Q = rotation_to_quaternion(R)
        fh.write("%d " % (cam_idx-1))
        Q.tofile(fh, sep=" ", format="%s")
        fh.write(" ")
        t.tofile(fh, sep=" ", format="%s")
        fh.write(" 1 rect_%03d_max.png\n\n" % cam_idx)
        ts.append(np.matmul(-R.transpose(),t))

# third: just a completely empty points3D file
with open(os.path.join(output_path, "udtu_points3D.txt"),'wt') as fh:
    fh.write("# nothing to see here")

# finally, create the neighbours file
with open(os.path.join(output_path, "udtu_patch-match.cfg"),'wt') as fh:
    ts = np.array(ts).squeeze()
    for cam_idx in range(0,nr_cameras):
        t = ts[cam_idx:cam_idx+1,:] # 1,3
        diffs = np.sum((ts - t)**2,axis=1)
        order = np.argsort(diffs)
        neighbours = order[1:nr_neighbours+1]
        fh.write("rect_%03d_max.png\n" % (cam_idx+1))
        (neighbours+1).tofile(fh, sep=", ",format="rect_%03d_max.png")
        fh.write("\n")

