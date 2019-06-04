import numpy as np
import cv2
import os
import sys

if len(sys.argv) < 2:
    raise UserWarning("Please pass the image source folder")
else:
    img_folder_src = sys.argv[1]

if len(sys.argv) >= 3:
    cam_folder_src = sys.argv[2]
else:
    raise UserWarning("Please pass the image source folder")

if len(sys.argv) >= 4:
    out_folder = sys.argv[3]
else:
    raise UserWarning("Please pass the workspace folder")

nr_views = 49
nr_neighbours = 10
image_scale = 0.25

# scale everything up by factor four: this is the easiest way for full-resolution results
upscale = 4

baseK = np.array([[800,0,800],[0,800,600],[0,0,1]])
K = baseK.copy()
K[0:2,:] = K[0:2,:] * image_scale * upscale

clocs = []

for view in range(0,nr_views):
    P = np.loadtxt(os.path.join(cam_folder_src,'pos_%03d.txt' % (view+1)))
    Rt = np.matmul(np.linalg.inv(baseK), P)
    clocs.append(np.matmul(-Rt[:3,:3].transpose(), Rt[:3,3:4]))
    Rt = np.concatenate((Rt, np.array([0,0,0,1]).reshape(1,4)), axis=0)
    with open(os.path.join(out_folder,"cams/%08d_cam.txt" % view), "wt") as cam_file:
        cam_file.write("extrinsic\n")
        np.savetxt(cam_file, Rt)
        cam_file.write("\nintrinsic\n")
        np.savetxt(cam_file, K)
        cam_file.write("\n10 4")
    img = cv2.imread(os.path.join(img_folder_src, 'rect_%03d_max.png' % (view+1)))
    H = img.shape[0]
    W = img.shape[1]
    img = cv2.resize(img, dsize=(upscale * W, upscale * H))
    cv2.imwrite(os.path.join(out_folder,"images/%08d.jpg" % view), img)

clocs = np.concatenate(clocs, axis=1)

with open(os.path.join(out_folder, "pair.txt"), "wt") as pair_file:
    pair_file.write("%d\n" % nr_views)
    for view in range(nr_views):
        dists = np.sum((clocs - clocs[:, view:view+1])**2, axis=0)
        order = np.argsort(dists)
        pair_file.write("%d\n" % view)
        pair_file.write("%d" % nr_neighbours)
        for n in range(nr_neighbours):
            pair_file.write(" %d 1.0" % order[n + 1])
        pair_file.write("\n")
