import numpy as np
import cv2
import os
import sys

if len(sys.argv) < 2:
    raise UserWarning("Please pass the image source folder")
else:
    img_folder_src = sys.argv[1]

if len(sys.argv) >= 3:
    out_folder = sys.argv[2]
else:
    raise UserWarning("Please pass the workspace folder")

nr_views = 10
nr_neighbours = 4
image_scale = 0.25

# scale everything up by factor four: this is the easiest way for full-resolution results
upscale = 4

baseK = np.array([[800,0,800],[0,800,600],[0,0,1]])
K = baseK.copy()
K[0:2,:] = K[0:2,:] * image_scale * upscale

clocs = []

scene_seed = img_folder_src.split("_")[-1].rstrip('/')

for view in range(0,nr_views):
    image_name = "scene_%s_frame_%03d.png" % (scene_seed, view + 1)
    P = np.loadtxt(os.path.join(img_folder_src, image_name + ".P"))
    baseK = np.loadtxt(os.path.join(img_folder_src, image_name + ".K"))
    Rt = np.matmul(np.linalg.inv(baseK), P)
    K = baseK.copy()
    K[0:2,:] = K[0:2,:] * upscale
    clocs.append(np.matmul(-Rt[:3,:3].transpose(), Rt[:3,3:4]))
    Rt = np.concatenate((Rt, np.array([0,0,0,1]).reshape(1,4)), axis=0)
    with open(os.path.join(out_folder,"cams/%08d_cam.txt" % view), "wt") as cam_file:
        cam_file.write("extrinsic\n")
        np.savetxt(cam_file, Rt)
        cam_file.write("\nintrinsic\n")
        np.savetxt(cam_file, K)
        cam_file.write("\n1 0.08")
    img = cv2.imread(os.path.join(img_folder_src, image_name))
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
