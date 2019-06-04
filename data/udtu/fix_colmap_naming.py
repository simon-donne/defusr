#TODO -- is this still necessary? this was fixed in the generating script itself, right?

import shutil
import glob
import os

from local_config import base_data_folder

base_folder = os.path.join(
    base_data_folder, 
    "unrealDTU/colmap/photometric/depth/Depth/0.25/"
)

for old_name in glob.glob(os.path.join(base_folder, "*/*_max.*")):
    new_name = old_name.replace("_max.", "_points.")
    print(old_name + " -> " + new_name)
    shutil.move(old_name, new_name)


# the problem is: all files are named scan###/rect_###_points.[png|npy] (1-indexed)
# when they should be scan###/rect_###_max.[png|npy] (0-indexed)

raise ValueError("Be really really sure you want to do this")

nr_views = 49

for scan_folder in glob.glob(os.path.join(base_folder, "*")):
    print(scan_folder)
    for i in range(0,nr_views):
        for extension in [".png", ".npy"]:
            old_name = os.path.join(scan_folder, "rect_%03d_points" % (i + 1)) + extension
            new_name = os.path.join(scan_folder, "rect_%03d_points" % (i)) + extension
            shutil.move(old_name, new_name)