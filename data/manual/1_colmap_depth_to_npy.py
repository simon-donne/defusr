import numpy as np
import cv2
import sys
import os
import glob

folder = sys.argv[1]
style = sys.argv[2]
out_folder = os.path.join(folder, style)
os.makedirs(out_folder, exist_ok=True)

for fn in glob.glob(os.path.join(folder, "*.jpg.%s.bin" % style)):
    with open(fn, 'rb') as fh:
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
    fn = fn.lstrip(folder)
    np.save(os.path.join(out_folder, fn + ".npy"), image)
    image = image / 50 * 255.0 * 2
    cv2.imwrite(os.path.join(out_folder, fn + ".png"), image)

