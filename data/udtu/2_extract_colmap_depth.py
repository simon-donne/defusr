import numpy as np
import cv2
import sys


folder = sys.argv[1]
out_folder = sys.argv[3]

for idx in range(1,49+1):
    fn = folder + "rect_%03d_max.png.%s.bin" % (idx, sys.argv[2])
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
    np.save(out_folder+"rect_%03d_points.npy" % (idx-1), image)
    cv2.imwrite(out_folder+"rect_%03d_points.png" % (idx-1), image)

