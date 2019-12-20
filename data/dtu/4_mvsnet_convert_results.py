import numpy as np
import cv2
import re
import sys
import os

def load_pfm(file):
    with open(file, 'rb') as file:
        color = None
        width = None
        height = None
        scale = None
        data_type = None
        header = file.readline().decode('ascii').rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('ascii'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')
        scale = float(file.readline().decode('ascii').rstrip())
        if scale < 0:
            data_type = '<f'
        else:
            data_type = '>f'
        data_string = file.read()
        data = np.fromstring(data_string, data_type)
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = cv2.flip(data, 0)
        return data

target_size = (400, 300)

if len(sys.argv) < 2:
    raise UserWarning("Pass the workpath")
else:
    workpath = sys.argv[1]

def main():
    nr_views = 49
    for view in range(nr_views):
        depth = load_pfm(os.path.join(workpath,"depths_mvsnet/%08d.pfm"%view))
        trust = load_pfm(os.path.join(workpath,"depths_mvsnet/%08d_prob.pfm"%view))
        base_name = os.path.join(workpath,"depth/rect_%03d_points" % (view + 1))
        depth = cv2.resize(depth, dsize=target_size, interpolation=cv2.INTER_NEAREST)
        trust = cv2.resize(trust, dsize=target_size, interpolation=cv2.INTER_NEAREST)
        np.save(base_name + ".npy", depth)
        cv2.imwrite(base_name + ".png", depth / 4)
        np.save(base_name + ".trust.npy", trust)
        cv2.imwrite(base_name + ".trust.png", trust * 255.)

if __name__ == "__main__":
    main()
