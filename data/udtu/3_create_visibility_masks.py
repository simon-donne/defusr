"""
Creating the visibility masks for all of the scenes.
"""

import pycuda
import pycuda.autoinit
import pycuda.driver as cuda
import torch
_ = torch.cuda.FloatTensor(8) # somehow required for correct pytorch-pycuda interfacing, assuring they use the same context
from pycuda.compiler import SourceModule

from datasets.unreal_DTU import UnrealDTUAdapter as DataAdapter
import numpy as np
from utils.file_system import ensure_dir

_surface_depth = 1
bb = np.array([
    [-60, -40, -5],
    [ 60,  40, 85],
])
voxel_size = 0.2

def compile_kernel():

    kernel_code = '''
        #include <stdio.h>

        __global__ void visibility_kernel(
            float *cameras,
            float *depth_maps,
            float *voxels,
            float *bb,
            int vgx, // voxels_gridsize
            int vgy, // voxels_gridsize
            int vgz, // voxels_gridsize
            float voxel_size,
            float surface_depth,
            int N,
            int H,
            int W
        ){
            int vx, vy, vz; // voxel indices
            float sx, sy, sz; // spatial coordinates

            // projection registers
            float px_n, py_n, pz_n; 
            int u, v, n;
            float d;

            for(vx = threadIdx.x + blockDim.x*blockIdx.x; vx < vgx; vx += blockDim.x*gridDim.x) {
            for(vy = threadIdx.y + blockDim.y*blockIdx.y; vy < vgy; vy += blockDim.y*gridDim.y) {
            for(vz = threadIdx.z + blockDim.z*blockIdx.z; vz < vgz; vz += blockDim.z*gridDim.z) {
                sx = bb[0] + (vx + 0.5f)*voxel_size;
                sy = bb[1] + (vy + 0.5f)*voxel_size;
                sz = bb[2] + (vz + 0.5f)*voxel_size;

                // project this spatial location on all the cameras; if it is visible there, it is marked visible
                // assume cheirality
                for(n = 0; n < N; n++) {
                    px_n = cameras[12*n + 0] * sx + cameras[12*n + 1] * sy + cameras[12*n +  2] * sz + cameras[12*n +  3];
                    py_n = cameras[12*n + 4] * sx + cameras[12*n + 5] * sy + cameras[12*n +  6] * sz + cameras[12*n +  7];
                    pz_n = cameras[12*n + 8] * sx + cameras[12*n + 9] * sy + cameras[12*n + 10] * sz + cameras[12*n + 11];

                    u = int(px_n / pz_n);
                    v = int(py_n / pz_n);
                    if(u > 0 && v > 0 && u < W && v < H) {
                        d = depth_maps[n * H * W + v * W + u];
                        if(pz_n  <= d + surface_depth) {
                            voxels[vx * vgy * vgz + vy * vgz + vz] = 1.0f;
                            break;
                        }
                    }
                }
            }
            }
            }
        }
    '''

    module = SourceModule(kernel_code)
    visibility_kernel = module.get_function('visibility_kernel')

    class TensorHolder(pycuda.driver.PointerHolderBase):
        def __init__(self, t):
            super().__init__()
            self.t = t
            self.gpudata = t.data_ptr()
        def get_pointer(self):
            return self.t.data_ptr()

    def visibility_gpu(cameras, depth_maps, voxels_bb, voxel_size, surface_depth):
        """
        Computes voxel visibilities over a given grid bb with given granularity.

        Arguments:
            cameras -- cuda torch float tensor (N x 3 x 4)
            depth_maps -- cuda torch float tensor (N x 1 x H x W)
            voxels_bb -- np.float32 bounding box of the voxel grid (2 x 3)
            voxel_size -- size of a given voxel, implies grid resolution (scalar)
        """

        N = cameras.shape[0]
        H = depth_maps.shape[2]
        W = depth_maps.shape[3]
        
        voxels_bb = voxels_bb.astype(np.float32)

        voxels_gridsize = (voxels_bb[1] - voxels_bb[0]) / voxel_size + 1

        voxels = depth_maps.new_zeros([int(e) for e in voxels_gridsize])

        # Call the kernel
        threadBlock = (8, 8, 8)
        blockGrid = (4, 4, 4)

        block = tuple([int(e) for e in threadBlock])
        grid = tuple([int(e) for e in blockGrid])


        voxels_bb = torch.Tensor(voxels_bb).float().cuda()

        visibility_kernel(
            TensorHolder(cameras), TensorHolder(depth_maps), TensorHolder(voxels),
            TensorHolder(voxels_bb),
            np.int32(voxels_gridsize[0]), np.int32(voxels_gridsize[1]), np.int32(voxels_gridsize[2]),
            np.float32(voxel_size), np.float32(surface_depth),
            np.int32(N), np.int32(H), np.int32(W),
            grid=grid, block=block
        )

        torch.cuda.synchronize()

        return voxels
    return visibility_gpu


def main():
    data_adapter = DataAdapter()
    data_adapter.im_scale = 1./4

    elements = data_adapter._all_elements()

    visibility_computation = compile_kernel()

    base_folder = data_adapter.datapath + "ObsMask_NPY/"
    ensure_dir(base_folder)

    for idx, element in enumerate(elements):
        print("Processing element %d/%d" % (idx + 1, len(elements)))
        # preload all depth maps
        cameras = data_adapter.get_element_cameras(element).cuda()
        depth_maps = data_adapter.get_element_depth_maps(element).cuda()

        visibility = visibility_computation(cameras, depth_maps, bb, voxel_size, _surface_depth)
        visibility = visibility.cpu().numpy()
        # compression factor ~250 -- worth it
        np.savez_compressed(base_folder + "ObsMask%d_10.npz" % element, visibility=visibility, bb=bb, voxel_size=voxel_size)

    print("Finished processing all elements")

if __name__ == "__main__":
    main()