__global__ void PatchedReprojectionCompleteBound_forward_kernel(
    float *depth_center,
    float *depths_neighbours,
    float *bounds_neighbours,
    float *cameras_neighbours,
    float *invKR_center,
    float *camloc_center,
    int B,
    int N,
    int H_center,
    int W_center,
    int H_neighbours,
    int W_neighbours,
    float dmin,
    float dmax,
    float dstep)
{
    int proj[2];
    float w[3];

    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < B; b += blockDim.x * gridDim.x) {
        float *camloc0 = camloc_center + b * 3;
        float *invKR0 = invKR_center + b * 9;
    for (int n = 0; n < N; n++) {
    for (int out_row = blockIdx.y * blockDim.y + threadIdx.y; out_row < H_center; out_row += blockDim.y * gridDim.y) {
    for (int out_col = blockIdx.z * blockDim.z + threadIdx.z; out_col < W_center;  out_col += blockDim.z * gridDim.z) {
        // cast this point into space at increasingly large depths (from camera 0)
        // the first depth at which it is invisible in view n (i.e. lies behind its depth map)
        // that is the lowest permissible depth for this pixel according to that view
        bool projected_in = false;
        float dhyp = dmin;
        for (; dhyp <= dmax; dhyp += dstep) {
            MYTH_unproject_point(camloc0, invKR0, out_col, out_row, dhyp, w);
            float *camera_n = cameras_neighbours + (b * N + n)* 12;
            if(MYTH_project_point(camera_n, w, proj, W_neighbours, H_neighbours)) {
                projected_in = true;
                float dhyp_depth_n = MYTH_get_point_depth(camera_n, w);
                float depth_n = depths_neighbours[((b * N + n) * H_neighbours + proj[1]) * W_neighbours + proj[0]];
                if(dhyp_depth_n > depth_n && depth_n > 0) {
                    break;
                }
            }
            else if (projected_in) {
                // just give up -- no value here is acceptable
                dhyp = dmax;
                break;
            }
        }
        if(dhyp < dmax) {
            // refine the estimate
            float ndhyp = dhyp;
            for (; ndhyp >= dhyp - dstep; ndhyp -= dstep/10) {
                MYTH_unproject_point(camloc0, invKR0, out_col, out_row, ndhyp, w);
                float *camera_n = cameras_neighbours + (b * N + n)* 12;
                // project it onto the first camera again
                if(MYTH_project_point(camera_n, w, proj, W_neighbours, H_neighbours)) {
                    projected_in = true;
                    float dhyp_depth_n = MYTH_get_point_depth(camera_n, w);
                    float depth_n = depths_neighbours[((b * N + n) * H_neighbours + proj[1]) * W_neighbours + proj[0]];
                    if(dhyp_depth_n < depth_n) {
                        break;
                    }
                }
                else {
                    break;
                }
            }
            dhyp = ndhyp;
        }
        else {
            dhyp = 0.0f;
        }
        bounds_neighbours[((b * N + n) * H_center + out_row) * W_center + out_col] = dhyp;
    }
    }
    }
    }
}

//the input dimension is (B x N x 1 x H x W)
//the output dimension is (B x N x 1 x H x W)
extern "C" void PatchedReprojectionCompleteBound_updateOutput_gpu(
    THCudaTensor *depth_center,
    THCudaTensor *depths_neighbours,
    THCudaTensor *bounds_neighbours,
    THCudaTensor *cameras_neighbours,
    THCudaTensor *invKR_center,
    THCudaTensor *camloc_center,
    float dmin,
    float dmax,
    float dstep)
{
    int blkdim = 16;
    int B = THCudaTensor_size(state, depths_neighbours, 0);
    int N = THCudaTensor_size(state, depths_neighbours, 1);
    int H_neighbours = THCudaTensor_size(state, depths_neighbours, 3);
    int W_neighbours = THCudaTensor_size(state, depths_neighbours, 4);
    int H_center = THCudaTensor_size(state, depth_center, 2);
    int W_center = THCudaTensor_size(state, depth_center, 3);

    // we will use one thread for all depth hypotheses, to save some calculations regarding the directions and matrix inversions
    const dim3 block = dim3(1, blkdim, blkdim);
    const dim3 grid = dim3(1, 8, 8);
    float *depth_center_p = THCudaTensor_data(state, depth_center);
    float *depths_neighbours_p = THCudaTensor_data(state, depths_neighbours);
    float *bounds_neighbours_p = THCudaTensor_data(state, bounds_neighbours);
    float *cameras_neighbours_p = THCudaTensor_data(state, cameras_neighbours);
    float *invKR_center_p = THCudaTensor_data(state, invKR_center);
    float *camloc_center_p = THCudaTensor_data(state, camloc_center);

    cudaStream_t stream = THCState_getCurrentStream(state);
    PatchedReprojectionCompleteBound_forward_kernel<<<grid, block, 0, stream>>>(
        depth_center_p, depths_neighbours_p, bounds_neighbours_p, cameras_neighbours_p, invKR_center_p, camloc_center_p,
        B, N, H_center, W_center, H_neighbours, W_neighbours,
        dmin, dmax, dstep
    );

    THCudaCheck(cudaGetLastError());
}