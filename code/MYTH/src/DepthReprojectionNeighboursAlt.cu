// we get as input a set of depth maps (B x N x 1 x H x W)
// and their cameras (B x N x 3 x 4), and generate the synthetic depth maps in the first views' image plane
// yes, the first image is simply left unchanged.

// input dimension: (B x N x 1 x H x W)
// output dimension: (B x N x 1 x H x W)

__global__ void DepthReprojectionNeighboursAlt_forward_depth_kernel(
                    float *input,
                    float *output,
                    float *cameras,
                    float *invKRs,
                    float *camlocs,
                    int batch_size,
                    int nrcams,
                    int input_height,
                    int input_width)
{
    int colstep = 1;
    int rowstep = colstep * input_width;
    int camstep = rowstep * input_height;
    int btcstep = camstep * nrcams;

    int clocs_camstep = 3;
    int clocs_btcstep = clocs_camstep * nrcams;
    int invKRs_camstep = 9;
    int invKRs_btcstep = invKRs_camstep * nrcams;
    
    int proj[2];
    float w[3];

    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < batch_size; b += blockDim.x * gridDim.x) {
        float *camera0 = cameras + b * nrcams * 12;
    for (int n = 0; n < nrcams; n++) {
    for (int in_row = blockIdx.y * blockDim.y + threadIdx.y; in_row < input_height; in_row += blockDim.y * gridDim.y) {
    for (int in_col = blockIdx.z * blockDim.z + threadIdx.z; in_col < input_width;  in_col += blockDim.z * gridDim.z) {
        float depth_n = input[b * btcstep + n * camstep + in_row * rowstep + in_col * colstep];
        if(n == 0) {
            // simply copy the first camera's depth map
            output[b * btcstep + n * camstep + in_row * rowstep + in_col * colstep] = depth_n;
        }
        else if(depth_n > 0) {
            // cast this point into space
            float *camloc = camlocs + b * clocs_btcstep + n * clocs_camstep;
            float *invKR = invKRs + b * invKRs_btcstep + n * invKRs_camstep;
            MYTH_unproject_point(camloc, invKR, in_col, in_row, depth_n, w);
            // project it onto the first camera again
            if(MYTH_project_point(camera0, w, proj, input_width, input_height)) {
                MYTH_atomicMinf(
                    output + b * btcstep + n * camstep + proj[1] * rowstep + proj[0] * colstep,
                    MYTH_get_point_depth(camera0, w)
                );
            }
        }
    }
    }
    }
    }
}

//the input dimension is (B x N x 1 x H x W)
//the output dimension is (B x N x 1 x H x W)
extern "C" void DepthReprojectionNeighboursAlt_updateOutput_gpu(
    THCudaTensor *input_depth,
    THCudaTensor *output_depth,
    THCudaTensor *cameras,
    THCudaTensor *invKRs,
    THCudaTensor *camlocs)
{
    int blkdim = 16;
    int batch_size = THCudaTensor_size(state, input_depth,0);
    int nrviews = THCudaTensor_size(state, input_depth,1);
    int input_height = THCudaTensor_size(state, input_depth,3);
    int input_width = THCudaTensor_size(state, input_depth,4);
    
    // we will use one thread for all depth hypotheses, to save some calculations regarding the directions and matrix inversions
    const dim3 block = dim3(1, blkdim, blkdim);
    const dim3 grid = dim3(1, 8, 8);
    float *input_depth_p = THCudaTensor_data(state, input_depth);
    float *output_depth_p = THCudaTensor_data(state, output_depth);
    float *cameras_p = THCudaTensor_data(state, cameras);
    float *invKRs_p = THCudaTensor_data(state, invKRs);
    float *camlocs_p = THCudaTensor_data(state, camlocs);

    cudaStream_t stream = THCState_getCurrentStream(state);
    DepthReprojectionNeighboursAlt_forward_depth_kernel<<<grid, block, 0, stream>>>(input_depth_p, output_depth_p, cameras_p, invKRs_p, camlocs_p, batch_size, nrviews, input_height, input_width);

    THCudaCheck(cudaGetLastError());
}
