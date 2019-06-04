// we get as input a set of depth maps (B x N x 1 x H x W) and their cameras (B x N x 3 x 4)
// as well as the result from DepthReprojectionNeighbours
// we want to fill in the unknown pixels with minimum depth values based on the other surfaces

// input dimension: (B x N x 1 x H x W)
// output dimension: (B x N x 1 x H x W)

__global__ void DepthReprojectionNonzeroCompleteBound_forward_kernel(
                    float *input,
                    float *output,
                    float *cameras,
                    float *invKRs,
                    float *camlocs,
                    int batch_size,
                    int nrcams,
                    int input_height,
                    int input_width,
                    float dmin,
                    float dmax,
                    float dstep)
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
        float *camloc0 = camlocs + b * clocs_btcstep;
        float *invKR0 = invKRs + b * invKRs_btcstep;
    for (int n = 0; n < nrcams; n++) {
    for (int out_row = blockIdx.y * blockDim.y + threadIdx.y; out_row < input_height; out_row += blockDim.y * gridDim.y) {
    for (int out_col = blockIdx.z * blockDim.z + threadIdx.z; out_col < input_width;  out_col += blockDim.z * gridDim.z) {
        // cast this point into space at increasingly large depths (from camera 0)
        // the first depth at which it is invisible in view n (i.e. lies behind its depth map)
        // that is the lowest permissible depth for this pixel according to that view
        // for very sharp depth edges, this *should* result in an interpolation of the depth map
        bool projected_in = false;
        float dhyp = dmin;
        for (; dhyp <= dmax; dhyp += dstep) {
            MYTH_unproject_point(camloc0, invKR0, out_col, out_row, dhyp, w);
            float *camera = cameras + (b * nrcams + n)* 12;
            // project it onto the first camera again
            if(MYTH_project_point(camera, w, proj, input_width, input_height)) {
                projected_in = true;
                float dhyp_depth_n = MYTH_get_point_depth(camera, w);
                float depth_n = input[b * btcstep + n * camstep + proj[1] * rowstep + proj[0] * colstep];
                if( (dhyp_depth_n > depth_n && depth_n > 0) || (depth_n <= 0)) {
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
                float *camera = cameras + (b * nrcams + n)* 12;
                // project it onto the first camera again
                if(MYTH_project_point(camera, w, proj, input_width, input_height)) {
                    projected_in = true;
                    float dhyp_depth_n = MYTH_get_point_depth(camera, w);
                    float depth_n = input[b * btcstep + n * camstep + proj[1] * rowstep + proj[0] * colstep];
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
        output[b * btcstep + n * camstep + out_row * rowstep + out_col * colstep] = dhyp;
    }
    }
    }
    }
}

//the input dimension is (B x N x 1 x H x W)
//the output dimension is (B x N x 1 x H x W)
extern "C" void DepthReprojectionNonzeroCompleteBound_updateOutput_gpu(
    THCudaTensor *input,
    THCudaTensor *output,
    THCudaTensor *cameras,
    THCudaTensor *invKRs,
    THCudaTensor *camlocs,
    float dmin,
    float dmax,
    float dstep)
{
    int blkdim = 16;
    int batch_size = THCudaTensor_size(state, input,0);
    int nrviews = THCudaTensor_size(state, input,1);
    int input_height = THCudaTensor_size(state, input,3);
    int input_width = THCudaTensor_size(state, input,4);
    
    // we will use one thread for all depth hypotheses, to save some calculations regarding the directions and matrix inversions
    const dim3 block = dim3(1, blkdim, blkdim);
    const dim3 grid = dim3(1,ceil(THCudaTensor_size(state, output, 3)*1.0f/blkdim), ceil(THCudaTensor_size(state, output, 4)*1.0f/blkdim));
    float *input_p = THCudaTensor_data(state, input);
    float *output_p = THCudaTensor_data(state, output);
    float *cameras_p = THCudaTensor_data(state, cameras);
    float *invKRs_p = THCudaTensor_data(state, invKRs);
    float *camlocs_p = THCudaTensor_data(state, camlocs);

    cudaStream_t stream = THCState_getCurrentStream(state);
    DepthReprojectionNonzeroCompleteBound_forward_kernel<<<grid, block, 0, stream>>>(input_p, output_p, cameras_p, invKRs_p, camlocs_p, batch_size, nrviews, input_height, input_width, dmin, dmax, dstep);

    THCudaCheck(cudaGetLastError());
}