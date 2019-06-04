// we get as input a set of depth maps (B x N x 1 x H x W)
// and their cameras (B x N x 3 x 4), and generate the synthetic depth maps in the first views' image plane
// yes, the first image is simply left unchanged.
// we also pass the angle between both camera's viewing directions for each reprojected point, through the cos

// input dimension: (B x N x 1 x H x W)
// output dimension: (B x N x 1 x H x W)

__global__ void DepthColorAngleReprojectionNeighbours_forward_depth_kernel(
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

__global__ void DepthColorAngleReprojectionNeighbours_forward_colorangle_kernel(
                    float *input_depth,
                    float *output_depth,
                    float *input_color,
                    float *output_color,
                    float *output_angle,
                    float *cameras,
                    float *invKRs,
                    float *camlocs,
                    int batch_size,
                    int nrcams,
                    int nrchans,
                    int input_height,
                    int input_width)
{
    int colstep = 1;
    int rowstep = colstep * input_width;

    int camstep_d = rowstep * input_height;
    int btcstep_d = camstep_d * nrcams;

    int chnstep_c = rowstep * input_height;
    int camstep_c = chnstep_c * nrchans;
    int btcstep_c = camstep_c * nrcams;

    int clocs_camstep = 3;
    int clocs_btcstep = clocs_camstep * nrcams;
    int invKRs_camstep = 9;
    int invKRs_btcstep = invKRs_camstep * nrcams;
    
    int proj[2];
    float w[3];

    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < batch_size; b += blockDim.x * gridDim.x) {
        float *camera0 = cameras + b * nrcams * 12;
        float *camloc0 = camlocs + b * clocs_btcstep;
    for (int n = 0; n < nrcams; n++) {
    for (int in_row = blockIdx.y * blockDim.y + threadIdx.y; in_row < input_height; in_row += blockDim.y * gridDim.y) {
    for (int in_col = blockIdx.z * blockDim.z + threadIdx.z; in_col < input_width;  in_col += blockDim.z * gridDim.z) {
        float depth_n = input_depth[b * btcstep_d + n * camstep_d + in_row * rowstep + in_col * colstep];
        if(n == 0) {
            // simply copy the first camera's color
            for(int c = 0; c < nrchans; c++) {
                float color_n = input_color[b * btcstep_c + n * camstep_c + c * chnstep_c + in_row * rowstep + in_col * colstep];
                output_color[b * btcstep_c + n * camstep_c + c * chnstep_c + in_row * rowstep + in_col * colstep] = color_n;
            }
            output_angle[b * btcstep_d + n * camstep_d + in_row * rowstep + in_col * colstep] = 1.0f;
        }
        else if(depth_n > 0) {
            // cast this point into space
            float *camloc = camloc0 + n * clocs_camstep;
            float *invKR = invKRs + b * invKRs_btcstep + n * invKRs_camstep;
            MYTH_unproject_point(camloc, invKR, in_col, in_row, depth_n, w);
            // project it onto the first camera again
            if(MYTH_project_point(camera0, w, proj, input_width, input_height)) {
                float zbuffer = output_depth[b * btcstep_d + n * camstep_d + proj[1] * rowstep + proj[0] * colstep];
                float this_z = MYTH_get_point_depth(camera0, w);
                if(this_z <= zbuffer) {
                    for(int c = 0; c < nrchans; c++) {
                        float color_n = input_color[b * btcstep_c + n * camstep_c + c * chnstep_c + in_row * rowstep + in_col * colstep];
                        output_color[b * btcstep_c + n * camstep_c + c * chnstep_c + proj[1] * rowstep + proj[0] * colstep] = color_n;
                    }
                    // also pass the cosine of the angle between the viewing lines as a feature
                    float angle = 0.0f;
                    float norm = 0.0f, norm0 = 0.0f;
                    for(int i = 0; i < 3; i++) {
                        angle += (camloc[i] - w[i]) * (camloc0[i] - w[i]);
                        norm += (camloc[i] - w[i]) * (camloc[i] - w[i]);
                        norm0 += (camloc0[i] - w[i]) * (camloc0[i] - w[i]);
                    }
                    output_angle[b * btcstep_d + n * camstep_d + proj[1] * rowstep + proj[0] * colstep] = angle / sqrt(norm * norm0);
                }
            }
        }
    }
    }
    }
    }
}

//the input dimension is (B x N x 1 x H x W)
//the output dimension is (B x N x 1 x H x W)
extern "C" void DepthColorAngleReprojectionNeighbours_updateOutput_gpu(
    THCudaTensor *input_depth,
    THCudaTensor *output_depth,
    THCudaTensor *input_color,
    THCudaTensor *output_color,
    THCudaTensor *output_angle,
    THCudaTensor *cameras,
    THCudaTensor *invKRs,
    THCudaTensor *camlocs)
{
    int blkdim = 16;
    int batch_size = THCudaTensor_size(state, input_depth,0);
    int nrviews = THCudaTensor_size(state, input_depth,1);
    int color_channels = THCudaTensor_size(state, input_color,2);
    int input_height = THCudaTensor_size(state, input_depth,3);
    int input_width = THCudaTensor_size(state, input_depth,4);
    
    // we will use one thread for all depth hypotheses, to save some calculations regarding the directions and matrix inversions
    const dim3 block = dim3(1, blkdim, blkdim);
    const dim3 grid = dim3(1,ceil(THCudaTensor_size(state, output_depth, 3)*1.0f/blkdim), ceil(THCudaTensor_size(state, output_depth, 4)*1.0f/blkdim));
    float *input_depth_p = THCudaTensor_data(state, input_depth);
    float *output_depth_p = THCudaTensor_data(state, output_depth);
    float *input_color_p = THCudaTensor_data(state, input_color);
    float *output_color_p = THCudaTensor_data(state, output_color);
    float *output_angle_p = THCudaTensor_data(state, output_angle);
    float *cameras_p = THCudaTensor_data(state, cameras);
    float *invKRs_p = THCudaTensor_data(state, invKRs);
    float *camlocs_p = THCudaTensor_data(state, camlocs);

    cudaStream_t stream = THCState_getCurrentStream(state);
    DepthColorAngleReprojectionNeighbours_forward_depth_kernel<<<grid, block, 0, stream>>>(input_depth_p, output_depth_p, cameras_p, invKRs_p, camlocs_p, batch_size, nrviews, input_height, input_width);
    cudaStreamSynchronize(stream);
    DepthColorAngleReprojectionNeighbours_forward_colorangle_kernel<<<grid, block, 0, stream>>>(input_depth_p, output_depth_p, input_color_p, output_color_p, output_angle_p, cameras_p, invKRs_p, camlocs_p, batch_size, nrviews, color_channels, input_height, input_width);

    THCudaCheck(cudaGetLastError());
}


__global__ void DepthColorAngleReprojectionNeighbours_backward_color_kernel(
                    float *input_depth,
                    float *output_depth,
                    float *dloss_input_color,
                    float *dloss_output_color,
                    float *cameras,
                    float *invKRs,
                    float *camlocs,
                    int batch_size,
                    int nrcams,
                    int nrchans,
                    int input_height,
                    int input_width)
{
    int colstep = 1;
    int rowstep = colstep * input_width;

    int camstep_d = rowstep * input_height;
    int btcstep_d = camstep_d * nrcams;

    int chnstep_c = rowstep * input_height;
    int camstep_c = chnstep_c * nrchans;
    int btcstep_c = camstep_c * nrcams;

    int clocs_camstep = 3;
    int clocs_btcstep = clocs_camstep * nrcams;
    int invKRs_camstep = 9;
    int invKRs_btcstep = invKRs_camstep * nrcams;
    
    int proj[2];
    float w[3];

    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < batch_size; b += blockDim.x * gridDim.x) {
        float *camera0 = cameras + b * nrcams * 12;
        float *camloc0 = camlocs + b * clocs_btcstep;
    for (int n = 0; n < nrcams; n++) {
    for (int in_row = blockIdx.y * blockDim.y + threadIdx.y; in_row < input_height; in_row += blockDim.y * gridDim.y) {
    for (int in_col = blockIdx.z * blockDim.z + threadIdx.z; in_col < input_width;  in_col += blockDim.z * gridDim.z) {
        float depth_n = input_depth[b * btcstep_d + n * camstep_d + in_row * rowstep + in_col * colstep];
        if(n == 0) {
            // simply copy the first camera's color
            for(int c = 0; c < nrchans; c++) {
                float dloss_output_n = dloss_output_color[b * btcstep_c + n * camstep_c + c * chnstep_c + in_row * rowstep + in_col * colstep];
                atomicAdd(
                    dloss_input_color + b * btcstep_c + n * camstep_c + c * chnstep_c + in_row * rowstep + in_col * colstep,
                    dloss_output_n
                );
            }
        }
        else if(depth_n > 0) {
            // cast this point into space
            float *camloc = camloc0 + n * clocs_camstep;
            float *invKR = invKRs + b * invKRs_btcstep + n * invKRs_camstep;
            MYTH_unproject_point(camloc, invKR, in_col, in_row, depth_n, w);
            // project it onto the first camera again
            if(MYTH_project_point(camera0, w, proj, input_width, input_height)) {
                float zbuffer = output_depth[b * btcstep_d + n * camstep_d + proj[1] * rowstep + proj[0] * colstep];
                float this_z = MYTH_get_point_depth(camera0, w);
                if(this_z <= zbuffer) {
                    for(int c = 0; c < nrchans; c++) {
                        float dloss_output_n = dloss_output_color[b * btcstep_c + n * camstep_c + c * chnstep_c + proj[1] * rowstep + proj[0] * colstep];
                        atomicAdd(
                            dloss_input_color + b * btcstep_c + n * camstep_c + c * chnstep_c + in_row * rowstep + in_col * colstep,
                            dloss_output_n
                        );
                    }
                }
            }
        }
    }
    }
    }
    }
}

//the input dimension is (B x N x 1 x H x W)
//the output dimension is (B x N x 1 x H x W)
extern "C" void DepthColorAngleReprojectionNeighbours_updateGradInput_gpu(
    THCudaTensor *input_depth,
    THCudaTensor *output_depth,
    THCudaTensor *dloss_input_color,
    THCudaTensor *dloss_output_color,
    THCudaTensor *cameras,
    THCudaTensor *invKRs,
    THCudaTensor *camlocs)
{
    int blkdim = 16;
    int batch_size = THCudaTensor_size(state, input_depth,0);
    int nrviews = THCudaTensor_size(state, input_depth,1);
    int color_channels = THCudaTensor_size(state, dloss_output_color,2);
    int input_height = THCudaTensor_size(state, input_depth,3);
    int input_width = THCudaTensor_size(state, input_depth,4);
    
    // we will use one thread for all depth hypotheses, to save some calculations regarding the directions and matrix inversions
    const dim3 block = dim3(1, blkdim, blkdim);
    const dim3 grid = dim3(1,ceil(THCudaTensor_size(state, output_depth, 3)*1.0f/blkdim), ceil(THCudaTensor_size(state, output_depth, 4)*1.0f/blkdim));
    float *input_depth_p = THCudaTensor_data(state, input_depth);
    float *output_depth_p = THCudaTensor_data(state, output_depth);
    float *dloss_input_color_p = THCudaTensor_data(state, dloss_input_color);
    float *dloss_output_color_p = THCudaTensor_data(state, dloss_output_color);
    float *cameras_p = THCudaTensor_data(state, cameras);
    float *invKRs_p = THCudaTensor_data(state, invKRs);
    float *camlocs_p = THCudaTensor_data(state, camlocs);

    cudaStream_t stream = THCState_getCurrentStream(state);
    DepthColorAngleReprojectionNeighbours_backward_color_kernel<<<grid, block, 0, stream>>>(input_depth_p, output_depth_p, dloss_input_color_p, dloss_output_color_p, cameras_p, invKRs_p, camlocs_p, batch_size, nrviews, color_channels, input_height, input_width);

    THCudaCheck(cudaGetLastError());
}