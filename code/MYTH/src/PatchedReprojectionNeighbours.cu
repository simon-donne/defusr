__global__ void PatchedReprojectionNeighbours_forward_depth_kernel(
    float *input_depths_neighbours,
    float *output_depth_reprojected,
    float *camera_center,
    float *invKRs_neighbours,
    float *camlocs_neighbours,
    int B, int N,
    int H_in, int W_in,
    int H_out, int W_out)
{
    int proj[2];
    float w[3];

    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < B; b += blockDim.x * gridDim.x) {
        float *camera0 = camera_center + b * 12;
    for (int n = 0; n < N; n++) {
    for (int in_row = blockIdx.y * blockDim.y + threadIdx.y; in_row < H_in; in_row += blockDim.y * gridDim.y) {
    for (int in_col = blockIdx.z * blockDim.z + threadIdx.z; in_col < W_in; in_col += blockDim.z * gridDim.z) {
        float depth_n = input_depths_neighbours[((b * N + n) * H_in + in_row) * W_in + in_col];
        if(depth_n > 0) {
            // cast this point into space
            float *camloc = camlocs_neighbours + (b * N + n) * 3;
            float *invKR = invKRs_neighbours + (b * N + n) * 9;
            MYTH_unproject_point(camloc, invKR, in_col, in_row, depth_n, w);
            // project it onto the first camera again
            if(MYTH_project_point(camera0, w, proj, W_out, H_out)) {
                MYTH_atomicMinf(
                    output_depth_reprojected + ((b * N + n) * H_out + proj[1]) * W_out + proj[0],
                    MYTH_get_point_depth(camera0, w)
                );
            }
        }
    }
    }
    }
    }
}

__global__ void PatchedReprojectionNeighbours_forward_color_kernel(
    float *input_depths_neighbours,
    float *output_depth_reprojected,
    float *input_colors_neighbours,
    float *output_color_reprojected,
    float *camera_center,
    float *invKRs_neighbours,
    float *camlocs_neighbours,
    int B, int N, int C,
    int H_in, int W_in,
    int H_out, int W_out)
{
    int proj[2];
    float w[3];

    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < B; b += blockDim.x * gridDim.x) {
        float *camera0 = camera_center + b * 12;
    for (int n = 0; n < N; n++) {
    for (int in_row = blockIdx.y * blockDim.y + threadIdx.y; in_row < H_in; in_row += blockDim.y * gridDim.y) {
    for (int in_col = blockIdx.z * blockDim.z + threadIdx.z; in_col < W_in; in_col += blockDim.z * gridDim.z) {
        float depth_n = input_depths_neighbours[((b * N + n) * H_in + in_row) * W_in + in_col];
        if(depth_n > 0) {
            // cast this point into space
            float *camloc = camlocs_neighbours + (b * N + n) * 3;
            float *invKR = invKRs_neighbours + (b * N + n) * 9;
            MYTH_unproject_point(camloc, invKR, in_col, in_row, depth_n, w);
            // project it onto the first camera again
            if(MYTH_project_point(camera0, w, proj, W_out, H_out)) {
                float zbuffer = output_depth_reprojected[((b * N + n) * H_out + proj[1]) * W_out + proj[0]];
                float this_z = MYTH_get_point_depth(camera0, w);
                if(this_z <= zbuffer) {
                    for(int c = 0; c < C; c++) {
                        float color_n = input_colors_neighbours[(((b * N + n) * C + c) * H_in + in_row) * W_in + in_col];
                        output_color_reprojected[(((b * N + n) * C + c) * H_out + proj[1]) * W_out + proj[0]] = color_n;
                    }
                }
            }
        }
    }
    }
    }
    }
}

extern "C" void PatchedReprojectionNeighbours_updateOutput_gpu(
    THCudaTensor *input_depths_neighbours,
    THCudaTensor *input_colors_neighbours,
    THCudaTensor *output_depth_reprojected,
    THCudaTensor *output_color_reprojected,
    THCudaTensor *camera_center,
    THCudaTensor *invKRs_neighbours,
    THCudaTensor *camlocs_neighbours)
{
    int blkdim = 16;
    int B = THCudaTensor_size(state, input_colors_neighbours, 0);
    int N = THCudaTensor_size(state, input_colors_neighbours, 1);
    int C = THCudaTensor_size(state, input_colors_neighbours, 2);
    int H_in = THCudaTensor_size(state, input_colors_neighbours, 3);
    int W_in = THCudaTensor_size(state, input_colors_neighbours, 4);
    int H_out = THCudaTensor_size(state, output_color_reprojected, 3);
    int W_out = THCudaTensor_size(state, output_color_reprojected, 4);
    
    const dim3 block = dim3(1, blkdim, blkdim);
    const dim3 grid = dim3(1, 8, 8);
    
    float *input_depths_neighbours_p = THCudaTensor_data(state, input_depths_neighbours);
    float *input_colors_neighbours_p = THCudaTensor_data(state, input_colors_neighbours);
    float *output_depth_reprojected_p = THCudaTensor_data(state, output_depth_reprojected);
    float *output_color_reprojected_p = THCudaTensor_data(state, output_color_reprojected);
    float *camera_center_p = THCudaTensor_data(state, camera_center);
    float *invKRs_neighbours_p = THCudaTensor_data(state, invKRs_neighbours);
    float *camlocs_neighbours_p = THCudaTensor_data(state, camlocs_neighbours);

    cudaStream_t stream = THCState_getCurrentStream(state);
    PatchedReprojectionNeighbours_forward_depth_kernel<<<grid, block, 0, stream>>>(
        input_depths_neighbours_p,
        output_depth_reprojected_p,
        camera_center_p,
        invKRs_neighbours_p,
        camlocs_neighbours_p,
        B, N, H_in, W_in, H_out, W_out);
    cudaStreamSynchronize(stream);
    PatchedReprojectionNeighbours_forward_color_kernel<<<grid, block, 0, stream>>>(
        input_depths_neighbours_p,
        output_depth_reprojected_p,
        input_colors_neighbours_p,
        output_color_reprojected_p,
        camera_center_p,
        invKRs_neighbours_p,
        camlocs_neighbours_p,
        B, N, C,
        H_in, W_in, H_out, W_out
    );

    THCudaCheck(cudaGetLastError());
}


__global__ void PatchedReprojectionNeighbours_backward_color_kernel(
    float *input_depths_neighbours,
    float *output_depth_reprojected,
    float *dloss_dinput_colors_neighbours,
    float *dloss_doutput_color_reprojected,
    float *camera_center,
    float *invKRs_neighbours,
    float *camlocs_neighbours,
    int B, int N, int C,
    int H_in, int W_in, int H_out, int W_out)
{
    int proj[2];
    float w[3];

    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < B; b += blockDim.x * gridDim.x) {
        float *camera0 = camera_center + b * 12;
    for (int n = 0; n < N; n++) {
    for (int in_row = blockIdx.y * blockDim.y + threadIdx.y; in_row < H_in; in_row += blockDim.y * gridDim.y) {
    for (int in_col = blockIdx.z * blockDim.z + threadIdx.z; in_col < W_in; in_col += blockDim.z * gridDim.z) {
        float depth_n = input_depths_neighbours[((b * N + n) * H_in + in_row) * W_in + in_col];
        if(depth_n > 0) {
            // cast this point into space
            float *camloc = camlocs_neighbours + (b * N + n) * 3;
            float *invKR = invKRs_neighbours + (b * N + n) * 9;
            MYTH_unproject_point(camloc, invKR, in_col, in_row, depth_n, w);
            // project it onto the first camera again
            if(MYTH_project_point(camera0, w, proj, W_out, H_out)) {
                float zbuffer = output_depth_reprojected[((b * N + n) * H_out + proj[1]) * W_out + proj[0]];
                float this_z = MYTH_get_point_depth(camera0, w);
                if(this_z <= zbuffer) {
                    for(int c = 0; c < C; c++) {
                        float dloss_doutput_n = dloss_doutput_color_reprojected[(((b * N + n) * C + c) * H_out + proj[1]) * W_out + proj[0]];
                        atomicAdd(
                            dloss_dinput_colors_neighbours + (((b * N + n) * C + c) * H_in + in_row) * W_in + in_col,
                            dloss_doutput_n
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
extern "C" void PatchedReprojectionNeighbours_updateGradInput_gpu(
    THCudaTensor *input_depths_neighbours,
    THCudaTensor *output_depth_reprojected,
    THCudaTensor *dloss_dinput_colors_neighbours,
    THCudaTensor *dloss_doutput_color_reprojected,
    THCudaTensor *camera_center,
    THCudaTensor *invKRs_neighbours,
    THCudaTensor *camlocs_neighbours)
{
    int blkdim = 16;
    int B = THCudaTensor_size(state, dloss_dinput_colors_neighbours, 0);
    int N = THCudaTensor_size(state, dloss_dinput_colors_neighbours, 1);
    int C = THCudaTensor_size(state, dloss_dinput_colors_neighbours, 2);
    int H_in = THCudaTensor_size(state, dloss_dinput_colors_neighbours, 3);
    int W_in = THCudaTensor_size(state, dloss_dinput_colors_neighbours, 4);
    int H_out = THCudaTensor_size(state, dloss_doutput_color_reprojected, 3);
    int W_out = THCudaTensor_size(state, dloss_doutput_color_reprojected, 4);
    
    const dim3 block = dim3(1, blkdim, blkdim);
    const dim3 grid = dim3(1, 8, 8);
    
    float *input_depths_neighbours_p = THCudaTensor_data(state, input_depths_neighbours);
    float *dloss_dinput_colors_neighbours_p = THCudaTensor_data(state, dloss_dinput_colors_neighbours);
    float *output_depth_reprojected_p = THCudaTensor_data(state, output_depth_reprojected);
    float *dloss_doutput_color_reprojected_p = THCudaTensor_data(state, dloss_doutput_color_reprojected);
    float *camera_center_p = THCudaTensor_data(state, camera_center);
    float *invKRs_neighbours_p = THCudaTensor_data(state, invKRs_neighbours);
    float *camlocs_neighbours_p = THCudaTensor_data(state, camlocs_neighbours);

    cudaStream_t stream = THCState_getCurrentStream(state);
    PatchedReprojectionNeighbours_backward_color_kernel<<<grid, block, 0, stream>>>(
        input_depths_neighbours_p,
        output_depth_reprojected_p,
        dloss_dinput_colors_neighbours_p,
        dloss_doutput_color_reprojected_p,
        camera_center_p,
        invKRs_neighbours_p,
        camlocs_neighbours_p,
        B, N, C,
        H_in, W_in, H_out, W_out
    );

    THCudaCheck(cudaGetLastError());
}