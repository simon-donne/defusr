// we get as input two depth feature maps, the lower bound and the reprojected depth
// in addition, we get a culling mask, indicating which of the reprojected values are valid
// we wish to pool them, taking into the culling mask (only for the last two maps)
// The pooling ignores the first element of the second dimension (i.e. the center view)

// input dimensions: (B x N x 1 x H x W), (B x N x 1 x H x W), (B x N x 1 x H x W)
// output dimension: (B x 1 x H x W), (B x 1 x H x W)

#define SKIP_FIRST (1)

__global__ void VariationReprojectionPooling_forward_kernel(
    float *avg_bound,
    float *avg_depth,
    float *input_bound,
    float *input_depth,
    float *input_mask,
    float *output_bound,
    float *output_depth,
    int B, int N, int H, int W
) {
    float var_bound, var_depth;
    int count;
    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < B; b += blockDim.x * gridDim.x) {
    for (int v = blockIdx.y * blockDim.y + threadIdx.y; v < H; v += blockDim.y * gridDim.y) {
    for (int u = blockIdx.z * blockDim.z + threadIdx.z; u < W; u += blockDim.z * gridDim.z) {
        var_bound = var_depth = 0.0f;
        count = 0;
        
        int pixel_index = u + v * W;
        for(int n = SKIP_FIRST; n < N; n++) {
            int image_index = (b * N + n)* H * W;
            var_bound += powf(input_bound[image_index + pixel_index] - avg_bound[b * H * W + pixel_index], 2);

            if(input_mask[image_index + pixel_index] > 0) {
                var_depth += powf(input_depth[image_index + pixel_index] - avg_depth[b * H * W + pixel_index], 2);
                count += 1;
            }
        }

        output_bound[b * H * W + pixel_index] = (var_bound / (N - SKIP_FIRST));

        if(var_depth > 0) {
            output_depth[b * H * W + pixel_index] = (var_depth / count);
        }
    }
    }
    }
}

extern "C" void VariationReprojectionPooling_updateOutput_gpu(
    THCudaTensor *avg_bound,
    THCudaTensor *avg_depth,
    THCudaTensor *input_bound,
    THCudaTensor *input_depth,
    THCudaTensor *output_bound,
    THCudaTensor *output_depth,
    THCudaTensor *input_mask)
{
    int B = THCudaTensor_size(state, input_bound, 0);
    int N = THCudaTensor_size(state, input_bound, 1);
    int H = THCudaTensor_size(state, input_bound, 3);
    int W = THCudaTensor_size(state, input_bound, 4);
 
    const dim3 block = dim3(1, 16, 16);
    const dim3 grid = dim3(1, 4, 4);

    float *avg_bound_p       = THCudaTensor_data(state, avg_bound);
    float *avg_depth_p       = THCudaTensor_data(state, avg_depth);
    float *input_bound_p     = THCudaTensor_data(state, input_bound);
    float *input_depth_p     = THCudaTensor_data(state, input_depth);
    float *input_mask_p      = THCudaTensor_data(state, input_mask);
    float *output_bound_p    = THCudaTensor_data(state, output_bound);
    float *output_depth_p    = THCudaTensor_data(state, output_depth);

    cudaStream_t stream = THCState_getCurrentStream(state);
    VariationReprojectionPooling_forward_kernel<<<grid, block, 0, stream>>>(
        avg_bound_p,
        avg_depth_p,
        input_bound_p,
        input_depth_p,
        input_mask_p,
        output_bound_p,
        output_depth_p,
        B, N, H, W
    );

    THCudaCheck(cudaGetLastError());
}

__global__ void VariationReprojectionPooling_backward_kernel(
    float *input_bound,
    float *input_depth,
    float *avg_bound,
    float *avg_depth,
    float *dloss_input_bound,
    float *dloss_input_depth,
    float *dloss_input_avg_bound,
    float *dloss_input_avg_depth,
    float *input_mask,
    float *dloss_output_bound,
    float *dloss_output_depth,
    int B, int N, int H, int W
) {
    int count = 0;
    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < B; b += blockDim.x * gridDim.x) {
    for (int v = blockIdx.y * blockDim.y + threadIdx.y; v < H; v += blockDim.y * gridDim.y) {
    for (int u = blockIdx.z * blockDim.z + threadIdx.z; u < W; u += blockDim.z * gridDim.z) {
        count = 0;

        int pixel_index = u + v * W;
        for(int n = SKIP_FIRST; n < N; n++) {
            int image_index = (b * N + n)* H * W;

            atomicAdd(
                dloss_input_bound + image_index + pixel_index,
                2*(input_bound[image_index + pixel_index] - avg_bound[b * H * W + pixel_index]) * dloss_output_bound[b * H * W + pixel_index] / (N - SKIP_FIRST)
            );
            atomicAdd(
                dloss_input_avg_bound + b * H * W + pixel_index,
                - 2*(input_bound[image_index + pixel_index] - avg_bound[b * H * W + pixel_index]) * dloss_output_bound[b * H * W + pixel_index] / (N - SKIP_FIRST)
            );

            if(input_mask[image_index + pixel_index] > 0) {
                count += 1;
            }
        }
        
        if(count > 0) {
            for(int n = SKIP_FIRST; n < N; n++) {
                int image_index = (b * N + n)* H * W;

                if(input_mask[image_index + pixel_index] > 0) {
                    atomicAdd(
                        dloss_input_depth + image_index + pixel_index,
                        2*(input_depth[image_index + pixel_index] - avg_depth[b * H * W + pixel_index]) * dloss_output_depth[b * H * W + pixel_index] / count
                    );

                    atomicAdd(
                        dloss_input_avg_depth + b * H * W + pixel_index,
                        - 2*(input_depth[image_index + pixel_index] - avg_depth[b * H * W + pixel_index]) * dloss_output_depth[b * H * W + pixel_index] / count
                    );
                }
            }
        }
    }
    }
    }
}

extern "C" void VariationReprojectionPooling_updateGradInput_gpu(
    THCudaTensor *avg_bound,
    THCudaTensor *avg_depth,
    THCudaTensor *input_bound,
    THCudaTensor *input_depth,
    THCudaTensor *dloss_input_bound,
    THCudaTensor *dloss_input_depth,
    THCudaTensor *dloss_input_avg_bound,
    THCudaTensor *dloss_input_avg_depth,
    THCudaTensor *dloss_output_bound,
    THCudaTensor *dloss_output_depth,
    THCudaTensor *input_mask)
{
    int B = THCudaTensor_size(state, dloss_input_bound, 0);
    int N = THCudaTensor_size(state, dloss_input_bound, 1);
    int H = THCudaTensor_size(state, dloss_input_bound, 3);
    int W = THCudaTensor_size(state, dloss_input_bound, 4);
 
    int blkdim = 16;   
    const dim3 block = dim3(1, blkdim, blkdim);
    const dim3 grid = dim3(1, 4, 4);

    float *avg_bound_p           = THCudaTensor_data(state, avg_bound);
    float *avg_depth_p           = THCudaTensor_data(state, avg_depth);
    float *input_bound_p           = THCudaTensor_data(state, input_bound);
    float *input_depth_p           = THCudaTensor_data(state, input_depth);
    float *dloss_input_bound_p     = THCudaTensor_data(state, dloss_input_bound);
    float *dloss_input_depth_p     = THCudaTensor_data(state, dloss_input_depth);
    float *dloss_input_avg_bound_p     = THCudaTensor_data(state, dloss_input_avg_bound);
    float *dloss_input_avg_depth_p     = THCudaTensor_data(state, dloss_input_avg_depth);
    float *input_mask_p            = THCudaTensor_data(state, input_mask);
    float *dloss_output_bound_p    = THCudaTensor_data(state, dloss_output_bound);
    float *dloss_output_depth_p    = THCudaTensor_data(state, dloss_output_depth);

    cudaStream_t stream = THCState_getCurrentStream(state);
    VariationReprojectionPooling_backward_kernel<<<grid, block, 0, stream>>>(
        input_bound_p,
        input_depth_p,
        avg_bound_p,
        avg_depth_p,
        dloss_input_bound_p,
        dloss_input_depth_p,
        dloss_input_avg_bound_p,
        dloss_input_avg_depth_p,
        input_mask_p,
        dloss_output_bound_p,
        dloss_output_depth_p,
        B, N, H, W
    );

    THCudaCheck(cudaGetLastError());
}