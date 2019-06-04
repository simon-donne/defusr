// we get as input two depth feature maps, the lower bound and the reprojected depth, as well as the features
// in addition, we get a culling mask, indicating which of the reprojected values are valid
// we wish to pool them, taking into the culling mask (only for the last two maps)
// The pooling ignores the first element of the second dimension (i.e. the center view)

// input dimensions: (B x N x 1 x H x W), (B x N x 1 x H x W), (B x N x F x H x W), (B x N x 1 x H x W)
// output dimension: (B x 1 x H x W), (B x 1 x H x W), (B x F x H x W)

#define PMRP_SKIP_FIRST (0)
#define MAX_CHANNELS (16)

__global__ void PatchedMaximumReprojectionPooling_forward_kernel(
    float *input_bound,
    float *input_depth,
    float *input_features,
    float *input_mask,
    float *output_bound,
    float *output_depth,
    float *output_features,
    int B, int N, int F, int H, int W
) {
    float max_bound, max_depth;
    float max_feature[MAX_CHANNELS];
    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < B; b += blockDim.x * gridDim.x) {
    for (int v = blockIdx.y * blockDim.y + threadIdx.y; v < H; v += blockDim.y * gridDim.y) {
    for (int u = blockIdx.z * blockDim.z + threadIdx.z; u < W; u += blockDim.z * gridDim.z) {
        for(int f = 0; f < F; f++) {
            max_feature[f] = 0.0f;
        }
        max_bound = 0.0f;
        max_depth = 0.0f;
        
        int pixel_index = u + v * W;
        for(int n = PMRP_SKIP_FIRST; n < N; n++) {
            int image_index = (b * N + n)* H * W;
            if(max_bound < input_bound[image_index + pixel_index]) {
                max_bound = input_bound[image_index + pixel_index];
            }
            
            if(input_mask[image_index + pixel_index] > 0) {
                if(max_depth < input_depth[image_index + pixel_index]) {
                    max_depth = input_depth[image_index + pixel_index];
                    for(int f = 0; f < F; f++) {
                        max_feature[f] = input_features[image_index * F + f * H * W + pixel_index];
                    }
                }
            }
        }

        output_bound[b * H * W + pixel_index] = max_bound;

        if(max_depth > 0) {
            output_depth[b * H * W + pixel_index] = max_depth;
            
            for(int f = 0; f < F; f++) {
                output_features[(b * F + f) * H * W + pixel_index] = max_feature[f];
            }
        }
    }
    }
    }
}

extern "C" void PatchedMaximumReprojectionPooling_updateOutput_gpu(
    THCudaTensor *input_bound,
    THCudaTensor *input_depth,
    THCudaTensor *input_features,
    THCudaTensor *output_bound,
    THCudaTensor *output_depth,
    THCudaTensor *output_features,
    THCudaTensor *input_mask)
{
    int B = THCudaTensor_size(state, input_features, 0);
    int N = THCudaTensor_size(state, input_features, 1);
    int F = THCudaTensor_size(state, input_features, 2);
    int H = THCudaTensor_size(state, input_features, 3);
    int W = THCudaTensor_size(state, input_features, 4);
 
    const dim3 block = dim3(1, 16, 16);
    const dim3 grid = dim3(1, 4, 4);

    float *input_bound_p     = THCudaTensor_data(state, input_bound);
    float *input_depth_p     = THCudaTensor_data(state, input_depth);
    float *input_features_p  = THCudaTensor_data(state, input_features);
    float *input_mask_p      = THCudaTensor_data(state, input_mask);
    float *output_bound_p    = THCudaTensor_data(state, output_bound);
    float *output_depth_p    = THCudaTensor_data(state, output_depth);
    float *output_features_p = THCudaTensor_data(state, output_features);

    cudaStream_t stream = THCState_getCurrentStream(state);
    PatchedMaximumReprojectionPooling_forward_kernel<<<grid, block, 0, stream>>>(
        input_bound_p,
        input_depth_p,
        input_features_p,
        input_mask_p,
        output_bound_p,
        output_depth_p,
        output_features_p,
        B, N, F, H, W
    );

    THCudaCheck(cudaGetLastError());
}

__global__ void PatchedMaximumReprojectionPooling_backward_kernel(
    float *input_bound,
    float *input_depth,
    float *dloss_input_bound,
    float *dloss_input_depth,
    float *dloss_input_features,
    float *input_mask,
    float *dloss_output_bound,
    float *dloss_output_depth,
    float *dloss_output_features,
    int B, int N, int F, int H, int W
) {
    float max_bound, max_depth;
    int bound_index, depth_index;
    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < B; b += blockDim.x * gridDim.x) {
    for (int v = blockIdx.y * blockDim.y + threadIdx.y; v < H; v += blockDim.y * gridDim.y) {
    for (int u = blockIdx.z * blockDim.z + threadIdx.z; u < W; u += blockDim.z * gridDim.z) {
        max_bound = 0.0f;
        max_depth = 0.0f;
        bound_index = 0;
        depth_index = 0;

        // pre-run to get the maximum index, both culled and bound
        int pixel_index = u + v * W;
        for(int n = PMRP_SKIP_FIRST; n < N; n++) {
            int image_index = (b * N + n)* H * W;
            if(max_bound < input_bound[image_index + pixel_index]) {
                max_bound = input_bound[image_index + pixel_index];
                bound_index = n;
            }
            
            if(input_mask[image_index + pixel_index] > 0) {
                if(max_depth < input_depth[image_index + pixel_index]) {
                    max_depth = input_depth[image_index + pixel_index];
                    depth_index = n;
                }
            }
        }
        
        int image_index_bound = (b * N + bound_index)* H * W;
        atomicAdd(
            dloss_input_bound + image_index_bound + pixel_index,
            dloss_output_bound[b * H * W + pixel_index]
        );

        if(max_depth > 0) {
            int image_index_depth = (b * N + depth_index)* H * W;
            atomicAdd(
                dloss_input_depth + image_index_depth + pixel_index,
                dloss_output_depth[b * H * W + pixel_index]
            );
            
            for(int f = 0; f < F; f++) {
                atomicAdd(
                    dloss_input_features + (b * F * N + depth_index * F + f) * H * W + pixel_index,
                    dloss_output_features[(b * F + f) * H * W + pixel_index]
                );
            }
        }
    }
    }
    }
}

extern "C" void PatchedMaximumReprojectionPooling_updateGradInput_gpu(
    THCudaTensor *input_bound,
    THCudaTensor *input_depth,
    THCudaTensor *dloss_input_bound,
    THCudaTensor *dloss_input_depth,
    THCudaTensor *dloss_input_features,
    THCudaTensor *dloss_output_bound,
    THCudaTensor *dloss_output_depth,
    THCudaTensor *dloss_output_features,
    THCudaTensor *input_mask)
{
    int B = THCudaTensor_size(state, dloss_input_features, 0);
    int N = THCudaTensor_size(state, dloss_input_features, 1);
    int F = THCudaTensor_size(state, dloss_input_features, 2);
    int H = THCudaTensor_size(state, dloss_input_features, 3);
    int W = THCudaTensor_size(state, dloss_input_features, 4);
 
    int blkdim = 16;   
    const dim3 block = dim3(1, blkdim, blkdim);
    const dim3 grid = dim3(1, 4, 4);

    float *input_bound_p     = THCudaTensor_data(state, input_bound);
    float *input_depth_p     = THCudaTensor_data(state, input_depth);
    float *dloss_input_bound_p     = THCudaTensor_data(state, dloss_input_bound);
    float *dloss_input_depth_p     = THCudaTensor_data(state, dloss_input_depth);
    float *dloss_input_features_p  = THCudaTensor_data(state, dloss_input_features);
    float *input_mask_p      = THCudaTensor_data(state, input_mask);
    float *dloss_output_bound_p    = THCudaTensor_data(state, dloss_output_bound);
    float *dloss_output_depth_p    = THCudaTensor_data(state, dloss_output_depth);
    float *dloss_output_features_p = THCudaTensor_data(state, dloss_output_features);

    cudaStream_t stream = THCState_getCurrentStream(state);
    PatchedMaximumReprojectionPooling_backward_kernel<<<grid, block, 0, stream>>>(
        input_bound_p,
        input_depth_p,
        dloss_input_bound_p,
        dloss_input_depth_p,
        dloss_input_features_p,
        input_mask_p,
        dloss_output_bound_p,
        dloss_output_depth_p,
        dloss_output_features_p,
        B, N, F, H, W
    );

    THCudaCheck(cudaGetLastError());
}