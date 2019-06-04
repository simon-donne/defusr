// we get as input two depth feature maps, the lower bound and the reprojected depth, as well as the features
// in addition, we get a culling mask, indicating which of the reprojected values are valid
// we wish to pool them, taking into the culling mask (only for the last two maps)
// The pooling ignores the first element of the second dimension (i.e. the center view)

// input dimensions: (B x N x 1 x H x W), (B x N x 1 x H x W), (B x N x F x H x W), (B x N x 1 x H x W)
// output dimension: (B x 1 x H x W), (B x 1 x H x W), (B x F x H x W)

#define SKIP_FIRST (1)
#define MAX_CHANNELS (16)

__global__ void AverageReprojectionPooling_forward_kernel(
    float *input_bound,
    float *input_depth,
    float *input_features,
    float *output_bound,
    float *output_depth,
    float *output_features,
    int B, int N, int F, int H, int W
) {
    float avg_bound, avg_depth;
    float avg_feature[MAX_CHANNELS];
    int bound_count, depth_count;
    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < B; b += blockDim.x * gridDim.x) {
    for (int v = blockIdx.y * blockDim.y + threadIdx.y; v < H; v += blockDim.y * gridDim.y) {
    for (int u = blockIdx.z * blockDim.z + threadIdx.z; u < W; u += blockDim.z * gridDim.z) {
        for(int f = 0; f < F; f++) {
            avg_feature[f] = 0.0f;
        }
        avg_bound = avg_depth = 0.0f;
        bound_count = 0;
        depth_count = 0;
        
        int pixel_index = u + v * W;
        for(int n = SKIP_FIRST; n < N; n++) {
            int image_index = (b * N + n)* H * W;
            if(input_bound[image_index + pixel_index] > 0) {
                avg_bound += input_bound[image_index + pixel_index];
                bound_count += 1;
            }
            if(input_depth[image_index + pixel_index] > 0) {
                avg_depth += input_depth[image_index + pixel_index];
                depth_count += 1;
                for(int f = 0; f < F; f++) {
                    avg_feature[f] += input_features[image_index * F + f * H * W + pixel_index];
                }
            }
        }

        if(bound_count > 0) {
            output_bound[b * H * W + pixel_index] = avg_bound / bound_count;
        }
        else {
            output_bound[b * H * W + pixel_index] = 0;
        }

        if(depth_count > 0) {
            output_depth[b * H * W + pixel_index] = avg_depth / depth_count;
            
            for(int f = 0; f < F; f++) {
                output_features[(b * F + f) * H * W + pixel_index] = avg_feature[f] / depth_count;
            }
        }
        else {
            output_depth[b * H * W + pixel_index] = 0;
            
            for(int f = 0; f < F; f++) {
                output_features[(b * F + f) * H * W + pixel_index] = 0;
            }
        }
    }
    }
    }
}

extern "C" void AverageReprojectionPooling_updateOutput_gpu(
    THCudaTensor *input_bound,
    THCudaTensor *input_depth,
    THCudaTensor *input_features,
    THCudaTensor *output_bound,
    THCudaTensor *output_depth,
    THCudaTensor *output_features)
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
    float *output_bound_p    = THCudaTensor_data(state, output_bound);
    float *output_depth_p    = THCudaTensor_data(state, output_depth);
    float *output_features_p = THCudaTensor_data(state, output_features);

    cudaStream_t stream = THCState_getCurrentStream(state);
    AverageReprojectionPooling_forward_kernel<<<grid, block, 0, stream>>>(
        input_bound_p,
        input_depth_p,
        input_features_p,
        output_bound_p,
        output_depth_p,
        output_features_p,
        B, N, F, H, W
    );

    THCudaCheck(cudaGetLastError());
}

__global__ void AverageReprojectionPooling_backward_kernel(
    float *input_bound,
    float *input_depth,
    float *dloss_input_bound,
    float *dloss_input_depth,
    float *dloss_input_features,
    float *dloss_output_bound,
    float *dloss_output_depth,
    float *dloss_output_features,
    int B, int N, int F, int H, int W
) {
    int bound_count, depth_count;
    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < B; b += blockDim.x * gridDim.x) {
    for (int v = blockIdx.y * blockDim.y + threadIdx.y; v < H; v += blockDim.y * gridDim.y) {
    for (int u = blockIdx.z * blockDim.z + threadIdx.z; u < W; u += blockDim.z * gridDim.z) {
        bound_count = 0;
        depth_count = 0;
        
        // pre-run to get the count
        int pixel_index = u + v * W;
        for(int n = SKIP_FIRST; n < N; n++) {
            int image_index = (b * N + n)* H * W;
            if(input_bound[image_index + pixel_index] > 0) {
                bound_count += 1;
            }
            if(input_depth[image_index + pixel_index] > 0) {
                depth_count += 1;
            }
        }
        
        for(int n = SKIP_FIRST; n < N; n++) {
            int image_index = (b * N + n)* H * W;
            if(bound_count > 0 && input_bound[image_index + pixel_index] > 0) {
                atomicAdd(
                    dloss_input_bound + image_index + pixel_index,
                    dloss_output_bound[b * H * W + pixel_index] / bound_count
                );
            }
            if(depth_count > 0 && input_depth[image_index + pixel_index] > 0) {
                atomicAdd(
                    dloss_input_depth + image_index + pixel_index,
                    dloss_output_depth[b * H * W + pixel_index] / depth_count
                );
                
                for(int f = 0; f < F; f++) {
                    atomicAdd(
                        dloss_input_features + (b * F * N + n * F + f) * H * W + pixel_index,
                        dloss_output_features[(b * F + f) * H * W + pixel_index] / depth_count
                    );
                }
            }
        }
    }
    }
    }
}

extern "C" void AverageReprojectionPooling_updateGradInput_gpu(
    THCudaTensor *input_bound,
    THCudaTensor *input_depth,
    THCudaTensor *dloss_input_bound,
    THCudaTensor *dloss_input_depth,
    THCudaTensor *dloss_input_features,
    THCudaTensor *dloss_output_bound,
    THCudaTensor *dloss_output_depth,
    THCudaTensor *dloss_output_features)
{
    int B = THCudaTensor_size(state, dloss_input_features, 0);
    int N = THCudaTensor_size(state, dloss_input_features, 1);
    int F = THCudaTensor_size(state, dloss_input_features, 2);
    int H = THCudaTensor_size(state, dloss_input_features, 3);
    int W = THCudaTensor_size(state, dloss_input_features, 4);
 
    int blkdim = 16;   
    const dim3 block = dim3(1, blkdim, blkdim);
    const dim3 grid = dim3(1, 4, 4);

    float *input_bound_p      = THCudaTensor_data(state, input_bound);
    float *input_depth_p      = THCudaTensor_data(state, input_depth);
    float *dloss_input_bound_p     = THCudaTensor_data(state, dloss_input_bound);
    float *dloss_input_depth_p     = THCudaTensor_data(state, dloss_input_depth);
    float *dloss_input_features_p  = THCudaTensor_data(state, dloss_input_features);
    float *dloss_output_bound_p    = THCudaTensor_data(state, dloss_output_bound);
    float *dloss_output_depth_p    = THCudaTensor_data(state, dloss_output_depth);
    float *dloss_output_features_p = THCudaTensor_data(state, dloss_output_features);

    cudaStream_t stream = THCState_getCurrentStream(state);
    AverageReprojectionPooling_backward_kernel<<<grid, block, 0, stream>>>(
        input_bound_p,
        input_depth_p,
        dloss_input_bound_p,
        dloss_input_depth_p,
        dloss_input_features_p,
        dloss_output_bound_p,
        dloss_output_depth_p,
        dloss_output_features_p,
        B, N, F, H, W
    );

    THCudaCheck(cudaGetLastError());
}