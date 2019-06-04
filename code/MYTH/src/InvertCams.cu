// Small bit of CUDA code for inverting a bunch of 3x3 cameras, and getting their camera locations

__global__ void InvertCams_kernel(
    float *cameras,
    float *invKRs,
    float *camlocs,
    int B
) {
    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < B; b += blockDim.x * gridDim.x) {
        float *cam = cameras + b*12;
        float KR_det = cam[0] * cam[5] * cam[10];
                KR_det += cam[1] * cam[6] * cam[8];
                KR_det += cam[2] * cam[4] * cam[9];
                KR_det -= cam[2] * cam[5] * cam[8];
                KR_det -= cam[1] * cam[4] * cam[10];
                KR_det -= cam[0] * cam[6] * cam[9];
        float KR_det_inv = 1.0f/KR_det;
        float *invKR = invKRs + b * 9;
        invKR[0] = (cam[5] * cam[10] - cam[6] * cam[9]) * KR_det_inv;
        invKR[1] = (cam[2] * cam[9] - cam[1] * cam[10]) * KR_det_inv;
        invKR[2] = (cam[1] * cam[6] - cam[2] * cam[5]) * KR_det_inv;
        invKR[3] = (cam[6] * cam[8] - cam[4] * cam[10]) * KR_det_inv;
        invKR[4] = (cam[0] * cam[10] - cam[2] * cam[8]) * KR_det_inv;
        invKR[5] = (cam[2] * cam[4] - cam[0] * cam[6]) * KR_det_inv;
        invKR[6] = (cam[4] * cam[9] - cam[5] * cam[8]) * KR_det_inv;
        invKR[7] = (cam[1] * cam[8] - cam[0] * cam[9]) * KR_det_inv;
        invKR[8] = (cam[0] * cam[5] - cam[1] * cam[4]) * KR_det_inv;
        float *camloc = camlocs + b * 3;
        camloc[0] = -(invKR[0] * cam[3] + invKR[1] * cam[7] + invKR[2] * cam[11]);
        camloc[1] = -(invKR[3] * cam[3] + invKR[4] * cam[7] + invKR[5] * cam[11]);
        camloc[2] = -(invKR[6] * cam[3] + invKR[7] * cam[7] + invKR[8] * cam[11]);
    }
}

// yes this is horribly slow/inefficient/whatever
// I just don't want to copying to CPU for this
extern "C" void InvertCams_gpu(
    THCudaTensor *cameras,
    THCudaTensor *invKRs,
    THCudaTensor *camlocs)
{
    int B = THCudaTensor_size(state, cameras, 0);
 
    const dim3 block = dim3(1, 1, 1);
    const dim3 grid = dim3(1, 1, 1);

    float *cameras_p     = THCudaTensor_data(state, cameras);
    float *invKRs_p      = THCudaTensor_data(state, invKRs);
    float *camlocs_p     = THCudaTensor_data(state, camlocs);

    cudaStream_t stream = THCState_getCurrentStream(state);
    InvertCams_kernel<<<grid, block, 0, stream>>>(
        cameras_p,
        invKRs_p,
        camlocs_p,
        B
    );

    THCudaCheck(cudaGetLastError());
}