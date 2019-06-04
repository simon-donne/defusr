#include <stdio.h>

__global__ void bruteforce_sparsity_count_kernel(
    float *points,
    float *counts1,
    float *counts2,
    int N,
    float sparsity1,
    float sparsity2
){
    int n, m_high, m_off;
    int count1, count2;
    float dist;
    __shared__ float ptcache[2048*3]; //about 24Kb
    float locn[3];
    for(n = threadIdx.x + blockDim.x*blockIdx.x; n < N; n += blockDim.x*gridDim.x) {
        locn[0] = points[n];
        locn[1] = points[n+N];
        locn[2] = points[n+2*N];
        count1 = 0;
        count2 = 0;
        // loop through all points, count neighbours
        // this'll ... take a while
        for(m_high = 0; m_high < ( (N-1) / 2048 + 1); m_high++) {
            // update our entries in the shared pointcache
            for(m_off=threadIdx.x; m_off < min(2048, N - m_high * 2048); m_off += blockDim.x) {
                ptcache[m_off*3+0] = points[m_high*2048 + m_off];
                ptcache[m_off*3+1] = points[m_high*2048 + m_off + N];
                ptcache[m_off*3+2] = points[m_high*2048 + m_off + 2*N];
            }
            __syncthreads();
            //perform the distances for this part
            for(m_off=0; m_off<min(2048, N - m_high * 2048); m_off++) {
                dist = (locn[0] - ptcache[m_off*3+0]) * (locn[0] - ptcache[m_off*3+0])
                        + (locn[1] - ptcache[m_off*3+1]) * (locn[1] - ptcache[m_off*3+1])
                        + (locn[2] - ptcache[m_off*3+2]) * (locn[2] - ptcache[m_off*3+2]);
                count1 += dist < sparsity1 * sparsity1;
                count2 += dist < sparsity2 * sparsity2;
            }
            __syncthreads();
        }
        counts1[n] = count1;
        counts2[n] = count2;
    }
}

extern "C" void bruteforce_sparsity_count_gpu(
    THCudaTensor *points,
    THCudaTensor *counts1,
    THCudaTensor *counts2,
    int N,
    float sparsity1,
    float sparsity2)
{
    const dim3 block = dim3(1024, 1, 1);
    const dim3 grid = dim3(60, 1, 1);

    float *points_p     = THCudaTensor_data(state, points);
    float *counts1_p    = THCudaTensor_data(state, counts1);
    float *counts2_p    = THCudaTensor_data(state, counts2);

    cudaStream_t stream = THCState_getCurrentStream(state);
    bruteforce_sparsity_count_kernel<<<grid, block, 0, stream>>>(
        points_p,
        counts1_p,
        counts2_p,
        N,
        sparsity1,
        sparsity2
    );

    THCudaCheck(cudaGetLastError());
}

__global__ void bruteforce_distance_kernel(
    float *points_from,
    float *points_to,
    float *dists,
    int N,
    int M
){
    int n, m_high, m_off;
    float locn[3];
    float dist, dist_m;
    __shared__ float ptcache[2048*3]; // 24Kb
    // threads that don't actually have any points associated are still used to help
    // out with the data loading (querying with a low number of points)
    for(n = threadIdx.x + blockDim.x*blockIdx.x; n < max(N, blockDim.x); n += blockDim.x*gridDim.x) {
        if(n < N) {
            locn[0] = points_from[n];
            locn[1] = points_from[n+N];
            locn[2] = points_from[n+2*N];
            dist = 1e9;
        }

        for(m_high = 0; m_high < ( (M-1) / 2048 + 1); m_high++) {
            //update the shared pointcache
            for(m_off = threadIdx.x; m_off < min(2048, M - m_high * 2048); m_off += blockDim.x) {
                ptcache[m_off*3+0] = points_to[m_high*2048 + m_off];
                ptcache[m_off*3+1] = points_to[m_high*2048 + m_off + M];
                ptcache[m_off*3+2] = points_to[m_high*2048 + m_off + 2*M];
            }
            __syncthreads();
            //do the distances for this part
            if(n < N) {
                for(m_off=0; m_off<min(2048, M - m_high * 2048); m_off++) {
                    dist_m = (locn[0] - ptcache[m_off*3+0]) * (locn[0] - ptcache[m_off*3+0])
                        + (locn[1] - ptcache[m_off*3+1]) * (locn[1] - ptcache[m_off*3+1])
                        + (locn[2] - ptcache[m_off*3+2]) * (locn[2] - ptcache[m_off*3+2]);
                    dist = min(dist, dist_m);
                }
            }
            __syncthreads();
        }
        dists[n] = dist;
    }
}

extern "C" void bruteforce_distance_gpu(
    THCudaTensor *points_from,
    THCudaTensor *points_to,
    THCudaTensor *dists,
    int N,
    int M)
{
    const dim3 block = dim3(1024, 1, 1);
    const dim3 grid = dim3(60, 1, 1);

    float *points_from_p     = THCudaTensor_data(state, points_from);
    float *points_to_p    = THCudaTensor_data(state, points_to);
    float *dists_p    = THCudaTensor_data(state, dists);

    cudaStream_t stream = THCState_getCurrentStream(state);
    bruteforce_distance_kernel<<<grid, block, 0, stream>>>(
        points_from_p,
        points_to_p,
        dists_p,
        N,
        M
    );

    THCudaCheck(cudaGetLastError());
}