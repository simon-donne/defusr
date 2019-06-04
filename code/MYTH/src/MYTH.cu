// Overarching file collecting some utility functions + all implementations
// gpu layers/functions are only implemented for THCudaTensor (i.e. single-precision)
#include <TH.h>
#include <THC.h>

extern THCState *state;

#define MAX_ACCUMULATED_CHANNELS (32)

__inline__ __device__ float MYTH_get_point_depth(float *camera, float *win) {
    return camera[8]*win[0] + camera[9]*win[1] + camera[10]*win[2]+ camera[11];
}

__inline__ __device__ bool MYTH_project_point(float *camera, float *win, int *out, int input_width, int input_height) {
    float cx = camera[0]*win[0] + camera[1]*win[1] + camera[2]*win[2] + camera[3];
    float cy = camera[4]*win[0] + camera[5]*win[1] + camera[6]*win[2] + camera[7];
    float cz = MYTH_get_point_depth(camera, win);
    out[0] = int(cx / cz + 0.5f);
    out[1] = int(cy / cz + 0.5f);
    return (out[0] >= 0) && (out[1] >= 0) && (out[0]<input_width) && (out[1]<input_height);
}

__inline__ __device__ bool MYTH_project_pointf(float *camera, float *win, float *out, int input_width, int input_height) {
    float cx = camera[0]*win[0] + camera[1]*win[1] + camera[2]*win[2] + camera[3];
    float cy = camera[4]*win[0] + camera[5]*win[1] + camera[6]*win[2] + camera[7];
    float cz = MYTH_get_point_depth(camera, win);
    out[0] = cx / cz;
    out[1] = cy / cz;
    return (out[0] >= 0) && (out[1] >= 0) && (out[0]<input_width) && (out[1]<input_height);
}

__inline__ __device__ void MYTH_unproject_point(float *camloc, float *invKR, int u, int v, float z, float *out) {
    out[0] = camloc[0] + (invKR[0] * (u + 0.5f) + invKR[1] * (v + 0.5f) + invKR[2]) * z;
    out[1] = camloc[1] + (invKR[3] * (u + 0.5f) + invKR[4] * (v + 0.5f) + invKR[5]) * z;
    out[2] = camloc[2] + (invKR[6] * (u + 0.5f) + invKR[7] * (v + 0.5f) + invKR[8]) * z;
}

__device__ static float MYTH_atomicMinf(float* addr, float val)
{
    float old;
    old = (val >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(val))) :
         __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(val)));

    return old;
}

__inline__ __device__ void MYTH_applyWorldT_3d(float *worldt_p, float sx, float sy, float sz, float *out) {
    out[0] = worldt_p[0]*sx + worldt_p[1]*sy + worldt_p[2]*sz + worldt_p[3];
    out[1] = worldt_p[4]*sx + worldt_p[5]*sy + worldt_p[6]*sz + worldt_p[7];
    out[2] = worldt_p[8]*sx + worldt_p[9]*sy + worldt_p[10]*sz+ worldt_p[11];
}

__inline__ __device__ void MYTH_applyWorldT(float *worldt_p, int x, int y, int z, int cube_dimension, float *out) {
    float sx = (x + 0.5f)/cube_dimension, sy = (y + 0.5f)/cube_dimension, sz = (z + 0.5f)/cube_dimension;
    MYTH_applyWorldT_3d(worldt_p, sx, sy, sz, out);
}

#include "InvertCams.cu"
#include "DepthReprojectionCompleteBound.cu"
#include "DepthReprojectionNonzeroCompleteBound.cu"
#include "DepthReprojectionNeighbours.cu"
#include "DepthReprojectionNeighboursAlt.cu"
#include "DepthColorAngleReprojectionNeighbours.cu"
#include "AverageReprojectionPooling.cu"
#include "MaximumReprojectionPooling.cu"
#include "VariationReprojectionPooling.cu"
#include "MinimumAbsReprojectionPooling.cu"

#include "PatchedReprojectionNeighbours.cu"
#include "PatchedReprojectionCompleteBound.cu"
#include "PatchedAverageReprojectionPooling.cu"
#include "PatchedMaximumReprojectionPooling.cu"
#include "PatchedMinimumAbsReprojectionPooling.cu"
#include "bruteforce_cloud_distances.cu"