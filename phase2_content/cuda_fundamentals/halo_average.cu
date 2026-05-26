#include<cuda_runtime_api.h>
#include<stdlib.h>

__global__ void haloAverage(float* input, float* output, int N)
{
    __shared__ float tile[256 + 2];

    int tid = threadIdx.x;
    int gid = tid + blockDim.x * blockIdx.x;

    tile[tid + 1] = (gid < N) ? input[gid] : 0;

    if (tid == 0)
        tile[0] = (blockIdx.x > 0) ? input[gid - 1] : input[0];

    if (tid == blockDim.x - 1)
        tile[blockDim.x + 1] = (gid + 1 < N) ? input[gid + 1] : input[N - 1]; // ✅

    __syncthreads();

    if (gid == 0 || gid == N - 1)
        output[gid] = input[gid];
    else if (gid < N)
        output[gid] = (tile[tid] + tile[tid + 1] + tile[tid + 2]) / 3.0f;  // ✅
}