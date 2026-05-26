#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>

__global__ void addPrefix(int* input, int* output, int vectorLength)
{
    __shared__ int tile[256];

    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;

    // Step 1: load
    if (gid < vectorLength)
        tile[tid] = input[gid];
    else
        tile[tid] = 0;

    __syncthreads();

    // Step 2: scan
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int val = 0;
        if (tid >= stride)
            val = tile[tid - stride];
        __syncthreads();
        tile[tid] += val;
        __syncthreads();
    }

    // Step 3: write back
    if (gid < vectorLength)
        output[gid] = tile[tid];
}

void serialAddPrefix(int* A, int* B, int vectorLength)
{
    int sum = 0;
    for (int i = 0; i < vectorLength; i++)
    {
        sum += A[i];
        B[i] = sum;
    }
}

void display(int* A)
{
    for (int i = 0; i < 8; i++)
        printf("%4d ", A[i]);
    printf("\n");
}

bool equalVector(int* A, int* B, int vectorLength)
{
    for (int i = 0; i < vectorLength; i++)
        if (A[i] != B[i]) return false;
    return true;
}

void initArray(int* A, int vectorLength)
{
    for (int i = 0; i < vectorLength; i++)
        A[i] = rand() % 10;   // small numbers so prefix sums are readable
}

int main()
{
    srand(0);
    int vectorLength = 256;   // single block only
    size_t size = vectorLength * sizeof(int);

    int* A = nullptr;
    int* B = nullptr;
    int* C = nullptr;

    cudaMallocHost(&A, size);
    cudaMallocHost(&B, size);
    cudaMallocHost(&C, size);

    initArray(A, vectorLength);

    int* d_A = nullptr;
    int* d_B = nullptr;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = 1;

    addPrefix<<<blocks, threads>>>(d_A, d_B, vectorLength);
    cudaDeviceSynchronize();

    cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);

    serialAddPrefix(A, C, vectorLength);

    if (equalVector(C, B, vectorLength))
        printf("GPU and CPU match\n");
    else
        printf("GPU and CPU do not match\n");

    printf("Input:      "); display(A);
    printf("GPU output: "); display(B);
    printf("CPU output: "); display(C);

    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}