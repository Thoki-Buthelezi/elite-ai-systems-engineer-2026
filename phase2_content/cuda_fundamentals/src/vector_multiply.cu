#include <cuda_runtime_api.h>
#include<device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>

__global__ void vectorMul(int* A, int* B, int* C, int vectorLength)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < vectorLength)
    {
        C[tid] = A[tid] * B[tid];
    }
}

void serialVectorLength(int* A, int* B, int* C, int vectorLength)
{
    for (int i = 0; i < vectorLength; i++)
    {
        C[i] = A[i] * B[i];
    }
}

bool vectorEqual(int* A, int* B, int vectorLength)
{
    for (int i = 0; i < vectorLength; i++)
    {
        if (A[i] != B[i])
        {
            return false;
        }
    }
    return true;
}

void displayVector(int* A, int* B, int* C)
{
    for (int i = 0; i < 8; i++)
    {
        printf("%d * %d = %d\n", A[i], B[i], C[i]);
    }
}

void initArray(int* arr, int vectorLength)
{
    for (int i = 0; i < vectorLength; i++)
    {
        arr[i] = rand();
    }
}

int main()
{
    srand(0);
    int vectorLength = 1024;
    size_t size = vectorLength * sizeof(int);

    int* A = nullptr;
    int* B = nullptr;
    int* C = nullptr;
    int* compare = nullptr;

    cudaMallocHost(&A, size);
    cudaMallocHost(&B, size);
    cudaMallocHost(&C, size);
    cudaMallocHost(&compare, size);

    initArray(A, vectorLength);
    initArray(B, vectorLength);

    int* d_A = nullptr;
    int* d_B = nullptr;
    int* d_C = nullptr;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (vectorLength + threads - 1) / threads;

    vectorMul<<<blocks, threads>>>(d_A, d_B, d_C, vectorLength);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    serialVectorLength(A, B, compare, vectorLength);

    if (vectorEqual(C, compare, vectorLength))
        printf("GPU and CPU match\n");
    else
        printf("GPU and CPU do not match\n");

    displayVector(A, B, C);

    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    cudaFreeHost(compare);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}