%%writefile test.cu
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>

__global__ void vectorDotProduct(int* A, int* B, int* result, int N)
{
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gid < N)
    {
        int product = A[gid] * B[gid];
        atomicAdd(result, product);
    }
}

int serialDotProduct(int* A, int* B, int N)
{
    int sum = 0;
    for (int i = 0; i < N; i++)
        sum += A[i] * B[i];
    return sum;
}

void initArray(int* arr, int length)
{
    for (int i = 0; i < length; i++)
        arr[i] = rand() % 100;
}

int main()
{
    srand(0);
    int N = 1024;
    size_t size = N * sizeof(int);

    int* A       = nullptr;
    int* B       = nullptr;
    int* h_result = nullptr;

    cudaMallocHost(&A,        size);
    cudaMallocHost(&B,        size);
    cudaMallocHost(&h_result, sizeof(int));

    initArray(A, N);
    initArray(B, N);

    int* d_A      = nullptr;
    int* d_B      = nullptr;
    int* d_result = nullptr;

    cudaMalloc(&d_A,      size);
    cudaMalloc(&d_B,      size);
    cudaMalloc(&d_result, sizeof(int));
    cudaMemset(d_result,  0, sizeof(int));  // zero before kernel runs

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    vectorDotProduct<<<blocks, threads>>>(d_A, d_B, d_result, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    int cpuResult = serialDotProduct(A, B, N);

    if (*h_result == cpuResult)
        printf("GPU and CPU match\n");
    else
        printf("GPU and CPU do not match\n");

    printf("Dot product: %d\n", *h_result);

    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(h_result);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_result);

    return 0;
}