#include<cuda_runtime_api.h>
#include<stdlib.h>
#include<stdio.h>

__global__ void calcSum(int* input, int* output, int N)
{
    __shared__ int tile[256];
    
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    
    tile[tid] = (gid < N) ? input[gid] : 0;
    __syncthreads(); //wait until each threads finishes loading from the global memory
    
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
        {
            if (tid < stride)
            {
                tile[tid] += tile[tid + stride];
            }
            __syncthreads(); //wait until each thread performs its operation
        }
        if (tid == 0)
            output[blockIdx.x] = tile[0];
}
int serialCalcSum(int* A, int N)
{
    int sum = 0;
    for (int r = 0; r < N; r++)
    {
        sum += A[r];
    }
    return sum;
}
void display(int* A)
{
    for (int i = 0; i < 8; i++)
    {
        printf("%4d " , A[i]);
    }
    printf("\n");
}
void initArray(int* A, int N)
{
    for (int i = 0; i < N; i++)
    {
        A[i] = rand() % 100;
    }
}

int main()
{
    srand(0);
    int N = 1024;
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;  // = 4

    size_t inputSize  = N * sizeof(int);
    size_t outputSize = blocks * sizeof(int);    // only 4 ints

    int* A = nullptr;
    int* B = nullptr;

    cudaMallocHost(&A, inputSize);
    cudaMallocHost(&B, outputSize);   // only 4 elements

    initArray(A, N);

    int* d_A = nullptr;
    int* d_B = nullptr;

    cudaMalloc(&d_A, inputSize);
    cudaMalloc(&d_B, outputSize);     // only 4 elements

    cudaMemcpy(d_A, A, inputSize, cudaMemcpyHostToDevice);

    calcSum<<<blocks, threads>>>(d_A, d_B, N);
    cudaDeviceSynchronize();

    cudaMemcpy(B, d_B, outputSize, cudaMemcpyDeviceToHost);

    // sum the 4 block results
    int gpuSum = 0;
    for (int i = 0; i < blocks; i++)
        gpuSum += B[i];

    int cpuSum = serialCalcSum(A, N);

    if (gpuSum == cpuSum)
        printf("GPU and CPU match\n");
    else
        printf("GPU and CPU do not match\n");

    display(A);
    printf("sum: %d\n", gpuSum);

    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}