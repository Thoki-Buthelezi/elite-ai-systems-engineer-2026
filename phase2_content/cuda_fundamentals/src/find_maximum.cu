#include <cuda_runtime_api.h>
#include <climits>
#include <stdlib.h>
#include <stdio.h>
__global__ void vectorMax(int* input, int* output, int vectorLength)
{
    //all threads in this block share this memory
    __shared__ int tile[256];
    
    //local thread id
    int tid = threadIdx.x;
    //global thread id
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    //check upper bound 
    if (gid < vectorLength)
    {
        //load from the global input to the shared memory
        tile[tid] = input[gid];
    }
    else
    {
        tile[tid] = INT_MIN;
    }
    __syncthreads();
    
    //since all threads in the block share memory, only one must do the work
    if (tid == 0)
    {
        int max = tile[0];
        for (int i = 0; i < blockDim.x; i++)
        {
            if (max < tile[i])
            {
                max = tile[i];
            }
        }
        output[blockIdx.x] = max;
    }
}
int serialVectorMax(int* A, int vectorLength)
{
    int max = A[0];
    for (int i = 0; i < vectorLength; i++)
    {
        if (max < A[i])
        {
            max = A[i];
        }
    }
    return max;
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
void displayVector(int* A, int* B)
{
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
    int threads = 256;
    int blocks = (vectorLength + threads - 1) / threads;  // = 4

    size_t inputSize  = vectorLength * sizeof(int);
    size_t outputSize = blocks * sizeof(int);             // only 4 ints needed

    int* A = nullptr;
    int* B = nullptr;                // will hold 4 block maxes
    int* d_A = nullptr;
    int* d_B = nullptr;

    cudaMallocHost(&A, inputSize);
    cudaMallocHost(&B, outputSize);  // only 4 elements
    initArray(A, vectorLength);

    cudaMalloc(&d_A, inputSize);
    cudaMalloc(&d_B, outputSize);    // only 4 elements on GPU too

    cudaMemcpy(d_A, A, inputSize, cudaMemcpyHostToDevice);

    vectorMax<<<blocks, threads>>>(d_A, d_B, vectorLength);
    cudaDeviceSynchronize();

    cudaMemcpy(B, d_B, outputSize, cudaMemcpyDeviceToHost);

    // CPU finds global max from the 4 block results
    int gpuMax = B[0];
    for (int i = 1; i < blocks; i++)
        if (B[i] > gpuMax)
            gpuMax = B[i];

    // Serial CPU max for comparison
    int cpuMax = serialVectorMax(A, vectorLength);

    if (gpuMax == cpuMax)
    {
        printf("CPU and GPU match\n");
        printf("Max: %d\n", gpuMax);
    }
    else
    {
        printf("CPU and GPU do not match\n");
        printf("CPU max: %d, GPU max: %d\n", cpuMax, gpuMax);
    }

    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFree(d_A);
    cudaFree(d_B);
    return 0;
}