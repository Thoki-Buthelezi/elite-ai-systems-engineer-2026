#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>

__global__ void vectorReverse(int* A, int* B, int vectorLength)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < vectorLength)
    {
        B[tid] = A[vectorLength - 1 - tid];
    }
}

void serialVectorReverse(int* A, int* B, int vectorLength)
{
    int i = vectorLength;
    int k = 0;
    while (i > 0)
    {
        i--;
        B[k] = A[i];
        k++;
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

void displayVector(int* A, int* B)
{
    for (int i = 0; i < 8; i++)
    {
        printf("%d, ", A[i]);
    }
    printf("\n");
    
    for (int i = 0; i < 8; i++)
    {
        printf("%d, ", B[i]);
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
    int vectorLength = 1024;
    int* A = nullptr;
    int* B = nullptr;
    int* compare = nullptr;
    
    size_t size = vectorLength * sizeof(int);
    
    cudaMallocHost(&A, size);
    cudaMallocHost(&B, size);
    cudaMallocHost(&compare, size);
    
    initArray(A, vectorLength);
    
    int* d_A = nullptr;
    int* d_B = nullptr;
    
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (vectorLength + threads - 1) / threads;
    
    vectorReverse<<<blocks,threads>>>(d_A,d_B,vectorLength);
    cudaDeviceSynchronize();
    
    cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);
    
    serialVectorReverse(A, compare, vectorLength);
    
    if (vectorEqual(B, compare, vectorLength))
    {
        printf("CPU and GPU match\n");
    }
    else
    {
        printf("CPU and GPU not match\n");
    }
    
    displayVector(compare, A);
    
    
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(compare);
    cudaFree(d_A);
    cudaFree(d_B);

}
