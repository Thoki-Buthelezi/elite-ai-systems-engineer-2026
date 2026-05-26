something is wrong with my code:
#include<cuda_runtime_api.h>
#include<stdlib.h>
#include<stdio.h>

__global__ void matrixMul(int* A, int* B, int* C, int N)
{
    int r = threadIdx.y + blockDim.y * blockIdx.y;
    int c = threadIdx.x + blockDim.x * blockIdx.x;

    if (r < N && c < N)
    {
        int sum = 0;
        for (int i = 0; i < N; i++)
        {
            sum += A[r * N + i] * B[i * N + c];
        }
        C[r * N + c] = sum;
    }
}

void serialMatrixMul(int* A, int* B, int* C, int N)
{
    for (int r = 0; r < N; r++)
    {
        for (int c = 0; c < N; c++)
        {
            int sum = 0;
            for (int k = 0; k < N; k++)
            {
                sum += A[r * N +k] * B[k * N + c];
            }
            C[r * N + c] = sum;
        }
    }
}

bool matrixEqual(int* A, int* B, int N)
{
    for (int i = 0; i < N * N; i++)
    {
        if (A[i] != B[i])
            return false;
    }
    return true;
}

void display(int* A, int N)
{
    for (int r = 0; r < 4; r++)
    {
        for (int c = 0; c < 4; c++)
        {
            printf("%4d ", A[r * N + c]);
        }
        printf("\n");
    }
    
}

void initArray(int* A, int N)
{
    for (int i = 0; i < N; i++)
    {
        A[i] = rand() % 10;
    }
}

int main()
{
    srand(0);
    int N = 16;
    size_t size = N * N * sizeof(int);
    dim3 threads(16, 16);
    dim3 blocks(1, 1);   // 16x16 matrix fits in one 16x16 block

    int* A = nullptr;
    int* B = nullptr;
    int* C = nullptr;
    int* compare = nullptr;

    cudaMallocHost(&A, size);
    cudaMallocHost(&B, size);
    cudaMallocHost(&C, size);
    cudaMallocHost(&compare, size);

    initArray(A, N * N);
    initArray(B, N * N);

    int* d_A = nullptr;
    int* d_B = nullptr;
    int* d_C = nullptr;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size);
    
    matrixMul<<<blocks,threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    
    serialMatrixMul(A, B, compare, N);
    
    if (matrixEqual(C, compare, N))
    {
        printf("GPU and CPU match\n");
    }
    else
    {
        printf("GPU and CPU do not match\n");
    }
    
    display(A, N);
    display(B, N);
    display(C, N);
    display(compare, N);
    
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    cudaFreeHost(compare);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}
