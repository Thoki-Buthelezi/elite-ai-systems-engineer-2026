#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>

#define TILE_DIM 16

__global__ void matrixTransposeShared(int* input, int* output, int N)
{
    __shared__ int tile[TILE_DIM][TILE_DIM + 1];  // +1 avoids bank conflicts

    int r = blockIdx.y * TILE_DIM + threadIdx.y;
    int c = blockIdx.x * TILE_DIM + threadIdx.x;

    if (r < N && c < N)
        tile[threadIdx.y][threadIdx.x] = input[r * N + c];

    __syncthreads();

    int transposed_r = blockIdx.x * TILE_DIM + threadIdx.y;
    int transposed_c = blockIdx.y * TILE_DIM + threadIdx.x;

    if (transposed_r < N && transposed_c < N)
        output[transposed_r * N + transposed_c] = tile[threadIdx.x][threadIdx.y];
}

void serialTranspose(int* input, int* output, int N)
{
    for (int r = 0; r < N; r++)
        for (int c = 0; c < N; c++)
            output[c * N + r] = input[r * N + c];
}

bool matrixEqual(int* A, int* B, int N)
{
    for (int i = 0; i < N * N; i++)
        if (A[i] != B[i]) return false;
    return true;
}

void printMatrix(int* M, int N)
{
    for (int r = 0; r < 4; r++)
    {
        for (int c = 0; c < N; c++)
            printf("%4d ", M[r * N + c]);
        printf("\n");
    }
}

void initArray(int* arr, int n)
{
    for (int i = 0; i < n; i++)
        arr[i] = rand() % 100;   // keep numbers small so output is readable
}

int main()
{
    srand(0);
    int N = 32;
    size_t size = N * N * sizeof(int);

    int* A       = nullptr;
    int* B       = nullptr;
    int* compare = nullptr;
    int* d_A     = nullptr;
    int* d_B     = nullptr;

    cudaMallocHost(&A,       size);
    cudaMallocHost(&B,       size);
    cudaMallocHost(&compare, size);
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    initArray(A, N * N);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks(N / TILE_DIM, N / TILE_DIM);

    matrixTransposeShared<<<blocks, threads>>>(d_A, d_B, N);
    cudaDeviceSynchronize();

    cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);

    serialTranspose(A, compare, N);

    if (matrixEqual(B, compare, N))
        printf("CPU and GPU match\n\n");
    else
        printf("CPU and GPU do not match\n\n");

    printf("Input (first 4 rows):\n");
    printMatrix(A, N);

    printf("\nTransposed (first 4 rows):\n");
    printMatrix(B, N);

    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(compare);
    cudaFree(d_A);
    cudaFree(d_B);
    return 0;
}