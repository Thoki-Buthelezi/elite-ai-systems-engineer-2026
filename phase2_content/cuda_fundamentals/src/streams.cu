#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>

__global__ void doubleIt(int* input, int* output, int N)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < N)
        output[gid] = input[gid] * 2;
}

void initArray(int* A, int N)
{
    for (int i = 0; i < N; i++)
        A[i] = rand() % 15;
}

int main()
{
    srand(0);
    int N      = 1024;
    int chunkN = N / 2;                      // 512 elements per chunk
    size_t chunkSize = chunkN * sizeof(int); // bytes per chunk

    int threads = 256;
    int blocks  = (chunkN + threads - 1) / threads;  // blocks per chunk

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // pinned CPU memory — required for async transfers
    int* A = nullptr;
    int* B = nullptr;
    int* C = nullptr;
    int* D = nullptr;

    cudaMallocHost(&A, chunkSize);
    cudaMallocHost(&B, chunkSize);
    cudaMallocHost(&C, chunkSize);
    cudaMallocHost(&D, chunkSize);

    initArray(A, chunkN);   //  element count not byte count
    initArray(B, chunkN);

    // GPU memory — allocated async in their respective streams
    int* d_A = nullptr;
    int* d_B = nullptr;
    int* d_C = nullptr;
    int* d_D = nullptr;

    cudaMallocAsync(&d_A, chunkSize, stream1);  //  stream1
    cudaMallocAsync(&d_C, chunkSize, stream1);  //  stream1
    cudaMallocAsync(&d_B, chunkSize, stream2);  //  stream2
    cudaMallocAsync(&d_D, chunkSize, stream2);  //  stream2

    // stream 1 pipeline — chunk A
    cudaMemcpyAsync(d_A, A, chunkSize, cudaMemcpyHostToDevice, stream1);
    doubleIt<<<blocks, threads, 0, stream1>>>(d_A, d_C, chunkN);
    cudaMemcpyAsync(C, d_C, chunkSize, cudaMemcpyDeviceToHost, stream1);  //  dst, src

    // stream 2 pipeline — chunk B (overlaps with stream 1)
    cudaMemcpyAsync(d_B, B, chunkSize, cudaMemcpyHostToDevice, stream2);
    doubleIt<<<blocks, threads, 0, stream2>>>(d_B, d_D, chunkN);
    cudaMemcpyAsync(D, d_D, chunkSize, cudaMemcpyDeviceToHost, stream2);  //  dst, src

    // wait for both streams to finish
    cudaDeviceSynchronize();

    // print first 4 elements of each chunk
    printf("Chunk A input:  %d, %d, %d, %d\n", A[0], A[1], A[2], A[3]);
    printf("Chunk A output: %d, %d, %d, %d\n", C[0], C[1], C[2], C[3]);
    printf("Chunk B input:  %d, %d, %d, %d\n", B[0], B[1], B[2], B[3]);
    printf("Chunk B output: %d, %d, %d, %d\n", D[0], D[1], D[2], D[3]);

    // verify
    bool match = true;
    for (int i = 0; i < chunkN; i++)
    {
        if (C[i] != A[i] * 2 || D[i] != B[i] * 2)
        {
            match = false;
            break;
        }
    }
    printf(match ? "Correct\n" : "Wrong\n");

    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    cudaFreeHost(D);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}