#include <stdio.h>
#include <iostream>
#include <cuda.h>

//Device Code
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
      C[i] = A[i] + B[i];
}

//Host Code
int main(int argc , char* argv[])
{
    int N = 5;
    size_t size = N * sizeof(float);

    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize input vectors
    int i;
    for (i = 0; i < N; i++){
        h_A[i] = i + 1;
        h_B[i] = i + 1;    
    }

    // Allocate vectors in device memory
    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);
    
    // Copy data source to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Invoke Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock -1) / threadsPerBlock;

    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    for(i=0; i < size; i++){
        printf("%f\n", h_C[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
}