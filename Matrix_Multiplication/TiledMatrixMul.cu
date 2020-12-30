/*
 *  file name: matrix.cu
 *
 *  CPE810A: Homework 1, matrix * matrix by using shared memory
 *  
 *  Yupeng Cao, 10454637
 *  
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define BLOCK_SIZE 16
#define TILE_SIZE 16


/*
*********************************************************************
function name: gpu_matrix_mult

parameters: 
            &a GPU device pointer to a m X n matrix (A)
            &b GPU device pointer to a n X k matrix (B)
            &c GPU device output purpose pointer to a m X k matrix (C) 
            to store the result

*********************************************************************
*/
__global__ void gpu_matrix_mult(float *a, float *b, float *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 

/*
*********************************************************************
function name: shared_matrix_mult

parameters: 
            &a GPU device pointer to a m X n matrix (A)
            &b GPU device pointer to a n X k matrix (B)
            &c GPU device output purpose pointer to a m X k matrix (C) 
            to store the result

Using Shared Memory

*********************************************************************
*/
__global__ void shared_matrix_mult(float* A, float* B, float* C, int m, int n, int k)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float CValue = 0.0;

    for (int t = 0; t * TILE_SIZE < n; ++t) 
    {
        if (row < m && t * TILE_SIZE + threadIdx.x < n)
            As[threadIdx.y][threadIdx.x] = A[row * n + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0;

        if (col < k && t * TILE_SIZE + threadIdx.y < n)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * k + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0;
        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) 
        {
            CValue += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < m && col < k)
        C[row * k + col] = CValue;
}


/*
*********************************************************************
function name: cpu_matrix_mult

description: dot product of two matrix in CPU, 
             for validating GPU results

*********************************************************************
*/
void cpu_matrix_mult(float *h_a, float *h_b, float *h_result, int m, int n, int k) {
    for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < k; ++j) 
        {
            float tmp = 0.0;
            for (int h = 0; h < n; ++h) 
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

/*
*********************************************************************
function name: printMatrix

description: Print calculation results
             for visualize GPU results

Note:  if the matrix size larger than 10,
       this function will not execute

*********************************************************************
*/
void printMatrix(float* result_matrix, int row, int col) {
    if (row > 10 || col > 10) return;
    printf("\n");
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%f\t", result_matrix[i*col + j]);
        }
        printf("\n");
    }
    return;
}

/*
*********************************************************************
function name: main

description: test and compare

*********************************************************************
*/
int main(int argc, char *argv[])
{   

    // input check
    if ( argc != 4)
    {
        printf("Error input Parameter \n");
        printf("Please input matrix size \n");
        printf("Matrix A: m by n; Matrix B: n by k \n");
        return 0;
    }

    /* 
        Matrix A: m * n
        Matrix B: n * k
    */
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    srand(1000);
    //printf("please type in m n and k\n");
    //scanf("%d %d %d", &m, &n, &k);

    // allocate memory in host, h_cc is used to store CPU result
    float *h_a, *h_b, *h_c, *h_cc;
    cudaMallocHost((void **) &h_a, sizeof(int)*m*n);
    cudaMallocHost((void **) &h_b, sizeof(int)*n*k);
    cudaMallocHost((void **) &h_c, sizeof(int)*m*k);
    cudaMallocHost((void **) &h_cc, sizeof(int)*m*k);

    // random initialize matrix A
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = rand() % 1024;
        }
    }

    // random initialize matrix B
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            h_b[i * k + j] = rand() % 1024;
        }
    }

    // Allocate memory space on the device 
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, sizeof(int)*m*n);
    cudaMalloc((void **) &d_b, sizeof(int)*n*k);
    cudaMalloc((void **) &d_c, sizeof(int)*m*k);

    // copy matrix A and B from host to device memory
    cudaMemcpy(d_a, h_a, sizeof(int)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int)*n*k, cudaMemcpyHostToDevice);

    // assign Grid and Block size
    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    // count the execution time
    float shared_gpu_time_ms, gpu_elapsed_time_ms, cpu_elapsed_time_ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // start to count execution time of GPU without using Shared Memory version
    cudaEventRecord(start, 0);
    // Launch kernel 
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);    
    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU without shared memory: %f ms.\n\n", m, n, n, k, gpu_elapsed_time_ms);


    // start to count execution time of GPU with using Shared Memory version
    cudaEventRecord(start, 0);
    // Launch kernel
    shared_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);
    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // compute time elapse on GPU computing
    cudaEventElapsedTime(&shared_gpu_time_ms, start, stop);    
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU with shared memory: %f ms.\n\n", m, n, n, k, shared_gpu_time_ms);


    // Transefr results from device to host 
    cudaMemcpy(h_c, d_c, sizeof(int)*m*k, cudaMemcpyDeviceToHost);


    // start the CPU version
    cudaEventRecord(start, 0);
    cpu_matrix_mult(h_a, h_b, h_cc, m, n, k);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on CPU: %f ms.\n\n", m, n, n, k, cpu_elapsed_time_ms);


    // validate results computed by GPU with shared memory
    int all_ok = 1;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            if(h_cc[i*k + j] != h_c[i*k + j])
            {
                all_ok = 0;
            }
        }
    }

    // compute speedup ratio
    // cpu time / shared_memory time
    if(all_ok)
    {
        printf("all results are correct!, speedup = %f\n", cpu_elapsed_time_ms / shared_gpu_time_ms);
    }
    else
    {
        printf("incorrect results\n");
    }

    printMatrix(h_c, m, k);

    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFreeHost(h_cc);
    return 0;
}
