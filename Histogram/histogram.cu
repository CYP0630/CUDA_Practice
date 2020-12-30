/*
 *  file name: histogram.cu
 *
 *  CPE810A: Homework 2: Implement a histogram routine using atomic operations and shared memory in
            CUDA.
 *  
 *  Yupeng Cao, 10454637
 *  
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>


#define BLOCK_SIZE 256

/*
*********************************************************************
function name: hist_GPU

parameters:
            vector:   input vector data on device
            hist_cpu: save results on device
            bin
            Size

Note: count histogram by using GPU
*********************************************************************
*/
__global__ void hist_GPU(int* d_vec, int* d_hist, int bin, int Size) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int bin_range = 1024 / bin;

    extern __shared__ int histo_s[];
    for (unsigned int binIdx = threadIdx.x; binIdx < bin; binIdx += blockDim.x) {
        histo_s[binIdx] = 0;
    }
    __syncthreads();

    for (unsigned int i = tid; i < Size; i += blockDim.x * gridDim.x) {
        atomicAdd(&(histo_s[d_vec[i] / bin_range]), 1);
    }
    __syncthreads();

    for (unsigned int binIdx = threadIdx.x; binIdx < bin; binIdx += blockDim.x) {
        atomicAdd(&(d_hist[binIdx]), histo_s[binIdx]);
    }
}


/*
*********************************************************************
function name: hist_CPU

parameters:
            vector:   input vector data
            hist_cpu: save results 
            bin
            Size

Note: count histogram by using CPU
*********************************************************************
*/
void hist_CPU(int* vector, int* hist_cpu, int bin, int Size){
    
    int bin_range = 1024 / bin;
    for (int i = 0; i < Size; ++i){
        ++hist_cpu[vector[i] / bin_range];
    }
    return;
}


/*
*********************************************************************
function name: check_input

parameters:
            binNum: Input Bin
            vecNum: Data (vector) Size

Note: if binNum isn't 2^N or vecNum < 0, input is invalid.
*********************************************************************
*/
int check_input(int binNum, int vecNum){

    if ((binNum & (binNum - 1)) != 0){
        printf("Invalid bin number \n");
        printf("bin must be 2^n \n");
        return -1;
    }

    if (vecNum < 0){
        printf("Invalid vector size \n");
        printf("vector size must be >= 0 \n");
        return -1;
    }

    return 1;
}


/*
*********************************************************************
Main Function
*********************************************************************
*/
int main(int argc, char *argv[])
{   

    // input parameter and data check
    if ( argc != 4 )
    {
        printf("Error input Parameter \n");
        printf("Please input BinNum and VecDim \n");
        return 0;
    }

    if (argc == 4 && (strcmp(argv[1], "-i") == 0)){
        printf("Input Data\n");
    }else{
        printf("Please Follow Format to Run Program: ./execute_file -i binNum vecNum\n");
        return -1;
    }
    
    int bin = atoi(argv[2]);
    int Size = atoi(argv[3]);

    if (check_input(bin, Size) == 1){
        printf("Input is Valid \n\n");   
    }else{
        return -1;
    }
    

    
    // initialize vector
    int *vector;
    cudaMallocHost((void **) &vector, sizeof(int)*Size);
    srand((unsigned)time(NULL));  // make sure the number in vector >= 0 
    for (int i = 0; i < Size; ++i){
        vector[i] = rand() % 1024;
    }

    // allocate memory on host for saving results
    int* hist_cpu = (int*)calloc(Size, sizeof(int));
    int* hist_gpu = (int*)calloc(Size, sizeof(int));

    // allocate memory on device
    int *d_vec, *d_hist;
    cudaMalloc((void **)&d_vec, sizeof(int)*Size);
    cudaMalloc((void **)&d_hist, sizeof(int)*bin);

    // transfer vector from host to device
    cudaMemcpy(d_vec, vector, sizeof(int)*Size, cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, bin);

    // prepare for recording the execution time
    float gpu_time_ms, cpu_time_ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // count histogram by using GPU 
    cudaEventRecord(start, 0);
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(128);
    
    hist_GPU<<< dimGrid, dimBlock, sizeof(int)*bin>>>(d_vec, d_hist, bin, Size);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    printf("Counting histogram by using GPU: %f ms.\n", gpu_time_ms);

    cudaMemcpy(hist_gpu, d_hist, sizeof(int)*bin, cudaMemcpyDeviceToHost);


    // count histogram by using CPU
    cudaEventRecord(start, 0);
    hist_CPU(vector, hist_cpu, bin, Size);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_time_ms, start, stop);
    printf("Counting histogram by using CPU: %f ms.\n", cpu_time_ms);


    // validate results computed by GPU with shared memory
    int all_ok = 1;
    for (int i = 0; i < bin; ++i)
        {
            if(hist_gpu[i] != hist_cpu[i])
            {
                all_ok = 0;
            }
    }

    if (all_ok == 1){
        printf("all results are correct!\n");
    }else{
        printf("Wrong Error!\n");
    }
    
   

    // free memory
    cudaFree(d_vec);
    cudaFree(d_hist);
    cudaFreeHost(vector);
    cudaFreeHost(hist_cpu);
    cudaFreeHost(hist_gpu);

    return 0;

}
