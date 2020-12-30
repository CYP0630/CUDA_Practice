/*
 *  file name: histogram_atomic.cu
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
    Define Error Checking methods

    cudaSafeCall:   Check data allocate
    cudaCheckError: Check kernel function execution
*********************************************************************
*/
#define CUDA_ERROR_CHECK
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}


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


// Print histogram for small bin size
void print_bin(int* vector, int Size) {
    if (Size < 20){
        for (int i = 0; i < Size; ++i) {
            printf("%d", vector[i]);
            printf(" ");
         }
    }
    printf("\n\n");
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
    CudaSafeCall(cudaMallocHost((void **) &vector, sizeof(int)*Size));
    srand((unsigned)time(NULL));  // make sure the number in vector >= 0 
    for (int i = 0; i < Size; ++i){
        vector[i] = rand() % 1024;
    }

    // allocate memory on host for saving results
    int* hist_cpu = (int*)calloc(Size, sizeof(int));
    int* hist_gpu = (int*)calloc(Size, sizeof(int));

    // allocate memory on device
    int *d_vec, *d_hist;
    CudaSafeCall(cudaMalloc((void **)&d_vec, sizeof(int)*Size));
    CudaSafeCall(cudaMalloc((void **)&d_hist, sizeof(int)*bin));

    // transfer vector from host to device
    CudaSafeCall(cudaMemcpy(d_vec, vector, sizeof(int)*Size, cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemset(d_hist, 0, bin));

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
    CudaCheckError();

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    printf("Counting histogram by using GPU: %f ms.\n", gpu_time_ms);

    cudaMemcpy(hist_gpu, d_hist, sizeof(int)*bin, cudaMemcpyDeviceToHost);
    printf("Histogram: ");
    print_bin(hist_gpu, bin);


    // count histogram by using CPU
    cudaEventRecord(start, 0);
    hist_CPU(vector, hist_cpu, bin, Size);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_time_ms, start, stop);
    printf("Counting histogram by using CPU: %f ms.\n", cpu_time_ms);

    printf("Histogram: ");
    print_bin(hist_cpu, bin);


    // validate results computed by GPU with shared memory
    int all_ok = 1;
    for (int i = 0; i < bin; ++i)
        {
            if(hist_gpu[i] != hist_cpu[i])
            {
                all_ok = 0;
            }
    }

    // compute speedup ratio
    // cpu time / gpu time
    if(all_ok)
    {
        printf("all results are correct!, speedup = %f\n", cpu_time_ms / gpu_time_ms);
    }
    else
    {
        printf("incorrect results\n");
    }


    // free memory
    cudaFree(d_vec);
    cudaFree(d_hist);
    cudaFreeHost(vector);
    cudaFreeHost(hist_cpu);
    cudaFreeHost(hist_gpu);

    return 0;

}
