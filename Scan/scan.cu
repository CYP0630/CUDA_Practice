/*
 *  file name: scan.cu
 *
 *  CPE810A: Homework 4: Scan List
 *  
 *  Yupeng Cao, 10454637
 *  
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <string.h>

 
#define BLOCK_SIZE 1024


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
    scanGPU

    1) generating scanned blocks 
    2) generating scanned aux array that has the scanned block sums. 
    3) generating final results

Reference: https://github.com/ShekharShiroor/Parallel_List_Scan_Using_CUDA_and_C
*********************************************************************
*/
__global__
static void scanGPU(unsigned int* d_list, unsigned int* flags, unsigned int* AuxArray, unsigned int* AuxScannedArray, int dim) {
    extern __shared__ unsigned int I;
    
    // Scan Segment 
    if (threadIdx.x == 0) {
         I = atomicAdd(&AuxScannedArray[0], 1);
    }
    __syncthreads();

    extern __shared__ unsigned int scanBlockSum[2 * BLOCK_SIZE];
    unsigned int t = threadIdx.x;
    unsigned int s = 2 * I * blockDim.x;

    if (s + t < dim) scanBlockSum[t] = d_list[s + t];
    if (s + t + blockDim.x < dim) scanBlockSum[blockDim.x + t] = d_list[s + blockDim.x + t];
    __syncthreads();

    // Scan for different block
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        int idx = (threadIdx.x + 1) * stride * 2 - 1;
        if (idx < 2 * blockDim.x) scanBlockSum[idx] += scanBlockSum[idx - stride];
        __syncthreads();
    }

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        __syncthreads();
        int idx = (threadIdx.x + 1) * stride * 2 - 1;
        if (idx + stride < 2 * blockDim.x) {
            scanBlockSum[idx + stride] += scanBlockSum[idx];
        }
    }
    __syncthreads();


    // Sum for each block.
    if (threadIdx.x == 0) {
        if (I == 0) {
            AuxArray[I] = scanBlockSum[2*blockDim.x - 1];
            atomicAdd(&flags[I], 1);
        }
        else {
            while (atomicAdd(&flags[I - 1], 0) == 0) { ; }
            AuxArray[I] = AuxArray[I - 1] + scanBlockSum[2 * blockDim.x - 1];
            __threadfence();
            atomicAdd(&flags[I], 1);
        }
    }
    __syncthreads();

    // all values of the scanned blocks
    if (I > 0) {
        scanBlockSum[t] += AuxArray[I - 1];
        scanBlockSum[t + blockDim.x] += AuxArray[I - 1];
    }
    __syncthreads();

    if (s + t < dim)  d_list[s + t] = scanBlockSum[t];
    if (s + t + blockDim.x < dim) d_list[s + blockDim.x + t] = scanBlockSum[blockDim.x + t];
}

/*
*********************************************************************
    scanCPU

    Naive Pre-Sum by using CPU
*********************************************************************
*/
void scanCPU(unsigned int* list, unsigned int* sum, int dim) {
    unsigned int res = 0;
    for (int i = 0; i < dim; ++i) {
        res += list[i];
        sum[i] = res;
    }
    return;
}

/*
*********************************************************************
    printList

    Print the first 10 elements
*********************************************************************
*/
void printList(unsigned int* list, int size) {
    for (int i = 0; i < size && i < 10; ++i) {
        if (i != 0) printf(",");
        printf("%4u", list[i]);
    }
    printf("\n");
    if (size > 10) printf(",... \n");
    return;
}


int check_input(int dim){

    if (dim < 0){
        printf("Invalid list size \n");
        printf("list size must be >= 0 \n");
        return -1;
    }
  
    return 1;
  }


/*
*********************************************************************
    main

    1) GPU implementation
    2) CPU implementation
    3) Check Results
    4) Calculate SpeedUp Ratio
*********************************************************************
*/
int main(int argc, char** argv)
{
	// Check Input
    if ( argc != 3 )
    {
      printf("Error input Parameter \n");
      printf("Please input dim for input list \n");
      printf("Example: ./execute_file -i dim \n");
      return 0;
    }

    if (argc == 3 && (strcmp(argv[1], "-i") == 0)){
      printf("Input Data\n");
    }else{
      printf("Please Follow Format to Run Program: ./execute_file -i dim\n");
      return -1;
    }

    const int Dim = atoi(argv[2]);

    if (check_input(Dim) == 1){
        printf("Input is Valid \n\n");   
      }else{
        return -1;
      }
    

    // Initialize list data
    unsigned int* list = (unsigned int*)malloc(Dim * sizeof(unsigned int));
    srand((unsigned)time(NULL));
    for (int i = 0; i < Dim; i++)
    {
        list[i] = rand() % 16;
    }    
    printf("Initialized List: \n");
    printList(list, Dim);    

    // allocate memory on host for saving results
    unsigned int* scan_CPU = (unsigned int*)malloc(Dim * sizeof(unsigned int));
    unsigned int* scan_GPU = (unsigned int*)malloc(Dim * sizeof(unsigned int));

    // allocate memory on device for variable
    unsigned int* d_list, *d_flags, *d_AuxArray, *d_AuxScannedArray;
    CudaSafeCall(cudaMalloc((unsigned int**)&d_list, Dim * sizeof(unsigned int)));
    CudaSafeCall(cudaMalloc((unsigned int**)&d_AuxScannedArray, sizeof(unsigned int)));
    CudaSafeCall(cudaMalloc((unsigned int**)&d_flags, (int)ceil(1.0 * Dim / BLOCK_SIZE) * sizeof(unsigned int)));
    CudaSafeCall(cudaMalloc((unsigned int**)&d_AuxArray, (int)ceil(1.0 * Dim / BLOCK_SIZE) * sizeof(unsigned int)));

    CudaSafeCall(cudaMemset(d_flags, 0, (int)ceil(1.0 * Dim / BLOCK_SIZE) * sizeof(unsigned int)));
    CudaSafeCall(cudaMemset(d_AuxScannedArray, 0, sizeof(unsigned int)));
    CudaSafeCall(cudaMemcpy(d_list, list, Dim * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // prepare for recording the execution time
    float gpu_time_ms, cpu_time_ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((int)ceil(1.0 * Dim / dimBlock.x));

    // count pre-sum by using GPU 
    cudaEventRecord(start, 0);
    scanGPU<<<dimGrid, dimBlock, (2 * BLOCK_SIZE + 1) * sizeof(unsigned int)>>>(d_list, d_flags, d_AuxArray, d_AuxScannedArray, Dim);
    CudaCheckError();  
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    CudaSafeCall(cudaMemcpy(scan_GPU, d_list, Dim * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    printf("GPU Results: \n");
    printList(scan_GPU, Dim);

    // count pre-sum by using CPU
    cudaEventRecord(start, 0);
    scanCPU(list, scan_CPU, Dim);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_time_ms, start, stop);
    printf("CPU Results: \n");
    printList(scan_CPU, Dim);

    // validate CPU results and GPU results 
    int all_ok = 1;
    for (int i = 0; i < Dim; ++i)
        {
            if(scan_GPU[i] != scan_CPU[i])
            {
                all_ok = 0;
            }
    }

    // compute speedup ratio
    // cpu time / gpu time
    if(all_ok)
    {
        printf("all results are correct!\n");
    }
    else
    {
        printf("incorrect results\n");
    }

    printf("Counting histogram by using GPU: %f ms.\n", gpu_time_ms);
    printf("Counting histogram by using CPU: %f ms.\n", cpu_time_ms);
    printf("SpeedUp = %f\n", cpu_time_ms / gpu_time_ms);

    printf("Throughput = %.4f Operations/s \n",  (Dim / gpu_time_ms / 1000));

    // free memory
    cudaFreeHost(list);
    cudaFreeHost(scan_CPU);
    cudaFreeHost(scan_GPU);
    cudaFree(d_list);
    cudaFree(d_AuxArray);
    cudaFree(d_AuxScannedArray);
    cudaFree(d_flags);

    return 0;
}
