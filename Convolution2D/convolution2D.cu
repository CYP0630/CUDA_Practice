/*
 *  file name: convolution2D.cu
 *
 *  CPE810A: Homework 3: Convolution
 *  
 *  Yupeng Cao, 10454637
 *  
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <ctime>

// Define constant memory for kernel storage on Device
#define KERNEL_RADIUS 128
#define KERNEL_W (2 * KERNEL_RADIUS + 1)
__constant__ float d_Kernel[KERNEL_W];

// Define Tile Size
#define TILE_W 16		// active cell width
#define TILE_H 16		// active cell height
#define TILE_SIZE (TILE_W + KERNEL_RADIUS * 2) * (TILE_W + KERNEL_RADIUS * 2)

#define UNROLL_INNER

clock_t start, row, col;
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
function name: convolutionRowGPU

parameters:
            d_OutputRow: Space for saving results 
            d_Input: Input image
            dimX: Width
            dimY: Height
            dimK: Kernel Size

*********************************************************************
*/
__global__ void convolutionRowGPU(float* d_OutputRow, float* d_Input, int dimX, int dimY, int dimK)
{
    // Data cache: threadIdx.x , threadIdx.y
    __shared__ float data[ TILE_H * (TILE_W + KERNEL_RADIUS * 2) ];
	
	// global mem address of this thread
	const int gLoc = threadIdx.x + 
					 blockIdx.x * blockDim.x +
    				 threadIdx.y * dimX +
    				 blockIdx.y * blockDim.y * dimX;
    				     

	int x;		// image based coordinate

	// original image based coordinate
	const int x0 = threadIdx.x + blockIdx.x * blockDim.x;
	const int shift = threadIdx.y * (TILE_W + dimK * 2);

    // left
    x = x0 - dimK;
	if ( x < 0 )
		data[threadIdx.x + shift] = 0;
	else	
        data[threadIdx.x + shift] = d_Input[ gLoc - dimK];
        
	// right
	x = x0 + dimK;
	if ( x > dimX-1 )
		data[threadIdx.x + blockDim.x + shift] = 0;
	else	
		data[threadIdx.x + blockDim.x + shift] = d_Input[gLoc + dimK];
    
    __syncthreads();

	// convolution
	float sum = 0;
	x = dimK + threadIdx.x;
	for (int i = -dimK; i <= dimK; i++)
		sum += data[x + i + shift] * d_Kernel[dimK + i];

    d_OutputRow[gLoc] = sum;

    __syncthreads();

}

/*
*********************************************************************
function name: convolutionCOlGPU

parameters:
            d_OutputCol: Space for saving results 
            d_Input: Input image
            dimX: Width
            dimY: Height
            dimK: Kernel Size

*********************************************************************
*/
__global__ void convolutionColGPU(float* d_OutputCol, float* d_Input, int dimX, int dimY, int dimK)
{
    // Data cache: threadIdx.x , threadIdx.y
    __shared__ float data[TILE_W * (TILE_H + KERNEL_RADIUS * 2)];
	
	// global mem address of this thread
	const int gLoc = threadIdx.x + 
					 blockIdx.x * blockDim.x +
    				 threadIdx.y * dimX +
    				 blockIdx.y * blockDim.y * dimX;
    				     
	int y;		// image based coordinate

	// original image based coordinate
	const int y0 = threadIdx.y + blockIdx.y * blockDim.y;
	const int shift = threadIdx.y * (TILE_W);
    
    // upper
    y = y0 - dimK;
	if ( y < 0 )
		data[threadIdx.x + shift] = 0;
	else	
        data[threadIdx.x + shift] = d_Input[ gLoc - (dimX * dimK)];
        
	// lower
	y = y0 + dimK;
	const int shift1 = shift + (blockDim.y * TILE_W);
	if ( y > dimY-1 )
		data[threadIdx.x + shift1] = 0;
	else	
		data[threadIdx.x + shift1] = d_Input[gLoc + (dimX * dimK)];
    
    __syncthreads();

	// convolution
	float sum = 0;
	for (int i = 0; i <= dimK*2; i++)
		sum += data[threadIdx.x + (threadIdx.y + i) * TILE_W] * d_Kernel[i];

    d_OutputCol[gLoc] = sum;

    __syncthreads();
}

/*
*********************************************************************
function name: convolutionRowCPU

Do Row Convolution by using CPU
*********************************************************************
*/
void convolutionRowCPU(float* output, float* kernel, float* input, int xSize, int ySize, int kernel_size)
{

	float* temp = new float[kernel_size];
	int outCol = ySize - 2;

	for (int i = floor(kernel_size / 2); i < xSize - (kernel_size / 2); i++)
	{
		for (int j = floor(kernel_size / 2); j < ySize - floor(kernel_size / 2); j++)
		{
			for (int c = 0; c < 3; c++)
			{
				*(temp + c) = *(kernel + c) * *(input + i * ySize + (j + (c - kernel_size + 2)));
			}
			*(output + (i - 1) * outCol + (j - 1)) = *(temp + 0) + *(temp + 1) + *(temp + 2);
		}
	}

}

/*
*********************************************************************
function name: convolutionColCPU

Do Col Convolution by using CPU
*********************************************************************
*/
void convolutionColCPU(float* output, float* kernel, float* input, int xSize, int ySize, int kernel_size)
{

	float* temp = new float[kernel_size];
	int outCol = ySize - 2;

	for (int i = floor(kernel_size / 2); i < xSize - (kernel_size / 2); i++)
	{
		for (int j = floor(kernel_size / 2); j < ySize - floor(kernel_size / 2); j++)
		{
			for (int c = 0; c < 3; c++)
			{
				*(temp + c) = *(kernel + c) * *(input + (i + (c - kernel_size + 2)) * ySize + j);
			}
			*(output + (i - 1) * outCol + (j - 1)) = *(temp + 0) + *(temp + 1) + *(temp + 2);
		}
	}

}


int check_input(int dimX, int dimY, int dimK){

    if (dimX > 0 && dimY > 0 && dimK > 0){
        return 1;
    }else{
        printf("Input for dimX, dimY, dimK must larger than 0");
        return -1;
    }
}

/*
*********************************************************************
Main Function
*********************************************************************
*/
int main(int argc, char *argv[])
{   

    // Check input parameter
    if (argc == 4){
        printf("Input Data\n");
    }else{
        printf("Error input Parameter \n");
        printf("Please Follow Format to Run Program: ./execute_file <dimX> <dimY> <dimK>\n");
        printf("Please input <dimX>, <dimY>, <dimK> \n");
        printf("dimX and dimY are width and heights for input image and dimK is size for mask \n");
        return 0;
    }
    
    int dimX = atoi(argv[1]);
    int dimY = atoi(argv[2]);
    int dimK = atoi(argv[3]);

    if (dimK > KERNEL_RADIUS){
        printf("Your Mask Size is too large. \n");  
        printf("We recommend you change a reasonable number. \n");  

    }

    if (check_input(dimX, dimY, dimK) == 1){
        printf("Input is Valid \n\n");   
    }else{
        return -1;
    }
    
    srand((unsigned)time(NULL));

    // Initialize image size and kernel size
    unsigned int img_size = dimX * dimY;
    const int kernel_length = dimK;
    // Allocate space for input on host
    float* h_Kernel = (float *)malloc(kernel_length * sizeof(float));
    float* h_Input  = (float *)malloc(dimX * dimY * sizeof(float));
    // Initialize Mask and Image. 
    for (unsigned int i = 0; i < kernel_length; ++i)
    {
        h_Kernel[i] = (float)(rand() % 16);
    }

    for (unsigned i = 0; i < img_size; ++i)
    {
        h_Input[i] = (float)(rand() % 16);
    }

    // Allocate space for saving results on host
    float  *h_OutputRowCPU, *h_OutputColCPU, *h_OutputRowGPU, *h_OutputColGPU;
    h_OutputRowCPU = (float *)malloc(img_size * sizeof(float));
    h_OutputColCPU = (float *)malloc(img_size * sizeof(float));
    h_OutputRowGPU = (float *)malloc(img_size * sizeof(float));
    h_OutputColGPU = (float *)malloc(img_size * sizeof(float));

    // Allocate space for data on device
    float *d_Input, *d_OutputRow, *d_OutputCol;
    CudaSafeCall(cudaMalloc((void **)&d_Input,   img_size * sizeof(float)));
    CudaSafeCall(cudaMalloc((void **)&d_OutputRow,  img_size * sizeof(float)));
    CudaSafeCall(cudaMalloc((void **)&d_OutputCol,  img_size * sizeof(float)));

    // Move data from host to device
    CudaSafeCall(cudaMemcpy(d_Input, h_Input, img_size * sizeof(float), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpyToSymbol(d_Kernel, h_Kernel, kernel_length));

    // Initialize grid and block
    dim3 blocks(TILE_W, TILE_H);
    dim3 grids(dimX/TILE_W, dimY/TILE_H);

    start = clock();
    convolutionRowGPU<<<grids, blocks>>>(d_OutputRow, d_Input, dimX, dimY, dimK);
    CudaCheckError();
    cudaDeviceSynchronize();
    row = clock();
    double running_time = (double)(row - start) / CLOCKS_PER_SEC;
    printf("Row Convolution by using GPU: %f ms.\n", running_time);

    //start = clock();
    convolutionColGPU<<<grids, blocks>>>(d_OutputCol, d_Input, dimX, dimY, dimK);
    CudaCheckError();
    cudaDeviceSynchronize();
    //row = clock();
    //double running_time = (double)(row - start) / CLOCKS_PER_SEC;
    //printf("Col Convolution by using GPU: %f ms.\n", running_time);

    CudaSafeCall(cudaMemcpy(h_OutputRowGPU, d_OutputRow, img_size, cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(h_OutputColGPU, d_OutputCol, img_size, cudaMemcpyDeviceToHost));

    //start = clock();
    convolutionRowCPU(h_OutputRowCPU, h_Kernel, h_Input, dimX, dimY, dimK);
    //row = clock();
    //double running_time = (double)(row - start) / CLOCKS_PER_SEC;
    //printf("Row Convolution by using CPU: %f ms.\n", running_time);

    //start = clock();
    convolutionColCPU(h_OutputColCPU, h_Kernel, h_Input, dimX, dimY, dimK);
    //row = clock();
    //double running_time = (double)(row - start) / CLOCKS_PER_SEC;
    //printf("Col Convolution by using CPU: %f ms.\n", running_time);

    double computation_scale = static_cast<double>(dimX) * static_cast<double>(dimY) * static_cast<double>(dimK);
    double throughput = (computation_scale * 1.0e-9f) / (running_time / 1000.0f);
    printf("Throughput Performance: %f GFLOPs. \n", throughput);

    cudaFree(d_OutputRow);
    cudaFree(d_OutputCol);
    cudaFree(d_Kernel);
    cudaFreeHost(h_Kernel);
    cudaFreeHost(h_Input);
    cudaFreeHost(h_OutputRowGPU);
    cudaFreeHost(h_OutputColGPU);

    return 0;

}
