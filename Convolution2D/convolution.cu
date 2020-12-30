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

// Define constant memory for kernel storage on Device
#define KERNEL_RADIUS 3
#define KERNEL_W (2 * KERNEL_RADIUS + 1)
__device__ __constant__ float d_Kernel[KERNEL_W];


#define TILE_W 16		// active cell width
#define TILE_H 16		// active cell height
#define TILE_SIZE (TILE_W + KERNEL_RADIUS * 2) * (TILE_W + KERNEL_RADIUS * 2)

#define IMUL(a,b) __mul24(a,b)
#define UNROLL_INNER


__global__ void convolutionRowGPU(
    float *d_Result,
    float *d_Data,
    int dataW,
    int dataH
)
{
    // Data cache: threadIdx.x , threadIdx.y
    __shared__ float data[ TILE_H * (TILE_W + KERNEL_RADIUS * 2) ];
	
	// global mem address of this thread
	const int gLoc = threadIdx.x + 
					 IMUL(blockIdx.x, blockDim.x) +
    				 IMUL(threadIdx.y, dataW) +
    				 IMUL(blockIdx.y, blockDim.y) * dataW;
    				     
    // load cache (32x16 shared memory, 16x16 threads blocks)
    // each threads loads two values from global memory into shared mem
    // if in image area, get value in global mem, else 0
	int x;		// image based coordinate

	// original image based coordinate
	const int x0 = threadIdx.x + IMUL(blockIdx.x, blockDim.x);
	const int shift = threadIdx.y * (TILE_W + KERNEL_RADIUS * 2);

	// right
	x = x0 + KERNEL_RADIUS;
	if ( x > dataW-1 )
		data[threadIdx.x + blockDim.x + shift] = 0;
	else	
		data[threadIdx.x + blockDim.x + shift] = d_Data[gLoc + KERNEL_RADIUS];
    
    __syncthreads();

	// convolution
	float sum = 0;
	x = KERNEL_RADIUS + threadIdx.x;
	for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++)
		sum += data[x + i + shift] * d_Kernel[KERNEL_RADIUS + i];

    d_Result[gLoc] = sum;

}

__global__ void convolutionColGPU(
    float *d_Result,
    float *d_Data,
    int dataW,
    int dataH
)
{
    // Data cache: threadIdx.x , threadIdx.y
    __shared__ float data[TILE_W * (TILE_H + KERNEL_RADIUS * 2)];
	
	// global mem address of this thread
	const int gLoc = threadIdx.x + 
					 IMUL(blockIdx.x, blockDim.x) +
    				 IMUL(threadIdx.y, dataW) +
    				 IMUL(blockIdx.y, blockDim.y) * dataW;
    				     
    // load cache (32x16 shared memory, 16x16 threads blocks)
    // each threads loads two values from global memory into shared mem
    // if in image area, get value in global mem, else 0
	int y;		// image based coordinate

	// original image based coordinate
	const int y0 = threadIdx.y + IMUL(blockIdx.y, blockDim.y);
	const int shift = threadIdx.y * (TILE_W);
	
	// case2: lower
	y = y0 + KERNEL_RADIUS;
	const int shift1 = shift + IMUL(blockDim.y, TILE_W);
	if ( y > dataH-1 )
		data[threadIdx.x + shift1] = 0;
	else	
		data[threadIdx.x + shift1] = d_Data[gLoc + IMUL(dataW, KERNEL_RADIUS)];
    
    __syncthreads();

	// convolution
	float sum = 0;
	for (int i = 0; i <= KERNEL_RADIUS*2; i++)
		sum += data[threadIdx.x + (threadIdx.y + i) * TILE_W] * d_Kernel[i];

    d_Result[gLoc] = sum;

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

    if (dimK != KERNEL_RADIUS){
        printf("Please change KERNEKL_RADIUS values \n");   
        printf("Keep dimK = KERNEKL_RADIUS \n");  
        return -1;
    }
    
    
    // Initialize image size and kernel size
    unsigned int img_size = dimX * dimY;
    const int kernel_length = dimK;

    // Allocate space on host
    float *h_Kernel, *h_Input, *h_OutputCPU, *h_OutputGPU;
    h_Kernel    = (float *)malloc(kernel_length * sizeof(float));
    h_Input     = (float *)malloc(img_size * sizeof(float));
    h_OutputCPU = (float *)malloc(img_size * sizeof(float));
    h_OutputGPU = (float *)malloc(img_size * sizeof(float));

    // Initialize Mask and Image. 
    srand(200);
    for (unsigned int i = 0; i < kernel_length; i++)
    {
        h_Kernel[i] = (float)(rand() % 16);
    }

    for (unsigned i = 0; i < img_size; i++)
    {
        h_Input[i] = (float)(rand() % 16);
    }

    float *d_Input, *d_Temporary_Output, *d_Output;
    cudaMalloc((void **)&d_Input,   img_size * sizeof(float));
    cudaMalloc((void **)&d_Temporary_Output,  img_size * sizeof(float));
    cudaMalloc((void **)&d_Output,  img_size * sizeof(float));

    cudaMemcpy(d_Input, h_Input, img_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_Kernel, h_Kernel, kernel_length);

    dim3 blocks(TILE_W, TILE_H);
	dim3 grids(dimX/TILE_W, dimY/TILE_H);

    convolutionRowGPU<<<grids, blocks>>>(d_Temporary_Output, d_Input, dimX, dimY);
    cudaDeviceSynchronize();
    convolutionColGPU<<<grids, blocks>>>(d_Output, d_Temporary_Output, dimX, dimY);
    cudaDeviceSynchronize();

    return 0;

}
