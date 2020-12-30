#include <iostream>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;
// define constant memory for 3*3 Sobel operator
__device__ __constant__ int d_sobel_x[6];
__device__ __constant__ int d_sobel_y[6];


/*
 * components for the kernel function:
 * inout image data
 * output image data
 * image height
 * image width
 */
__global__ void sobelGpu(unsigned char* input, unsigned char* output, int imgH, int imgW) {
    //computing with multiple threads
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = xIndex + yIndex * imgW;

    // calculate image
    int Gx = 0;
    int Gy = 0;

    while (offset < (imgH - 2) * (imgW - 2)) {
        // X direction
        Gx = d_sobel_x[0] * input[(yIndex)*imgW + xIndex] + d_sobel_x[1] * input[(yIndex + 1) * imgW + xIndex]
           + d_sobel_x[2] * input[(yIndex + 2) * imgW + xIndex] + d_sobel_x[3] * input[(yIndex)*imgW + xIndex + 2]
           + d_sobel_x[4] * input[(yIndex + 1) * imgW + xIndex + 2] + d_sobel_x[5] * input[(yIndex + 2) * imgW + xIndex + 2];

        // Y gradient
        Gy = d_sobel_y[0] * input[(yIndex)*imgW + xIndex] + d_sobel_y[1] * input[(yIndex)*imgW + xIndex + 1]
           + d_sobel_y[2] * input[(yIndex)*imgW + xIndex + 2] + d_sobel_y[3] * input[(yIndex + 2) * imgW + xIndex]
           + d_sobel_y[4] * input[(yIndex + 2) * imgW + xIndex + 1] + d_sobel_y[5] * input[(yIndex + 2) * imgW + xIndex + 2];

        // using absolute value to accelerate calculation
        int sum = abs(Gx) + abs(Gy);
        // the maximum of gray scale vale is 255
        if (sum > 255) {
            sum = 255;
        }
        output[offset] = sum;
        xIndex += blockDim.x * gridDim.x;

        // boundary condition
        if (xIndex > imgW - 2) {
            yIndex += blockDim.y * gridDim.y;
            xIndex = threadIdx.x + blockIdx.x * blockDim.x;
        }
        offset = xIndex + yIndex * imgW;
    }
}



// the main function
int main() {
    // input nad check
    Mat gray_img = imread("test.jpg", 0);
    if (gray_img.data == NULL) {
        cout << "Wrong input!" << endl;
        return -1;
    }
    imshow("input image (gray)", gray_img);
    int imgH = gray_img.rows;
    int imgW = gray_img.cols;

    // initialze the image after gauss filter
    Mat gaussImg;
    // implementation of the gauss filter with a 3 X 3 kernel
    GaussianBlur(gray_img, gaussImg, Size(3, 3), 0, 0, BORDER_DEFAULT);

    // Sobel Operator
    int* sobel_x;
    int* sobel_y;
    sobel_x = (int*)malloc(6 * sizeof(int));
    sobel_y = (int*)malloc(6 * sizeof(int));
    // horizontal operator
    sobel_x[0] = -1; sobel_x[3] = 1;
    sobel_x[1] = -2; sobel_x[4] = 2;
    sobel_x[2] = -1; sobel_x[5] = 1;
    // vertical operator
    sobel_y[0] = -1; sobel_y[1] = -2; sobel_y[2] = -1;
    sobel_y[3] = 1; sobel_y[4] = 2; sobel_y[5] = 1;

    //the image for data after processed by GPU
    Mat out_img(imgH, imgW, CV_8UC1, Scalar(0));

    // initial time recording
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // start recording
    cudaEventRecord(start, 0);

    // allocate device memory
    unsigned char* d_in;
    unsigned char* d_out;
    cudaMalloc((void**)&d_in, imgH * imgW * sizeof(unsigned char));
    cudaMalloc((void**)&d_out, imgH * imgW * sizeof(unsigned char));

    // copy input image and Sobel operator to device
    cudaMemcpy(d_in, gaussImg.data, imgH * imgW * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_sobel_x, sobel_x, 6 * sizeof(int));
    cudaMemcpyToSymbol(d_sobel_y, sobel_y, 6 * sizeof(int));
    dim3 blocks((int)((imgW + 31) / 32), (int)(imgH + 31) / 32);
    dim3 threads(16, 16);

    // call the kernel function
    sobelGpu << <blocks, threads >> > (d_in, d_out, imgH, imgW);

    // copy output image back to host
    cudaMemcpy(out_img.data, d_out, imgH * imgW * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // stop time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // compute the performance
    float totalTime;
    cudaEventElapsedTime(&totalTime, start, stop);
    cout << "Time for edge detection using Sobel operator in GPU is: " << static_cast<double>(totalTime) << " ms." << endl;
    //printf( "The time for execution with ognized threads and block dimentions: %.6f ms \n", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //save the output image
    imwrite("edge detection_gpu.jpg", out_img);
    imshow("edge detection_gpu", out_img);

    //free memory
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_sobel_x);
    cudaFree(d_sobel_y);
    cudaFreeHost(sobel_x);
    cudaFreeHost(sobel_y);

    waitKey();
}
