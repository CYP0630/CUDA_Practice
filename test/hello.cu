# include<stdio.h>

__global__ void mykernel()
{
    printf("hello world for GPU\n");
}

int main()
{
    mykernel<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}

