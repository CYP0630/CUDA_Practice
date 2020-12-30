#include<stdio.h>

int dev=0;

cudaDeviceProp devProp;
cudaGetDeviceProperties(&devProp, dev);
cout << "GPU device" << dev << ":" << devProp.name << std::endl;
