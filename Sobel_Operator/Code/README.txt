CPE810 Final Project
Hao Lu & Yupeng Cao

Our project is finished on Windows and Linux. All code has been tested.
Please make sure that you have installed environment correct:
1. CUDA > 10.0 
2. OpenCV4.5.0

If you run code on Windows: 
Build the environment in Visual Studio and Run code.

If you run code on Linux:
nvcc filename.cu -o filename `pkg-config --cflags --libs opencv`


-------------------------------------------------------------------------------
sobelWithOpenCV.cpp: CPU implemenation with OpenCV
main.cpp: CPU implementation

SobelWithMul.cu: GPU implemenation
silbefilter_shared_memory.cu: Optimized by using Shared Memory

prewitt.cu: Prewitt Operator
