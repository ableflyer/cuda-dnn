#include "ReLU.cuh"

__global__ void ReLU_F_GPU(float* x, float* y, int n){
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < n){
        y[id] = fmaxf(0.0f, x[id]);
    }
    
}

__host__ void ReLUForward(const Tensor& x, Tensor& fx){
    if (x.data == nullptr){
        fprintf(stderr, "ReLU: a tensor is not allocated\n");
        exit(EXIT_FAILURE);
    }
    if (x.cols != fx.cols || x.rows != fx.rows) {
        fprintf(stderr, "ReLU: x is %dx%d but fx is %dx%d (x.cols must equal fx.cols and x.rows must equal fx.rows)\n",
                x.rows, x.cols, fx.rows, fx.cols);
        exit(EXIT_FAILURE);
    }
    if(x.cols == 0 || x.rows == 0){
        fprintf(stderr, "ReLU: x should not have 0 rows or 0 columns. x is %dx%d\n",
                x.rows, x.cols);
        exit(EXIT_FAILURE);
    }
    if(fx.cols == 0 || fx.rows == 0){
        fprintf(stderr, "ReLU: fx should not have 0 rows or 0 columns. fx is %dx%d\n",
                fx.rows, fx.cols);
        exit(EXIT_FAILURE);
    }
    int n = (x.rows*x.cols);
    int NumThreads = 256;
    int NumBlocks = (n + NumThreads - 1) / NumThreads;
    ReLU_F_GPU<<<NumBlocks, NumThreads>>>(x.data, fx.data, n);
    KERNEL_CHECK();
}