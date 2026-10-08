#include "softmax.cuh"

__global__ void softmax_gpu(float* x, float* y, int w, int h){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < h && col < w){
        float maxval = x[row*w];
        for(int i = 1; i < w; i++){
            maxval = fmaxf(maxval, x[row*w+i]);
        }
        float divisor = 0.0f;
        for(int i = 0; i < w; i++){
            divisor += expf(x[row * w + i] - maxval);
        }
        y[row * w + col] = expf(x[row * w + col] - maxval)/(divisor);
    }
}

__host__ void softmax(const Tensor& x, Tensor& y){
    if (x.data == nullptr){
        fprintf(stderr, "Softmax: a tensor is not allocated\n");
        exit(EXIT_FAILURE);
    }
    if (x.cols != y.cols || x.rows != y.rows) {
        fprintf(stderr, "Softmax: x is %dx%d but y is %dx%d (x.cols must equal y.cols and x.rows must equal y.rows)\n",
                x.rows, x.cols, y.rows, y.cols);
        exit(EXIT_FAILURE);
    }
    if(x.cols == 0 || x.rows == 0){
        fprintf(stderr, "Softmax: x should not have 0 rows or 0 columns. x is %dx%d\n",
                x.rows, x.cols);
        exit(EXIT_FAILURE);
    }
    dim3 numThreads(16,16);
    dim3 numBlocks(((y.cols+15)/16), ((x.rows+15)/16));
    softmax_gpu<<<numBlocks, numThreads>>>(x.data, y.data, x.cols, x.rows);
    KERNEL_CHECK();
}