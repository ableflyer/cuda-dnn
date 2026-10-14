#include "bias_add.cuh"

__global__ void bias_gpu(float* arr, const float* b, int n, int arr_cols){
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n){
        arr[id] = arr[id] + b[id % arr_cols];
    }
}

__host__ void bias(Tensor& arr, const Tensor& b){
    if (arr.data == nullptr){
        fprintf(stderr, "Bias: arr tensor is not allocated\n");
        exit(EXIT_FAILURE);
    }
    if (b.data == nullptr){
        fprintf(stderr, "Bias: b tensor is not allocated\n");
        exit(EXIT_FAILURE);
    }
    if(arr.cols == 0 || arr.rows == 0){
        fprintf(stderr, "Bias: arr should not have 0 rows or 0 columns. arr is %dx%d\n",
                arr.rows, arr.cols);
        exit(EXIT_FAILURE);
    }
    if(b.cols == 0 || b.rows == 0){
        fprintf(stderr, "Bias: b should not have 0 rows or 0 columns. b is %dx%d\n",
                b.rows, b.cols);
        exit(EXIT_FAILURE);
    }
    if (arr.cols != b.cols) {
        fprintf(stderr, "Bias: arr is %dx%d but b is %dx%d (arr.cols must equal b.cols)\n",
                arr.rows, arr.cols, b.rows, b.cols);
        exit(EXIT_FAILURE);
    }
    if (b.rows != 1) {
        fprintf(stderr, "Bias: b should only have 1 row. b is %dx%d\n",
                b.rows, b.cols);
        exit(EXIT_FAILURE);
    }

    int n = (arr.rows*arr.cols);
    int NumThreads = 256;
    int NumBlocks = (n + NumThreads - 1) / NumThreads;
    bias_gpu<<<NumBlocks, NumThreads>>>(arr.data, b.data, n, arr.cols);
    KERNEL_CHECK();
}