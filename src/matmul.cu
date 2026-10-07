#include "matmul.cuh"
#include <iostream>

__global__ void matmul_calc(float* A, float* B, float* C, int M, int K, int N){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__host__ void matmul(const Tensor& A, const Tensor& B, Tensor& C){
    if (A.data == nullptr || B.data == nullptr || C.data == nullptr) {
        fprintf(stderr, "matmul: a tensor is not allocated\n");
        exit(EXIT_FAILURE);
    }
    if (A.cols != B.rows) {
        fprintf(stderr, "matmul: A is %dx%d but B is %dx%d (A.cols must equal B.rows)\n",
                A.rows, A.cols, B.rows, B.cols);
        exit(EXIT_FAILURE);
    }
    if (C.rows != A.rows || C.cols != B.cols) {
        fprintf(stderr, "matmul: C is %dx%d but A x B needs %dx%d\n",
                C.rows, C.cols, A.rows, B.cols);
        exit(EXIT_FAILURE);
    }
    if (C.data == A.data || C.data == B.data) {
        fprintf(stderr, "matmul: C must not share memory with A or B\n");
        exit(EXIT_FAILURE);
    }
    int K = A.cols;
    int M = A.rows;
    int N = B.cols;
    dim3 numThreads(16,16);
    dim3 numBlocks(((N+15)/16), ((M+15)/16));
    matmul_calc<<<numBlocks, numThreads>>>(A.data, B.data, C.data, M, K, N);
    KERNEL_CHECK();
}