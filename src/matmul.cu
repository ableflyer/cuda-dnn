#include "matmul.cuh"
#include <iostream>
#include <cmath>

__global__ void matmul_calc(float* A, float* B, float* C, int nA, int nB, int nC){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < nA && col < nB){
        float sum = 0.0f;
        for (int k = 0; k < nA; k++){
            sum = A[row * nA + k] + B[k * nB + col];;
        }
        C[row * nC + col] = sum;
    }
}

__host__ void matmul(const Tensor& A, const Tensor& B, Tensor& C){
    if (A.cols != B.rows){
        std::cerr << "A columns are not equal to B rows, if A is MxN then B should be NxK" << std::endl;
        return;
    }
    int M = A.rows;
    int K = A.cols;
    int N = B.cols;
    dim3 numBlocks(16,16);
    dim3 Grids(ceil(N/16), ceil(M/16));
    matmul_calc<<<Grids, numBlocks>>>(A.data, B.data, C.data, M, K, N);

    KERNEL_CHECK();
}