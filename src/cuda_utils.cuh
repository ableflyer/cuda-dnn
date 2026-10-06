#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Wraps any CUDA runtime call: cudaMalloc, cudaMemcpy, cudaEventCreate, and so on.
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d\n  %s: %s\n  call: %s\n",      \
                    __FILE__, __LINE__, cudaGetErrorName(err_),                 \
                    cudaGetErrorString(err_), #call);                           \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Wraps cuBLAS calls, which return cublasStatus_t.
#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t st_ = (call);                                            \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuBLAS error at %s:%d\n  %s\n  call: %s\n",        \
                    __FILE__, __LINE__, cublasGetStatusString(st_), #call);     \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Call after every kernel launch.
// Release builds check launch errors only. Debug builds also synchronise,
// so an illegal memory access is reported at the kernel that caused it.
#ifdef NDEBUG
  #define KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())
#else
  #define KERNEL_CHECK()                                                        \
      do {                                                                      \
          CUDA_CHECK(cudaGetLastError());                                       \
          CUDA_CHECK(cudaDeviceSynchronize());                                  \
      } while (0)
#endif