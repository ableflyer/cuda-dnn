#include <iostream>
#include <cuda_runtime.h>
#include "tensor.cuh"

// 1. The GPU Kernel (Runs on the graphics card)
__global__ void hello_from_gpu(float* d_data, int size) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < size) {
        printf("  [GPU] Thread %d (Block %d, Local %d) -> value = %f\n", 
               global_id, blockIdx.x, threadIdx.x, d_data[global_id]);
    }
}

// 2. The Host Wrapper Function (Runs on the CPU, called by main.cpp)
extern "C" void run_kernel() {
    std::vector<float> h = {10, 11, 12, 20, 21, 22};
    Tensor t;

    t.allocate(2, 2);
    t.allocate(2, 3); // this is to test if it actually releases the previous one
    std::cout << t.rows << " " << t.cols << " " << t.bytes() << std::endl;
    t.from_host(h);

    std::cout << "[CPU] Launching GPU kernel now..." << std::endl;

    // Launch the kernel with 1 block of 5 parallel threads
    hello_from_gpu<<<1, 6>>>(t.data, (t.rows * t.cols));

    // Force the CPU to wait until the GPU completes its work
    cudaError_t err = cudaDeviceSynchronize();
    
    // Check if the kernel encountered any errors during execution
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "[CPU] GPU kernel finished execution successfully!" << std::endl;
    }

    t.release();
    t.release();
    std::cout << t.rows << " " << t.cols << " " << t.bytes() << std::endl;
}
