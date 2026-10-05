#include <iostream>
#include <cuda_runtime.h>

// 1. The GPU Kernel (Runs on the graphics card)
__global__ void hello_from_gpu() {
    int thread_id = threadIdx.x;
    printf("  [GPU] Hello from thread %d!\n", thread_id);
}

// 2. The Host Wrapper Function (Runs on the CPU, called by main.cpp)
extern "C" void run_kernel() {
    std::cout << "[CPU] Launching GPU kernel now..." << std::endl;

    // Launch the kernel with 1 block of 5 parallel threads
    hello_from_gpu<<<1, 5>>>();

    // Force the CPU to wait until the GPU completes its work
    cudaError_t err = cudaDeviceSynchronize();
    
    // Check if the kernel encountered any errors during execution
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "[CPU] GPU kernel finished execution successfully!" << std::endl;
    }
}
