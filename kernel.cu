#include <iostream>
#include <cuda_runtime.h>
#include "softmax.cuh"

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
    std::vector<float> h_a = {1,2,3,4,5,6}, h_b = {1, 0, -1, 0, 1, 0, -1, 0, 1};
    Tensor a, b, c, d, e;

    a.allocate(2,3);
    b.allocate(3,3);
    
    c.allocate(2,3);
    d.allocate(2,3);
    e.allocate(2,3);

    a.from_host(h_a);
    b.from_host(h_b);

    std::cout << "[CPU] Launching GPU kernel now..." << std::endl;

    // Launch the kernel with 1 block of 5 parallel threads
    matmul(a, b, c);

    ReLUForward(c, d);

    softmax(d, e);
    int rows = e.rows;
    int columns = e.cols;
    std::vector<float> res = e.to_host();

    for(int i = 0; i < rows; i++){
        for(int j = 0; j < columns; j++){
            std::cout << res[i*columns+j] << " ";
        }
        std::cout << "\n";
    }

    // Force the CPU to wait until the GPU completes its work
    cudaError_t err = cudaDeviceSynchronize();
    
    // Check if the kernel encountered any errors during execution
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "[CPU] GPU kernel finished execution successfully!" << std::endl;
    }

    a.release();
    b.release();
    c.release();
    d.release();
    e.release();
    KERNEL_CHECK();
}
