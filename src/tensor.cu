#include "tensor.cuh"
#include <iostream>
#include <stdexcept>

__host__ size_t Tensor::bytes() const{
    return static_cast<size_t>(rows) * cols * sizeof(float);
}

__host__ void Tensor::allocate(int r, int c){
    if(data != nullptr) release();
    if(r > 0 && c > 0){
        rows = r;
        cols = c;
        CUDA_CHECK(cudaMalloc(&data, bytes()));
    }
}

__host__ void Tensor::from_host(const std::vector<float>& h){
    size_t total = static_cast<size_t>(rows) * cols;
    if(h.size() != total){
        std::cout << "Host size and GPU allocation size is not equal, exiting" << std::endl;
        return;
    }
    CUDA_CHECK(cudaMemcpy(data, h.data(), bytes(), cudaMemcpyHostToDevice));
}

__host__ std::vector<float> Tensor::to_host() const{
    if (data == nullptr) {
        throw std::runtime_error("Tensor::to_host() failed: Device pointer 'data' is null.");
    }

    size_t total = static_cast<size_t>(rows) * cols;
    std::vector<float> h(total);
    CUDA_CHECK(cudaMemcpy(h.data(), data, bytes(), cudaMemcpyDeviceToHost));
    return h;
}

__host__ void Tensor::release(){
    if(data != nullptr){
        cudaFree(data);
        data = nullptr;
        rows = 0;
        cols = 0;
    }
}