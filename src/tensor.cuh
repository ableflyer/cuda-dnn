#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>
#include "cuda_utils.cuh"

struct Tensor{
    float* data = nullptr;
    int rows = 0;
    int cols = 0;

    size_t bytes() const;

    void allocate(int r, int c);

    void from_host(const std::vector<float>& h);

    std::vector<float> to_host() const;

    void release();
};

