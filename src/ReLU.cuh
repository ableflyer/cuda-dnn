#include "matmul.cuh"

void ReLUForward(const Tensor& x, Tensor& fx);

void ReLUBackward(const Tensor& x, Tensor& fx);