#include<time.h>
#include<iostream>

#define DLLEXPORT extern "C" __declspec(dllexport)


__global__ void activation(float* in, float* out, int size, int type) {

    int i = blockIdx.x * 256 + threadIdx.x;

    if (i >= size) { return; }

    if (type == 0) {
        out[i] = (in[i] <= 0) ? 0 : in[i];
    }

}

__global__ void activation_back(float* in, float* grad, float* out, int size, int type) {

    int i = blockIdx.x * 256 + threadIdx.x;

    if (i >= size) { return; }

    if (type == 0) {
        out[i] = (in[i] <= 0) ? 0 : grad[i];
    }

}


DLLEXPORT void cuda_activation(float* din, float* dout, int size, int type) {
    dim3 gridDims((int)ceil((float)size / 256), 1, 1);
    dim3 blockDims(256, 1, 1);
    activation << <gridDims, blockDims >> > (din, dout, size, type);
}

DLLEXPORT void cuda_activation_back(float* din, float* dgrad, float* dout, int size, int type) {
    dim3 gridDims((int)ceil((float)size / 256), 1, 1);
    dim3 blockDims(256, 1, 1);
    activation_back << <gridDims, blockDims >> > (din, dgrad, dout, size, type);
}