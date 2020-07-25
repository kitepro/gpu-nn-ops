#include<time.h>
#include<iostream>

#define DLLEXPORT extern "C" __declspec(dllexport)

__global__ void mrs(float* x, float* g, float* m, float* r, int size, float lr, float beta1, float beta2, float eps, int ts) {

    int i = blockIdx.x * 256 + threadIdx.x;

    if (i < size) {
        float nlr = lr * sqrt(1 - pow(beta2, ts)) / (1 - pow(beta1, ts));
        m[i] = beta1 * m[i] + (1 - beta1) * g[i];
        r[i] = beta2 * r[i] + (1 - beta2) * g[i] * g[i];
        x[i] -= nlr * m[i] / (sqrt(r[i]) + eps);
    }

}

DLLEXPORT void cuda_mrs(float* dx, float* dg, float* dm, float* dr, int size, float lr, float beta1, float beta2, float eps, int ts) {
    dim3 gridDims((int)ceil((float)size / 256), 1, 1);
    dim3 blockDims(256, 1, 1);
    mrs << <gridDims, blockDims >> > (dx, dg, dm, dr, size, lr, beta1, beta2, eps, ts);
}
