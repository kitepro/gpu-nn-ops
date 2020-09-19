#include<time.h>
#include<iostream>

#define DLLEXPORT extern "C" __declspec(dllexport)

__global__ void mrs(float* x, float* g, float* m, float* r, int size, float lr, float beta1, float beta2, float eps, int ts) {

    int i = (blockIdx.x * 256 + threadIdx.x) * 4;

    if (i < size) {
        float nlr = lr * sqrt(1 - pow(beta2, ts)) / (1 - pow(beta1, ts));
        if (i + 4 <= size) {
            float4 m4 = *reinterpret_cast<float4*>(m + i);
            float4 r4 = *reinterpret_cast<float4*>(r + i);
            float4 g4 = *reinterpret_cast<float4*>(g + i);
            float4 x4 = *reinterpret_cast<float4*>(x + i);
            m4.x = beta1 * m4.x + (1 - beta1) * g4.x;
            m4.y = beta1 * m4.y + (1 - beta1) * g4.y;
            m4.z = beta1 * m4.z + (1 - beta1) * g4.z;
            m4.w = beta1 * m4.w + (1 - beta1) * g4.w;
            *reinterpret_cast<float4*>(m + i) = m4;
            r4.x = beta2 * r4.x + (1 - beta2) * g4.x * g4.x;
            r4.y = beta2 * r4.y + (1 - beta2) * g4.y * g4.y;
            r4.z = beta2 * r4.z + (1 - beta2) * g4.z * g4.z;
            r4.w = beta2 * r4.w + (1 - beta2) * g4.w * g4.w;
            *reinterpret_cast<float4*>(r + i) = r4;
            x4.x -= nlr * m4.x / (sqrt(r4.x) + eps);
            x4.y -= nlr * m4.y / (sqrt(r4.y) + eps);
            x4.z -= nlr * m4.z / (sqrt(r4.z) + eps);
            x4.w -= nlr * m4.w / (sqrt(r4.w) + eps);
            *reinterpret_cast<float4*>(x + i) = x4;
        }
        else if(i + 3 <= size) {
            m[i + 0] = beta1 * m[i + 0] + (1 - beta1) * g[i + 0];
            m[i + 1] = beta1 * m[i + 1] + (1 - beta1) * g[i + 1];
            m[i + 2] = beta1 * m[i + 2] + (1 - beta1) * g[i + 2];
            r[i + 0] = beta2 * r[i + 0] + (1 - beta2) * g[i + 0] * g[i + 0];
            r[i + 1] = beta2 * r[i + 1] + (1 - beta2) * g[i + 1] * g[i + 1];
            r[i + 2] = beta2 * r[i + 2] + (1 - beta2) * g[i + 2] * g[i + 2];
            x[i + 0] -= nlr * m[i + 0] / (sqrt(r[i + 0]) + eps);
            x[i + 1] -= nlr * m[i + 1] / (sqrt(r[i + 1]) + eps);
            x[i + 2] -= nlr * m[i + 2] / (sqrt(r[i + 2]) + eps);
        }
        else if (i + 2 <= size) {
            m[i + 0] = beta1 * m[i + 0] + (1 - beta1) * g[i + 0];
            m[i + 1] = beta1 * m[i + 1] + (1 - beta1) * g[i + 1];
            r[i + 0] = beta2 * r[i + 0] + (1 - beta2) * g[i + 0] * g[i + 0];
            r[i + 1] = beta2 * r[i + 1] + (1 - beta2) * g[i + 1] * g[i + 1];
            x[i + 0] -= nlr * m[i + 0] / (sqrt(r[i + 0]) + eps);
            x[i + 1] -= nlr * m[i + 1] / (sqrt(r[i + 1]) + eps);
        }
        else if (i + 1 <= size) {
            m[i + 0] = beta1 * m[i + 0] + (1 - beta1) * g[i + 0];
            r[i + 0] = beta2 * r[i + 0] + (1 - beta2) * g[i + 0] * g[i + 0];
            x[i + 0] -= nlr * m[i + 0] / (sqrt(r[i + 0]) + eps);
        }
    }

}

DLLEXPORT void cuda_mrs(float* dx, float* dg, float* dm, float* dr, int size, float lr, float beta1, float beta2, float eps, int ts) {
    dim3 gridDims((int)ceil((float)size / 1024), 1, 1);
    dim3 blockDims(256, 1, 1);
    mrs << <gridDims, blockDims >> > (dx, dg, dm, dr, size, lr, beta1, beta2, eps, ts);
}
