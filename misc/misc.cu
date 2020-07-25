#include<time.h>
#include<iostream>

#define DLLEXPORT extern "C" __declspec(dllexport)

// Memory

DLLEXPORT float* avx2_malloc(unsigned long long size) {
    return (float*)_aligned_malloc(sizeof(float) * size, 32);
}

DLLEXPORT float* cuda_malloc(unsigned long long size) {
    float* mem;
    cudaMalloc(&mem, sizeof(float) * size);
    cudaMemset(mem, 0, sizeof(float) * size);
    return mem;
}

DLLEXPORT void cuda_memreset(float* mem, unsigned long long size) {
    cudaMemset(mem, 0, sizeof(float) * size);
}

DLLEXPORT void cuda_memcpy(float* h, float* d, long size, int dir) {
    if (dir == 0) {
        cudaMemcpy(d, h, sizeof(float) * size, cudaMemcpyHostToDevice);
    }
    else {
        cudaMemcpy(h, d, sizeof(float) * size, cudaMemcpyDeviceToHost);
    }
}

DLLEXPORT void cuda_memcpy2d(float* h, float* d, long M, long N, long ldn, int dir) {
    if (dir == 0) {
        for (int m = 0; m < M; m++) {
            cudaMemcpy(d + m * ldn, h + m * N, sizeof(float) * N, cudaMemcpyHostToDevice);
        }
    }
    else {
        for (int m = 0; m < M; m++) {
            cudaMemcpy(h + m * N, d + m * ldn, sizeof(float) * N, cudaMemcpyDeviceToHost);
        }
    }
}

DLLEXPORT void cuda_memcpy3d(float* h, float* d, long H, long W, long C, long ldw, long ldc, int dir) {
    if (dir == 0) {
        for (int x = 0; x < H; x++) {
            for (int y = 0; y < W; y++) {
                cudaMemcpy(d + x * ldw * ldc + y * ldc, h + x * W * C + y * C, sizeof(float) * C, cudaMemcpyHostToDevice);
            }
        }
    }
    else {
        for (int x = 0; x < H; x++) {
            for (int y = 0; y < W; y++) {
                cudaMemcpy(h + x * W * C + y * C, d + x * ldw * ldc + y * ldc, sizeof(float) * C, cudaMemcpyDeviceToHost);
            }
        }
    }
}


DLLEXPORT void cuda_free(float* x) {
    cudaFree(x);
}


// In-device copy

__global__ void indevcpy(float* x, float* y, int size, int xoffset, int yoffset) {

    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < size) {
        y[i + yoffset] = x[i + xoffset];
    }

}

DLLEXPORT void cuda_indevcpy(float* x, float* y, int size, int xoffset, int yoffset) {
    dim3 gridDims((int)ceil((float)size / 256), 1, 1);
    dim3 blockDims(256, 1, 1);
    indevcpy << <gridDims, blockDims >> > (x, y, size, xoffset, yoffset);
}



// Memset

__global__ void memset(float* x, float v, int size) {

    int i = blockIdx.x * 256 + threadIdx.x;

    if (i >= size) { return; }

    x[i] = v;

}

DLLEXPORT void cuda_memsetvalue(float* dx, float v, int size) {
    dim3 gridDims((int)ceil((float)size / 256), 1, 1);
    dim3 blockDims(256, 1, 1);
    memset << <gridDims, blockDims >> > (dx, v, size);
}


__global__ void memset_pntr(float* x, float* v, int index, int size) {

    int i = blockIdx.x * 256 + threadIdx.x;

    if (i >= size) { return; }

    x[i] = v[index];

}

DLLEXPORT void cuda_memset_pntr(float* dx, float* v, int index, int size) {
    dim3 gridDims((int)ceil((float)size / 256), 1, 1);
    dim3 blockDims(256, 1, 1);
    memset_pntr << <gridDims, blockDims >> > (dx, v, index, size);
}



// Squared Difference

__global__ void squared_difference(float* h, float* y, float* out, int size) {

    int x = blockIdx.x * 256 + threadIdx.x;

    if (x >= size) { return; }

    float v = (h[x] - y[x]);
    out[x] = v * v;

}

DLLEXPORT void cuda_squared_difference(float* dh, float* dy, float* dout, int size) {
    dim3 gridDims((int)ceil((float)size / 256), 1, 1);
    dim3 blockDims(256, 1, 1);
    squared_difference << <gridDims, blockDims >> > (dh, dy, dout, size);
}



// Multiply Subtract

__global__ void submul(float* x, float* y, float* z, float* out, int size) {

    int i = blockIdx.x * 256 + threadIdx.x;

    if (i >= size) { return; }

    out[i] = (x[i] - y[i]) * z[i];

}

DLLEXPORT void cuda_submul(float* dx, float* dy, float* dz, float* dout, int size) {
    dim3 gridDims((int)ceil((float)size / 256), 1, 1);
    dim3 blockDims(256, 1, 1);
    submul << <gridDims, blockDims >> > (dx, dy, dz, dout, size);
}



// Broadcast Multiply

__global__ void broadcastmul(float* x, float v, float* out, int size) {

    int i = blockIdx.x * 256 + threadIdx.x;

    if (i >= size) { return; }

    out[i] = x[i] * v;

}

DLLEXPORT void cuda_broadcastmul(float* dx, float v, float* dout, int size) {
    dim3 gridDims((int)ceil((float)size / 256), 1, 1);
    dim3 blockDims(256, 1, 1);
    broadcastmul << <gridDims, blockDims >> > (dx, v, dout, size);
}


__global__ void broadcastmul_pntr(float* x, float* v, int index, float* out, int size) {

    int i = blockIdx.x * 256 + threadIdx.x;

    if (i >= size) { return; }

    out[i] = x[i] * v[index];

}

DLLEXPORT void cuda_broadcastmul_pntr(float* dx, float* v, int index, float* dout, int size) {
    dim3 gridDims((int)ceil((float)size / 256), 1, 1);
    dim3 blockDims(256, 1, 1);
    broadcastmul_pntr << <gridDims, blockDims >> > (dx, v, index, dout, size);
}



// Strided Sum

__global__ void strided_sum(float* x, float* out, int stride, int size) {

    int i = blockIdx.x * 256 + threadIdx.x;

    if (i < stride) {
        int c = size / stride;
        float s = 0;
        int p = i;
        for (int j = 0; j < c; j++) {
            s += x[p];
            p += stride;
        }
        out[i] = s;
    }

}

DLLEXPORT void cuda_strided_sum(float* dx, float* dout, int stride, int size) {
    dim3 gridDims((int)ceil((float)stride / 256), 1, 1);
    dim3 blockDims(256, 1, 1);
    strided_sum << <gridDims, blockDims >> > (dx, dout, stride, size);
}



// Tiled Add

__global__ void tiled_add(float* x, float* out, int xsize, int osize) {

    int i = blockIdx.x * 256 + threadIdx.x;

    if (i < osize) {
        out[i] += x[i % xsize];
    }

}

DLLEXPORT void cuda_tiled_add(float* dx, float* dout, int xsize, int osize) {
    dim3 gridDims((int)ceil((float)osize / 256), 1, 1);
    dim3 blockDims(256, 1, 1);
    tiled_add << <gridDims, blockDims >> > (dx, dout, xsize, osize);
}



// Add

__global__ void add(float* x, float* y, float* out, int size) {

    int i = blockIdx.x * 256 + threadIdx.x;

    if (i < size) {
        out[i] = x[i] + y[i];
    }

}

DLLEXPORT void cuda_add(float* dx, float* dy, float* dout, int size) {
    dim3 gridDims((int)ceil((float)size / 256), 1, 1);
    dim3 blockDims(256, 1, 1);
    add << <gridDims, blockDims >> > (dx, dy, dout, size);
}



// Total Sum

__global__ void total_sum_pass(float* x, float* tmp, int size, int threads, int stride) {

    int i = blockIdx.x * threads + threadIdx.x;

    int c = (int)ceil((float)size / stride);

    float s = 0;
    float* p = x + i;
    for (int j = 0; j < c - 1; j++) {
        s += *p;
        p += stride;
    }
    if ((c - 1) * stride + i < size) {
        s += x[(c - 1) * stride + i];
    }
    tmp[i] = s;

}

DLLEXPORT void cuda_total_sum(float* dx, float* dtmp, float* dout, int size) {
    total_sum_pass << <dim3(8, 1, 1), dim3(256, 1, 1) >> > (dx, dtmp, size, 256, 2048);
    total_sum_pass << <dim3(1, 1, 1), dim3(64, 1, 1) >> > (dtmp, dtmp, 2048, 64, 32);
    total_sum_pass << <dim3(1, 1, 1), dim3(16, 1, 1) >> > (dtmp, dtmp, 32, 16, 2);
    total_sum_pass << <dim3(1, 1, 1), dim3(1, 1, 1) >> > (dtmp, dtmp, 2, 1, 1);

    indevcpy << <dim3(1, 1, 1), dim3(1, 1, 1) >> > (dtmp, dout, 1, 0, 0);
}
