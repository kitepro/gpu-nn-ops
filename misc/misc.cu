#include<time.h>
#include<iostream>

#define DLLEXPORT extern "C" __declspec(dllexport)

cudaStream_t mem_streams[4];
int mem_stream_p = 0;

// Init
DLLEXPORT void init() {
    for (int i = 0; i < 4; i++) {
        cudaError_t result = cudaStreamCreate(&(mem_streams[i]));
        if (result != cudaSuccess) {
            exit(0);
        }
    }
}


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

DLLEXPORT float* cuda_malloc_host(unsigned long long size) {
    float* mem;
    cudaMallocHost(&mem, sizeof(float) * size);
    cudaMemset(mem, 0, sizeof(float) * size);
    return mem;
}

DLLEXPORT float* cuda_pointer_offset(float* d, long offset) {
    return d + offset;
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

DLLEXPORT int cuda_memcpyasync(float* h, float* d, long size, int dir) {
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (dir == 0) {
        cudaMemcpyAsync(d, h, sizeof(float) * size, cudaMemcpyHostToDevice, stream);
    }
    else {
        cudaMemcpyAsync(h, d, sizeof(float) * size, cudaMemcpyDeviceToHost, stream);
    }
    return mem_stream_p - 1;
}

DLLEXPORT void cuda_free(float* x) {
    cudaFree(x);
}

DLLEXPORT void cuda_sync() {
    cudaDeviceSynchronize();
}

DLLEXPORT long cuda_get_free_mem() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return free;
}

DLLEXPORT void cuda_reset() {
    cudaDeviceReset();
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


// Membroadcast

__global__ void membroadcast(float* x, float* v, int xsize, int vsize) {

    int i = blockIdx.x * 256 + threadIdx.x;

    if (i < xsize) {
        int p = i / (xsize / vsize);
        x[i] = v[p];
    }

}

DLLEXPORT void cuda_membroadcast(float* dx, float* v, int xsize, int vsize) {

    dim3 gridDims((int)ceil((float)xsize / 256), 1, 1);
    dim3 blockDims(256, 1, 1);
    membroadcast << <gridDims, blockDims >> > (dx, v, xsize, vsize);
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

__global__ void squared_difference_back(float* h, float* y, float* grad, float* out, int size) {

    int x = blockIdx.x * 256 + threadIdx.x;

    if (x >= size) { return; }

    float v = (h[x] - y[x]) * grad[x];
    out[x] = 2 * v;

}

DLLEXPORT void cuda_squared_difference_back(float* dh, float* dy, float* dgrad, float* dout, int size) {
    dim3 gridDims((int)ceil((float)size / 256), 1, 1);
    dim3 blockDims(256, 1, 1);
    squared_difference_back << <gridDims, blockDims >> > (dh, dy, dgrad, dout, size);
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
        int b = blockIdx.y * c / gridDim.y;
        c = (blockIdx.y + 1) * c / gridDim.y;
        float s = 0;
        int p = i + b * stride;
        for (int j = b; j < c; j++) {
            s += x[p];
            p += stride;
        }
        out[blockIdx.y * stride + i] = s;
    }
}

DLLEXPORT void cuda_strided_sum(float* dx, float* dout, int stride, int size) {
    strided_sum << <dim3((int)ceil((float)stride / 256), 1, 1), dim3(256, 1, 1) >> > (dx, dout, stride, size);
}

DLLEXPORT void cuda_strided_sum_batched(float* dx, float* dout, float* dtmp, int stride, int size, int batches) {
    strided_sum << <dim3((int)ceil((float)stride / 256), batches, 1), dim3(256, 1, 1) >> > (dx, dtmp, stride, size);
    strided_sum << <dim3((int)ceil((float)stride / 256), 1, 1), dim3(256, 1, 1) >> > (dtmp, dout, stride, batches * stride);
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



// Batched Sum

__global__ void batched_sum_pass(float* x, float* tmp, int size, int stride) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    x += b * size;

    int c = (int)ceil((float)(size - i) / stride);

    float s = 0;
    int p = i;
    for (int j = 0; j < c; j++) {
        s += x[p];
        p += stride;
    }
    tmp[b * stride + i] = s;
}

DLLEXPORT void cuda_batched_sum(float* dx, float* dtmp1, float* dtmp2, float* dout, int size, int batches) {
    batched_sum_pass << <dim3(8, batches, 1), dim3(256, 1, 1) >> > (dx, dtmp1, size, 2048);
    batched_sum_pass << <dim3(1, batches, 1), dim3(64, 1, 1) >> > (dtmp1, dtmp2, 2048, 64);
    batched_sum_pass << <dim3(1, batches, 1), dim3(32, 1, 1) >> > (dtmp2, dtmp1, 64, 32);
    batched_sum_pass << <dim3(1, batches, 1), dim3(1, 1, 1) >> > (dtmp1, dtmp2, 32, 1);
    indevcpy << <dim3((int)(ceil((float)batches/256)), 1, 1), dim3(256, 1, 1) >> > (dtmp2, dout, batches, 0, 0);
}
