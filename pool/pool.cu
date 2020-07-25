#include<time.h>
#include<iostream>

#define DLLEXPORT extern "C" __declspec(dllexport)

#define BX 4
#define BY 4
#define BC 32

__global__ void maxpool2d(float* img, int* idx, float* out, int H, int W, int C, int Kx, int Ky, int stride) {

    int c = blockIdx.x * BC + threadIdx.x;
    int nh = blockIdx.y * BX + threadIdx.y;
    int nw = blockIdx.z * BY + threadIdx.z;

    int NH = (H - Kx) / stride + 1;
    int NW = (W - Ky) / stride + 1;

    int WC = W * C;

    if (nh < NH && nw < NW && c < C) {
        int t = nh * NW * C + nw * C + c;
        float m = -99999999999;
        int mi = 0;
        for (int x = 0; x < Kx; x++) {
            for (int y = 0; y < Ky; y++) {
                int i = (nh * stride + x) * WC + (nw * stride + y) * C + c;
                float v = img[i];
                if (m < v) {
                    m = v;
                    mi = i;
                }
            }
        }
        out[t] = m;
        idx[t] = mi;
    }

}

__global__ void maxpool2d_back(float* img, int* idx, float* out, int H, int W, int C, int Kx, int Ky, int stride) {

    int c = blockIdx.x * BC + threadIdx.x;
    int h = blockIdx.y * BX + threadIdx.y;
    int w = blockIdx.z * BY + threadIdx.z;

    int NW = (W - Ky) / stride + 1;

    int WC = W * C;
    int NWC = NW * C;

    if (h < H && w < W && c < C) {
        int t = h * WC + w * C + c;
        float s = 0;
        int xl = h - (h / stride) * stride;
        int yl = w - (w / stride) * stride;
        for (int x = xl; x < Kx; x += stride) {
            int nh = (h - x) / stride;
            for (int y = yl; y < Ky; y += stride) {
                int nw = (w - y) / stride;
                int i = nh * NWC + nw * C + c;
                if (idx[i] == t) {
                    s += out[i];
                }
            }
        }
        img[t] = s;
    }

}


DLLEXPORT void cuda_maxpool2d(float* dimg, int* idx, float* dout, int H, int W, int C, int Kx, int Ky, int stride) {
    int NH = (H - Kx) / stride + 1;
    int NW = (W - Ky) / stride + 1;

    dim3 gridDims((int)ceil((float)C / BC), (int)ceil((float)NH / BX), (int)ceil((float)NW / BY));
    dim3 blockDims(BC, BX, BY);
    maxpool2d << <gridDims, blockDims >> > (dimg, idx, dout, H, W, C, Kx, Ky, stride);
}

DLLEXPORT void cuda_maxpool2d_back(float* dimg, int* idx, float* dout, int H, int W, int C, int Kx, int Ky, int stride) {
    dim3 gridDims((int)ceil((float)C / BC), (int)ceil((float)H / BX), (int)ceil((float)W / BY));
    dim3 blockDims(BC, BX, BY);
    maxpool2d_back << <gridDims, blockDims >> > (dimg, idx, dout, H, W, C, Kx, Ky, stride);
}