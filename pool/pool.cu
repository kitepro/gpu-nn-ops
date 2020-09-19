#include<time.h>
#include<iostream>

#define DLLEXPORT extern "C" __declspec(dllexport)

#define BX 4
#define BY 4
#define BC 32

__global__ void maxpool2d(float* img, float* out, int H, int W, int C, int Kx, int Ky, int stride, int batchsize) {

    int c = blockIdx.x * BC + threadIdx.x;
    int nh = blockIdx.y * BX + threadIdx.y;
    int nw = blockIdx.z * BY + threadIdx.z;

    int NH = (H - Kx) / stride + 1;
    int NW = (W - Ky) / stride + 1;

    int WC = W * C;
    int HWC = H * WC;
    int NWC = NW * C;
    int NHNWC = NH * NWC;

    if (nh < NH && nw < NW && c < C) {
        for (int b = 0; b < batchsize; b++) {
            int t = b * NHNWC + nh * NWC + nw * C + c;
            float m = -99999999999;
            for (int x = 0; x < Kx; x++) {
                for (int y = 0; y < Ky; y++) {
                    int i = b * HWC + (nh * stride + x) * WC + (nw * stride + y) * C + c;
                    float v = img[i];
                    if (m < v) {
                        m = v;
                    }
                }
            }
            out[t] = m;
        }
    }

}

__global__ void maxpool2d_back(float* img, float* grad, float* out, int H, int W, int C, int Kx, int Ky, int stride, int batchsize) {

    int c = blockIdx.x * BC + threadIdx.x;
    int nh = blockIdx.y * BX + threadIdx.y;
    int nw = blockIdx.z * BY + threadIdx.z;

    int NH = (H - Kx) / stride + 1;
    int NW = (W - Ky) / stride + 1;

    int WC = W * C;
    int HWC = H * WC;
    int NWC = NW * C;
    int NHNWC = NH * NWC;

    if (nh < NH && nw < NW && c < C) {
        for (int b = 0; b < batchsize; b++) {
            int t = b * NHNWC + nh * NWC + nw * C + c;
            float m = -99999999999;
            int mi = 0;
            for (int x = 0; x < Kx; x++) {
                for (int y = 0; y < Ky; y++) {
                    int i = b * HWC + (nh * stride + x) * WC + (nw * stride + y) * C + c;
                    float v = img[i];
                    if (m < v) {
                        m = v;
                        mi = i;
                    }
                }
            }
            out[mi] = grad[t];
        }
    }

}


DLLEXPORT void cuda_maxpool2d(float* dimg, float* dout, int H, int W, int C, int Kx, int Ky, int stride, int batchsize) {
    int NH = (H - Kx) / stride + 1;
    int NW = (W - Ky) / stride + 1;

    dim3 gridDims((int)ceil((float)C / BC), (int)ceil((float)NH / BX), (int)ceil((float)NW / BY));
    dim3 blockDims(BC, BX, BY);
    maxpool2d << <gridDims, blockDims >> > (dimg, dout, H, W, C, Kx, Ky, stride, batchsize);
}

DLLEXPORT void cuda_maxpool2d_back(float* dimg, float* dgrad, float* dout, int H, int W, int C, int Kx, int Ky, int stride, int batchsize) {
    dim3 gridDims((int)ceil((float)C / BC), (int)ceil((float)H / BX), (int)ceil((float)W / BY));
    dim3 blockDims(BC, BX, BY);
    maxpool2d_back << <gridDims, blockDims >> > (dimg, dgrad, dout, H, W, C, Kx, Ky, stride, batchsize);
}