#include<time.h>
#include<iostream>

#define DLLEXPORT extern "C" __declspec(dllexport)

// Image transformations

__global__ void conv2d_wino_4_3_i_4x4x32(float* img, float* out, int H, int W, int C, int ldw, int ldc, int padding) {

    __shared__ float sI[10][10][32];

    int tx = threadIdx.x;
    int cb = blockIdx.x * 32;
    int nhb = blockIdx.y * 4;
    int nwb = blockIdx.z * 4;
    int hb = nhb * 2;
    int wb = nwb * 2;

    int txm8 = tx % 8;
    int txb8 = tx / 8;
    int txb8m2 = txb8 % 2;
    int txb8b2 = txb8 / 2;
    int txb8m4 = txb8 % 4;
    int txb8b4 = txb8 / 4;
    int txb8m8 = txb8 % 8;
    int txb8b8 = txb8 / 8;
    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txb32m2 = txb32 % 2;
    int txb32b2 = txb32 / 2;

    int WC = W * C;
    int C16 = ldc * 16;
    int NW16C = ldw * C16;

    int x, y;
    float rI[6][6];
    float a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p;

#pragma unroll
    for (x = 0; x < 10; x += 2) {
#pragma unroll
        for (y = 0; y < 10; y += 2) {
            if (hb + x + txb32b2 - padding >= 0 && hb + x + txb32b2 - padding < H && wb + y + txb32m2 - padding >= 0 && wb + y + txb32m2 - padding < W && cb + txm32 < C) {
                sI[x + txb32b2][y + txb32m2][txm32] = img[(hb + x + txb32b2 - padding) * WC + (wb + y + txb32m2 - padding) * C + cb + txm32];
            }
            else {
                sI[x + txb32b2][y + txb32m2][txm32] = 0;
            }
        }
    }

    __syncthreads();

#pragma unroll
    for (x = 0; x < 6; x++) {
#pragma unroll
        for (y = 0; y < 6; y++) {
            rI[x][y] = sI[txb32b2 * 4 + x][txb32m2 * 4 + y][txm32];
        }
    }

#pragma unroll
    for (x = 0; x < 2; x++) {
#pragma unroll
        for (y = 0; y < 2; y++) {
            a = rI[x * 2 + 0][y * 2 + 0];
            b = rI[x * 2 + 0][y * 2 + 1];
            c = rI[x * 2 + 0][y * 2 + 2];
            d = rI[x * 2 + 0][y * 2 + 3];
            e = rI[x * 2 + 1][y * 2 + 0];
            f = rI[x * 2 + 1][y * 2 + 1];
            g = rI[x * 2 + 1][y * 2 + 2];
            h = rI[x * 2 + 1][y * 2 + 3];
            i = rI[x * 2 + 2][y * 2 + 0];
            j = rI[x * 2 + 2][y * 2 + 1];
            k = rI[x * 2 + 2][y * 2 + 2];
            l = rI[x * 2 + 2][y * 2 + 3];
            m = rI[x * 2 + 3][y * 2 + 0];
            n = rI[x * 2 + 3][y * 2 + 1];
            o = rI[x * 2 + 3][y * 2 + 2];
            p = rI[x * 2 + 3][y * 2 + 3];

            out[(nhb + txb32b2 * 2 + x) * NW16C + (nwb + txb32m2 * 2 + y) * C16 + 0 * ldc + cb + txm32] = +a - i - c + k;
            out[(nhb + txb32b2 * 2 + x) * NW16C + (nwb + txb32m2 * 2 + y) * C16 + 1 * ldc + cb + txm32] = +b - j + c - k;
            out[(nhb + txb32b2 * 2 + x) * NW16C + (nwb + txb32m2 * 2 + y) * C16 + 2 * ldc + cb + txm32] = -b + j + c - k;
            out[(nhb + txb32b2 * 2 + x) * NW16C + (nwb + txb32m2 * 2 + y) * C16 + 3 * ldc + cb + txm32] = +b - j - d + l;
            out[(nhb + txb32b2 * 2 + x) * NW16C + (nwb + txb32m2 * 2 + y) * C16 + 4 * ldc + cb + txm32] = +e + i - g - k;
            out[(nhb + txb32b2 * 2 + x) * NW16C + (nwb + txb32m2 * 2 + y) * C16 + 5 * ldc + cb + txm32] = +f + j + g + k;
            out[(nhb + txb32b2 * 2 + x) * NW16C + (nwb + txb32m2 * 2 + y) * C16 + 6 * ldc + cb + txm32] = -f - j + g + k;
            out[(nhb + txb32b2 * 2 + x) * NW16C + (nwb + txb32m2 * 2 + y) * C16 + 7 * ldc + cb + txm32] = +f + j - h - l;
            out[(nhb + txb32b2 * 2 + x) * NW16C + (nwb + txb32m2 * 2 + y) * C16 + 8 * ldc + cb + txm32] = -e + i + g - k;
            out[(nhb + txb32b2 * 2 + x) * NW16C + (nwb + txb32m2 * 2 + y) * C16 + 9 * ldc + cb + txm32] = -f + j - g + k;
            out[(nhb + txb32b2 * 2 + x) * NW16C + (nwb + txb32m2 * 2 + y) * C16 + 10 * ldc + cb + txm32] = +f - j - g + k;
            out[(nhb + txb32b2 * 2 + x) * NW16C + (nwb + txb32m2 * 2 + y) * C16 + 11 * ldc + cb + txm32] = -f + j + h - l;
            out[(nhb + txb32b2 * 2 + x) * NW16C + (nwb + txb32m2 * 2 + y) * C16 + 12 * ldc + cb + txm32] = +e - m - g + o;
            out[(nhb + txb32b2 * 2 + x) * NW16C + (nwb + txb32m2 * 2 + y) * C16 + 13 * ldc + cb + txm32] = +f - n + g - o;
            out[(nhb + txb32b2 * 2 + x) * NW16C + (nwb + txb32m2 * 2 + y) * C16 + 14 * ldc + cb + txm32] = -f + n + g - o;
            out[(nhb + txb32b2 * 2 + x) * NW16C + (nwb + txb32m2 * 2 + y) * C16 + 15 * ldc + cb + txm32] = +f - n - h + p;
        }
    }

}

__global__ void conv2d_wino_4_3_i_4x4x16(float* img, float* out, int H, int W, int C, int ldw, int ldc, int padding) {

    __shared__ float sI[10][10][16];

    int tx = threadIdx.x;
    int cb = blockIdx.x * 16;
    int nhb = blockIdx.y * 4;
    int nwb = blockIdx.z * 4;
    int hb = nhb * 2;
    int wb = nwb * 2;

    int txm8 = tx % 8;
    int txb8 = tx / 8;
    int txb8m2 = txb8 % 2;
    int txb8b2 = txb8 / 2;
    int txb8m4 = txb8 % 4;
    int txb8b4 = txb8 / 4;
    int txb8m8 = txb8 % 8;
    int txb8b8 = txb8 / 8;
    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txb16m2 = txb16 % 2;
    int txb16b2 = txb16 / 2;

    int WC = W * C;
    int C16 = ldc * 16;
    int NW16C = ldw * C16;

    int x, y;
    float rI[6][6];
    float a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p;

#pragma unroll
    for (x = 0; x < 10; x += 2) {
#pragma unroll
        for (y = 0; y < 10; y += 2) {
            if (hb + x + txb16b2 - padding >= 0 && hb + x + txb16b2 - padding < H && wb + y + txb16m2 - padding >= 0 && wb + y + txb16m2 - padding < W && cb + txm16 < C) {
                sI[x + txb16b2][y + txb16m2][txm16] = img[(hb + x + txb16b2 - padding) * WC + (wb + y + txb16m2 - padding) * C + cb + txm16];
            }
            else {
                sI[x + txb16b2][y + txb16m2][txm16] = 0;
            }
        }
    }

    __syncthreads();

#pragma unroll
    for (x = 0; x < 6; x++) {
#pragma unroll
        for (y = 0; y < 6; y++) {
            rI[x][y] = sI[txb16b2 * 4 + x][txb16m2 * 4 + y][txm16];
        }
    }

#pragma unroll
    for (x = 0; x < 2; x++) {
#pragma unroll
        for (y = 0; y < 2; y++) {
            a = rI[x * 2 + 0][y * 2 + 0];
            b = rI[x * 2 + 0][y * 2 + 1];
            c = rI[x * 2 + 0][y * 2 + 2];
            d = rI[x * 2 + 0][y * 2 + 3];
            e = rI[x * 2 + 1][y * 2 + 0];
            f = rI[x * 2 + 1][y * 2 + 1];
            g = rI[x * 2 + 1][y * 2 + 2];
            h = rI[x * 2 + 1][y * 2 + 3];
            i = rI[x * 2 + 2][y * 2 + 0];
            j = rI[x * 2 + 2][y * 2 + 1];
            k = rI[x * 2 + 2][y * 2 + 2];
            l = rI[x * 2 + 2][y * 2 + 3];
            m = rI[x * 2 + 3][y * 2 + 0];
            n = rI[x * 2 + 3][y * 2 + 1];
            o = rI[x * 2 + 3][y * 2 + 2];
            p = rI[x * 2 + 3][y * 2 + 3];

            out[(nhb + txb16b2 * 2 + x) * NW16C + (nwb + txb16m2 * 2 + y) * C16 + 0 * ldc + cb + txm16] = +a - i - c + k;
            out[(nhb + txb16b2 * 2 + x) * NW16C + (nwb + txb16m2 * 2 + y) * C16 + 1 * ldc + cb + txm16] = +b - j + c - k;
            out[(nhb + txb16b2 * 2 + x) * NW16C + (nwb + txb16m2 * 2 + y) * C16 + 2 * ldc + cb + txm16] = -b + j + c - k;
            out[(nhb + txb16b2 * 2 + x) * NW16C + (nwb + txb16m2 * 2 + y) * C16 + 3 * ldc + cb + txm16] = +b - j - d + l;
            out[(nhb + txb16b2 * 2 + x) * NW16C + (nwb + txb16m2 * 2 + y) * C16 + 4 * ldc + cb + txm16] = +e + i - g - k;
            out[(nhb + txb16b2 * 2 + x) * NW16C + (nwb + txb16m2 * 2 + y) * C16 + 5 * ldc + cb + txm16] = +f + j + g + k;
            out[(nhb + txb16b2 * 2 + x) * NW16C + (nwb + txb16m2 * 2 + y) * C16 + 6 * ldc + cb + txm16] = -f - j + g + k;
            out[(nhb + txb16b2 * 2 + x) * NW16C + (nwb + txb16m2 * 2 + y) * C16 + 7 * ldc + cb + txm16] = +f + j - h - l;
            out[(nhb + txb16b2 * 2 + x) * NW16C + (nwb + txb16m2 * 2 + y) * C16 + 8 * ldc + cb + txm16] = -e + i + g - k;
            out[(nhb + txb16b2 * 2 + x) * NW16C + (nwb + txb16m2 * 2 + y) * C16 + 9 * ldc + cb + txm16] = -f + j - g + k;
            out[(nhb + txb16b2 * 2 + x) * NW16C + (nwb + txb16m2 * 2 + y) * C16 + 10 * ldc + cb + txm16] = +f - j - g + k;
            out[(nhb + txb16b2 * 2 + x) * NW16C + (nwb + txb16m2 * 2 + y) * C16 + 11 * ldc + cb + txm16] = -f + j + h - l;
            out[(nhb + txb16b2 * 2 + x) * NW16C + (nwb + txb16m2 * 2 + y) * C16 + 12 * ldc + cb + txm16] = +e - m - g + o;
            out[(nhb + txb16b2 * 2 + x) * NW16C + (nwb + txb16m2 * 2 + y) * C16 + 13 * ldc + cb + txm16] = +f - n + g - o;
            out[(nhb + txb16b2 * 2 + x) * NW16C + (nwb + txb16m2 * 2 + y) * C16 + 14 * ldc + cb + txm16] = -f + n + g - o;
            out[(nhb + txb16b2 * 2 + x) * NW16C + (nwb + txb16m2 * 2 + y) * C16 + 15 * ldc + cb + txm16] = +f - n - h + p;
        }
    }

}


__global__ void conv2d_wino_4_3_k_16x16(float* kern, float* out, int N, int C, int ldc) {

    int tx = threadIdx.x;
    int nb = blockIdx.y * 16;
    int cb = blockIdx.x * 16;

    int txm16 = tx % 16;
    int txb16 = tx / 16;

    int NC = N * C;
    int KyNC = 3 * NC;
    int C16 = ldc * 16;

    float4 t;
    float a, b, c, d, e, f, g, h, i;

    if (nb + txb16 < N && cb + txm16 < C) {
        a = kern[0 * KyNC + 0 * NC + (nb + txb16) * C + cb + txm16];
        b = kern[0 * KyNC + 1 * NC + (nb + txb16) * C + cb + txm16];
        c = kern[0 * KyNC + 2 * NC + (nb + txb16) * C + cb + txm16];
        d = kern[1 * KyNC + 0 * NC + (nb + txb16) * C + cb + txm16];
        e = kern[1 * KyNC + 1 * NC + (nb + txb16) * C + cb + txm16];
        f = kern[1 * KyNC + 2 * NC + (nb + txb16) * C + cb + txm16];
        g = kern[2 * KyNC + 0 * NC + (nb + txb16) * C + cb + txm16];
        h = kern[2 * KyNC + 1 * NC + (nb + txb16) * C + cb + txm16];
        i = kern[2 * KyNC + 2 * NC + (nb + txb16) * C + cb + txm16];
    }
    else {
        a = b = c = d = e = f = g = h = i = 0;
    }

    // kern is Nx16xC

    float s = a + b + c + d + e + f + g + h + i;

    out[(nb + txb16) * C16 + 0 * ldc + cb + txm16] = a;
    out[(nb + txb16) * C16 + 1 * ldc + cb + txm16] = (float)0.5 * (a + b + c);
    out[(nb + txb16) * C16 + 2 * ldc + cb + txm16] = (float)0.5 * (a - b + c);
    out[(nb + txb16) * C16 + 3 * ldc + cb + txm16] = c;
    out[(nb + txb16) * C16 + 4 * ldc + cb + txm16] = (float)0.5 * (a + d + g);
    out[(nb + txb16) * C16 + 5 * ldc + cb + txm16] = (float)0.25 * s;
    out[(nb + txb16) * C16 + 6 * ldc + cb + txm16] = (float)0.25 * s - (float)0.5 * (b + e + h);
    out[(nb + txb16) * C16 + 7 * ldc + cb + txm16] = (float)0.5 * (c + f + i);
    out[(nb + txb16) * C16 + 8 * ldc + cb + txm16] = (float)0.5 * (a - d + g);
    out[(nb + txb16) * C16 + 9 * ldc + cb + txm16] = (float)0.25 * s - (float)0.5 * (d + e + f);
    out[(nb + txb16) * C16 + 10 * ldc + cb + txm16] = (float)0.25 * (a - d + g - b + e - h + c - f + i);
    out[(nb + txb16) * C16 + 11 * ldc + cb + txm16] = (float)0.5 * (c - f + i);
    out[(nb + txb16) * C16 + 12 * ldc + cb + txm16] = g;
    out[(nb + txb16) * C16 + 13 * ldc + cb + txm16] = (float)0.5 * (g + h + i);
    out[(nb + txb16) * C16 + 14 * ldc + cb + txm16] = (float)0.5 * (g - h + i);
    out[(nb + txb16) * C16 + 15 * ldc + cb + txm16] = i;
}


__global__ void conv2d_wino_4_3_o_2x2x32(float* res, float* out, int H, int W, int N, int ldh, int ldw, int ldn) {

    // res is WIxWIx16xN

    int tx = threadIdx.x;

    int nb = blockIdx.x * 32;
    int hb = blockIdx.y * 2;
    int wb = blockIdx.z * 2;

    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txb32m2 = txb32 % 2;
    int txb32b2 = txb32 / 2;

    int N16 = N * 16;
    int TW16N = W * N16;
    int NWN = ldw * ldn;

    if ((hb + txb32b2) * 2 < ldh && (wb + txb32m2) * 2 < ldw && nb + txm32 < ldn) {
        float a = res[(hb + txb32b2) * TW16N + (wb + txb32m2) * N16 + 0 * N + nb + txm32];
        float b = res[(hb + txb32b2) * TW16N + (wb + txb32m2) * N16 + 1 * N + nb + txm32];
        float c = res[(hb + txb32b2) * TW16N + (wb + txb32m2) * N16 + 2 * N + nb + txm32];
        float d = res[(hb + txb32b2) * TW16N + (wb + txb32m2) * N16 + 3 * N + nb + txm32];
        float e = res[(hb + txb32b2) * TW16N + (wb + txb32m2) * N16 + 4 * N + nb + txm32];
        float f = res[(hb + txb32b2) * TW16N + (wb + txb32m2) * N16 + 5 * N + nb + txm32];
        float g = res[(hb + txb32b2) * TW16N + (wb + txb32m2) * N16 + 6 * N + nb + txm32];
        float h = res[(hb + txb32b2) * TW16N + (wb + txb32m2) * N16 + 7 * N + nb + txm32];
        float i = res[(hb + txb32b2) * TW16N + (wb + txb32m2) * N16 + 8 * N + nb + txm32];
        float j = res[(hb + txb32b2) * TW16N + (wb + txb32m2) * N16 + 9 * N + nb + txm32];
        float k = res[(hb + txb32b2) * TW16N + (wb + txb32m2) * N16 + 10 * N + nb + txm32];
        float l = res[(hb + txb32b2) * TW16N + (wb + txb32m2) * N16 + 11 * N + nb + txm32];
        float m = res[(hb + txb32b2) * TW16N + (wb + txb32m2) * N16 + 12 * N + nb + txm32];
        float n = res[(hb + txb32b2) * TW16N + (wb + txb32m2) * N16 + 13 * N + nb + txm32];
        float o = res[(hb + txb32b2) * TW16N + (wb + txb32m2) * N16 + 14 * N + nb + txm32];
        float p = res[(hb + txb32b2) * TW16N + (wb + txb32m2) * N16 + 15 * N + nb + txm32];

        out[((hb + txb32b2) * 2 + 0) * NWN + ((wb + txb32m2) * 2 + 0) * ldn + nb + txm32] = a + e + i + b + f + j + c + g + k;
        out[((hb + txb32b2) * 2 + 0) * NWN + ((wb + txb32m2) * 2 + 1) * ldn + nb + txm32] = b + f + j - c - g - k - d - h - l;
        out[((hb + txb32b2) * 2 + 1) * NWN + ((wb + txb32m2) * 2 + 0) * ldn + nb + txm32] = e - i - m + f - j - n + g - k - o;
        out[((hb + txb32b2) * 2 + 1) * NWN + ((wb + txb32m2) * 2 + 1) * ldn + nb + txm32] = f - j - n - g + k + o - h + l + p;
    }
}

__global__ void conv2d_wino_4_3_o_2x2x16(float* res, float* out, int H, int W, int N, int ldh, int ldw, int ldn) {

    // res is WIxWIx16xN

    int tx = threadIdx.x;

    int nb = blockIdx.x * 16;
    int hb = blockIdx.y * 2;
    int wb = blockIdx.z * 2;

    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txb16m2 = txb16 % 2;
    int txb16b2 = txb16 / 2;

    int N16 = N * 16;
    int TW16N = W * N16;
    int NWN = ldw * ldn;

    if ((hb + txb16b2) * 2 < ldh && (wb + txb16m2) * 2 < ldw && nb + txm16 < ldn) {
        float a = res[(hb + txb16b2) * TW16N + (wb + txb16m2) * N16 + 0 * N + nb + txm16];
        float b = res[(hb + txb16b2) * TW16N + (wb + txb16m2) * N16 + 1 * N + nb + txm16];
        float c = res[(hb + txb16b2) * TW16N + (wb + txb16m2) * N16 + 2 * N + nb + txm16];
        float d = res[(hb + txb16b2) * TW16N + (wb + txb16m2) * N16 + 3 * N + nb + txm16];
        float e = res[(hb + txb16b2) * TW16N + (wb + txb16m2) * N16 + 4 * N + nb + txm16];
        float f = res[(hb + txb16b2) * TW16N + (wb + txb16m2) * N16 + 5 * N + nb + txm16];
        float g = res[(hb + txb16b2) * TW16N + (wb + txb16m2) * N16 + 6 * N + nb + txm16];
        float h = res[(hb + txb16b2) * TW16N + (wb + txb16m2) * N16 + 7 * N + nb + txm16];
        float i = res[(hb + txb16b2) * TW16N + (wb + txb16m2) * N16 + 8 * N + nb + txm16];
        float j = res[(hb + txb16b2) * TW16N + (wb + txb16m2) * N16 + 9 * N + nb + txm16];
        float k = res[(hb + txb16b2) * TW16N + (wb + txb16m2) * N16 + 10 * N + nb + txm16];
        float l = res[(hb + txb16b2) * TW16N + (wb + txb16m2) * N16 + 11 * N + nb + txm16];
        float m = res[(hb + txb16b2) * TW16N + (wb + txb16m2) * N16 + 12 * N + nb + txm16];
        float n = res[(hb + txb16b2) * TW16N + (wb + txb16m2) * N16 + 13 * N + nb + txm16];
        float o = res[(hb + txb16b2) * TW16N + (wb + txb16m2) * N16 + 14 * N + nb + txm16];
        float p = res[(hb + txb16b2) * TW16N + (wb + txb16m2) * N16 + 15 * N + nb + txm16];

        out[((hb + txb16b2) * 2 + 0) * NWN + ((wb + txb16m2) * 2 + 0) * ldn + nb + txm16] = a + e + i + b + f + j + c + g + k;
        out[((hb + txb16b2) * 2 + 0) * NWN + ((wb + txb16m2) * 2 + 1) * ldn + nb + txm16] = b + f + j - c - g - k - d - h - l;
        out[((hb + txb16b2) * 2 + 1) * NWN + ((wb + txb16m2) * 2 + 0) * ldn + nb + txm16] = e - i - m + f - j - n + g - k - o;
        out[((hb + txb16b2) * 2 + 1) * NWN + ((wb + txb16m2) * 2 + 1) * ldn + nb + txm16] = f - j - n - g + k + o - h + l + p;
    }
}


// Derivatives

__global__ void conv2d_wino_4_3_di_2x2x32(float* img, float* out, int H, int W, int C, int ldh, int ldw, int ldc, int padding) {

    __shared__ float sO[2][2][16][32];

    int tx = threadIdx.x;
    int cb = blockIdx.x * 32;
    int hb = blockIdx.y * 2;
    int wb = blockIdx.z * 2;

    int nhb = (hb + padding) / 2 - 1;
    int nwb = (wb + padding) / 2 - 1;

    int C16 = ldc * 16;
    int NWC16 = ldw * C16;
    int WC = W * C;

    int txm8 = tx % 8;
    int txb8 = tx / 8;
    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txb16m2 = txb16 % 2;
    int txb32m2 = txb32 % 2;
    int txb32b2 = txb32 / 2;

    int h, w;

    if (cb + txm32 < C) {
#pragma unroll
        for (h = 0; h < 2; h++) {
#pragma unroll
            for (w = 0; w < 2; w++) {
                if (nhb + h >= 0 && nhb + h < ldh && nwb + w >= 0 && nwb + w < ldw) {
                    sO[h][w][txb32 * 4 + 0][txm32] = out[(nhb + h) * NWC16 + (nwb + w) * C16 + (txb32 * 4 + 0) * ldc + cb + txm32];
                    sO[h][w][txb32 * 4 + 1][txm32] = out[(nhb + h) * NWC16 + (nwb + w) * C16 + (txb32 * 4 + 1) * ldc + cb + txm32];
                    sO[h][w][txb32 * 4 + 2][txm32] = out[(nhb + h) * NWC16 + (nwb + w) * C16 + (txb32 * 4 + 2) * ldc + cb + txm32];
                    sO[h][w][txb32 * 4 + 3][txm32] = out[(nhb + h) * NWC16 + (nwb + w) * C16 + (txb32 * 4 + 3) * ldc + cb + txm32];
                }
                else {
                    sO[h][w][txb32 * 4 + 0][txm32] = 0;
                    sO[h][w][txb32 * 4 + 1][txm32] = 0;
                    sO[h][w][txb32 * 4 + 2][txm32] = 0;
                    sO[h][w][txb32 * 4 + 3][txm32] = 0;
                }
            }
        }
    }

    __syncthreads();

    if (cb + txm32 < C && hb + txb32b2 - padding >= 0 && hb + txb32b2 - padding < H && wb + txb32m2 - padding >= 0 && wb + txb32m2 - padding < W) {
        float r = 0;

        if (txb32 == 0) {
            r += sO[0][0][0][txm32];
            r -= sO[0][0][1][txm32];
            r -= sO[0][0][2][txm32];
            r -= sO[0][0][4][txm32];
            r += sO[0][0][5][txm32];
            r += sO[0][0][6][txm32];
            r -= sO[0][0][8][txm32];
            r += sO[0][0][9][txm32];
            r += sO[0][0][10][txm32];
            r += sO[0][1][4][txm32];
            r += sO[0][1][8][txm32];
            r -= sO[1][0][0][txm32];
            r += sO[1][0][1][txm32];
            r += sO[1][0][2][txm32];
            r += sO[1][1][0][txm32];
        }
        else if (txb32 == 1) {
            r += sO[0][0][3][txm32];
            r -= sO[0][0][7][txm32];
            r -= sO[0][0][11][txm32];
            r -= sO[0][1][1][txm32];
            r += sO[0][1][2][txm32];
            r -= sO[0][1][3][txm32];
            r += sO[0][1][5][txm32];
            r -= sO[0][1][6][txm32];
            r += sO[0][1][7][txm32];
            r += sO[0][1][9][txm32];
            r -= sO[0][1][10][txm32];
            r += sO[0][1][11][txm32];
            r -= sO[1][0][3][txm32];
            r += sO[1][1][1][txm32];
            r -= sO[1][1][2][txm32];
            r += sO[1][1][3][txm32];
        }
        else if (txb32 == 2) {
            r += sO[0][0][12][txm32];
            r -= sO[0][0][13][txm32];
            r -= sO[0][0][14][txm32];
            r -= sO[0][1][12][txm32];
            r -= sO[1][0][4][txm32];
            r += sO[1][0][5][txm32];
            r += sO[1][0][6][txm32];
            r += sO[1][0][8][txm32];
            r -= sO[1][0][9][txm32];
            r -= sO[1][0][10][txm32];
            r -= sO[1][0][12][txm32];
            r += sO[1][0][13][txm32];
            r += sO[1][0][14][txm32];
            r += sO[1][1][4][txm32];
            r -= sO[1][1][8][txm32];
            r += sO[1][1][12][txm32];
        }
        else if (txb32 == 3) {
            r += sO[0][0][15][txm32];
            r -= sO[0][1][13][txm32];
            r += sO[0][1][14][txm32];
            r -= sO[0][1][15][txm32];
            r -= sO[1][0][7][txm32];
            r += sO[1][0][11][txm32];
            r -= sO[1][0][15][txm32];
            r += sO[1][1][5][txm32];
            r -= sO[1][1][6][txm32];
            r += sO[1][1][7][txm32];
            r -= sO[1][1][9][txm32];
            r += sO[1][1][10][txm32];
            r -= sO[1][1][11][txm32];
            r += sO[1][1][13][txm32];
            r -= sO[1][1][14][txm32];
            r += sO[1][1][15][txm32];
        }

        img[(hb + txb32b2 - padding) * WC + (wb + txb32m2 - padding) * C + cb + txm32] = r;// / 16 / N;
    }

}

__global__ void conv2d_wino_4_3_di_2x2x16(float* img, float* out, int H, int W, int C, int ldh, int ldw, int ldc, int padding) {

    __shared__ float sO[2][2][16][16];

    int tx = threadIdx.x;
    int cb = blockIdx.x * 16;
    int hb = blockIdx.y * 2;
    int wb = blockIdx.z * 2;

    int nhb = (hb + padding) / 2 - 1;
    int nwb = (wb + padding) / 2 - 1;

    int C16 = ldc * 16;
    int NWC16 = ldw * C16;
    int WC = W * C;

    int txm8 = tx % 8;
    int txb8 = tx / 8;
    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txb16m2 = txb16 % 2;
    int txb16b2 = txb16 / 2;

    int h, w;

    if (cb + txm16 < C) {
#pragma unroll
        for (h = 0; h < 2; h++) {
#pragma unroll
            for (w = 0; w < 2; w++) {
                if (nhb + h >= 0 && nhb + h < ldh && nwb + w >= 0 && nwb + w < ldw) {
                    sO[h][w][txb16 * 4 + 0][txm16] = out[(nhb + h) * NWC16 + (nwb + w) * C16 + (txb16 * 4 + 0) * ldc + cb + txm16];
                    sO[h][w][txb16 * 4 + 1][txm16] = out[(nhb + h) * NWC16 + (nwb + w) * C16 + (txb16 * 4 + 1) * ldc + cb + txm16];
                    sO[h][w][txb16 * 4 + 2][txm16] = out[(nhb + h) * NWC16 + (nwb + w) * C16 + (txb16 * 4 + 2) * ldc + cb + txm16];
                    sO[h][w][txb16 * 4 + 3][txm16] = out[(nhb + h) * NWC16 + (nwb + w) * C16 + (txb16 * 4 + 3) * ldc + cb + txm16];
                }
                else {
                    sO[h][w][txb16 * 4 + 0][txm16] = 0;
                    sO[h][w][txb16 * 4 + 1][txm16] = 0;
                    sO[h][w][txb16 * 4 + 2][txm16] = 0;
                    sO[h][w][txb16 * 4 + 3][txm16] = 0;
                }
            }
        }
    }

    __syncthreads();

    if (cb + txm16 < C && hb + txb16b2 - padding >= 0 && hb + txb16b2 - padding < H && wb + txb16m2 - padding >= 0 && wb + txb16m2 - padding < W) {
        float r = 0;

        if (txb16 == 0) {
            r += sO[0][0][0][txm16];
            r -= sO[0][0][1][txm16];
            r -= sO[0][0][2][txm16];
            r -= sO[0][0][4][txm16];
            r += sO[0][0][5][txm16];
            r += sO[0][0][6][txm16];
            r -= sO[0][0][8][txm16];
            r += sO[0][0][9][txm16];
            r += sO[0][0][10][txm16];
            r += sO[0][1][4][txm16];
            r += sO[0][1][8][txm16];
            r -= sO[1][0][0][txm16];
            r += sO[1][0][1][txm16];
            r += sO[1][0][2][txm16];
            r += sO[1][1][0][txm16];
        }
        else if (txb16 == 1) {
            r += sO[0][0][3][txm16];
            r -= sO[0][0][7][txm16];
            r -= sO[0][0][11][txm16];
            r -= sO[0][1][1][txm16];
            r += sO[0][1][2][txm16];
            r -= sO[0][1][3][txm16];
            r += sO[0][1][5][txm16];
            r -= sO[0][1][6][txm16];
            r += sO[0][1][7][txm16];
            r += sO[0][1][9][txm16];
            r -= sO[0][1][10][txm16];
            r += sO[0][1][11][txm16];
            r -= sO[1][0][3][txm16];
            r += sO[1][1][1][txm16];
            r -= sO[1][1][2][txm16];
            r += sO[1][1][3][txm16];
        }
        else if (txb16 == 2) {
            r += sO[0][0][12][txm16];
            r -= sO[0][0][13][txm16];
            r -= sO[0][0][14][txm16];
            r -= sO[0][1][12][txm16];
            r -= sO[1][0][4][txm16];
            r += sO[1][0][5][txm16];
            r += sO[1][0][6][txm16];
            r += sO[1][0][8][txm16];
            r -= sO[1][0][9][txm16];
            r -= sO[1][0][10][txm16];
            r -= sO[1][0][12][txm16];
            r += sO[1][0][13][txm16];
            r += sO[1][0][14][txm16];
            r += sO[1][1][4][txm16];
            r -= sO[1][1][8][txm16];
            r += sO[1][1][12][txm16];
        }
        else if (txb16 == 3) {
            r += sO[0][0][15][txm16];
            r -= sO[0][1][13][txm16];
            r += sO[0][1][14][txm16];
            r -= sO[0][1][15][txm16];
            r -= sO[1][0][7][txm16];
            r += sO[1][0][11][txm16];
            r -= sO[1][0][15][txm16];
            r += sO[1][1][5][txm16];
            r -= sO[1][1][6][txm16];
            r += sO[1][1][7][txm16];
            r -= sO[1][1][9][txm16];
            r += sO[1][1][10][txm16];
            r -= sO[1][1][11][txm16];
            r += sO[1][1][13][txm16];
            r -= sO[1][1][14][txm16];
            r += sO[1][1][15][txm16];
        }

        img[(hb + txb16b2 - padding) * WC + (wb + txb16m2 - padding) * C + cb + txm16] = r;// / 16 / N;
    }

}


__global__ void conv2d_wino_4_3_dk_16x16(float* kern, float* out, int N, int C, int ldc, int ldn) {

    int tx = threadIdx.x;
    int nb = blockIdx.y * 16;
    int cb = blockIdx.x * 16;

    int txm16 = tx % 16;
    int txb16 = tx / 16;

    int NC = N * C;
    int KyNC = 3 * NC;
    int C16 = ldc * 16;

    if (nb + txb16 < N && cb + txm16 < C) {

        float t;

        float o00 = out[(nb + txb16) * C16 + 0 * ldc + cb + txm16];
        float o01 = out[(nb + txb16) * C16 + 1 * ldc + cb + txm16];
        float o02 = out[(nb + txb16) * C16 + 2 * ldc + cb + txm16];
        float o03 = out[(nb + txb16) * C16 + 3 * ldc + cb + txm16];
        float o10 = out[(nb + txb16) * C16 + 4 * ldc + cb + txm16];
        float o11 = out[(nb + txb16) * C16 + 5 * ldc + cb + txm16];
        float o12 = out[(nb + txb16) * C16 + 6 * ldc + cb + txm16];
        float o13 = out[(nb + txb16) * C16 + 7 * ldc + cb + txm16];
        float o20 = out[(nb + txb16) * C16 + 8 * ldc + cb + txm16];
        float o21 = out[(nb + txb16) * C16 + 9 * ldc + cb + txm16];
        float o22 = out[(nb + txb16) * C16 + 10 * ldc + cb + txm16];
        float o23 = out[(nb + txb16) * C16 + 11 * ldc + cb + txm16];
        float o30 = out[(nb + txb16) * C16 + 12 * ldc + cb + txm16];
        float o31 = out[(nb + txb16) * C16 + 13 * ldc + cb + txm16];
        float o32 = out[(nb + txb16) * C16 + 14 * ldc + cb + txm16];
        float o33 = out[(nb + txb16) * C16 + 15 * ldc + cb + txm16];

        t = o00 + 0.5 * (o01 + o02 + o10 + o20) + 0.25 * (o11 + o12 + o21 + o22);
        kern[0 * KyNC + 0 * NC + (nb + txb16) * C + cb + txm16] = t;

        t = 0.5 * (o01 - o02) + 0.25 * (o11 - o12 + o21 - o22);
        kern[0 * KyNC + 1 * NC + (nb + txb16) * C + cb + txm16] = t;

        t = 0.5 * (o01 + o02 + o13 + o23) + o03 + 0.25 * (o11 + o12 + o21 + o22);
        kern[0 * KyNC + 2 * NC + (nb + txb16) * C + cb + txm16] = t;

        t = 0.5 * (o10 - o20) + 0.25 * (o11 + o12 - o21 - o22);
        kern[1 * KyNC + 0 * NC + (nb + txb16) * C + cb + txm16] = t;

        t = 0.25 * (o11 - o12 - o21 + o22);
        kern[1 * KyNC + 1 * NC + (nb + txb16) * C + cb + txm16] = t;

        t = 0.5 * (o13 - o23) + 0.25 * (o11 + o12 - o21 - o22);
        kern[1 * KyNC + 2 * NC + (nb + txb16) * C + cb + txm16] = t;

        t = o30 + 0.5 * (o10 + o20 + o31 + o32) + 0.25 * (o11 + o12 + o21 + o22);
        kern[2 * KyNC + 0 * NC + (nb + txb16) * C + cb + txm16] = t;

        t = 0.5 * (o31 - o32) + 0.25 * (o11 - o12 + o21 - o22);
        kern[2 * KyNC + 1 * NC + (nb + txb16) * C + cb + txm16] = t;

        t = o33 + 0.5 * (o13 + o23 + o31 + o32) + 0.25 * (o11 + o12 + o21 + o22);
        kern[2 * KyNC + 2 * NC + (nb + txb16) * C + cb + txm16] = t;
    }
}


__global__ void conv2d_wino_4_3_do_2x2x32(float* res, float* out, int H, int W, int N, int ldh, int ldw, int ldn) {

    // res is WIxWIx16xN

    int tx = threadIdx.x;

    int nb = blockIdx.x * 32;
    int hb = blockIdx.y * 2;
    int wb = blockIdx.z * 2;

    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txb32m2 = txb32 % 2;
    int txb32b2 = txb32 / 2;

    int N16 = N * 16;
    int TW16N = W * N16;
    int NWN = ldw * ldn;

    float o1, o2, o3, o4;
    o1 = o2 = o3 = o4 = 0;
    if ((hb + txb32b2) * 2 < ldh && (wb + txb32m2) * 2 < ldw && (nb + txm32) < ldn) {
        o1 = out[((hb + txb32b2) * 2 + 0) * NWN + ((wb + txb32m2) * 2 + 0) * ldn + nb + txm32];
        o2 = out[((hb + txb32b2) * 2 + 0) * NWN + ((wb + txb32m2) * 2 + 1) * ldn + nb + txm32];
        o3 = out[((hb + txb32b2) * 2 + 1) * NWN + ((wb + txb32m2) * 2 + 0) * ldn + nb + txm32];
        o4 = out[((hb + txb32b2) * 2 + 1) * NWN + ((wb + txb32m2) * 2 + 1) * ldn + nb + txm32];
    }

    float* a = res + (hb + txb32b2) * TW16N + (wb + txb32m2) * N16 + 0 * N + nb + txm32;
    float* b = res + (hb + txb32b2) * TW16N + (wb + txb32m2) * N16 + 1 * N + nb + txm32;
    float* c = res + (hb + txb32b2) * TW16N + (wb + txb32m2) * N16 + 2 * N + nb + txm32;
    float* d = res + (hb + txb32b2) * TW16N + (wb + txb32m2) * N16 + 3 * N + nb + txm32;
    float* e = res + (hb + txb32b2) * TW16N + (wb + txb32m2) * N16 + 4 * N + nb + txm32;
    float* f = res + (hb + txb32b2) * TW16N + (wb + txb32m2) * N16 + 5 * N + nb + txm32;
    float* g = res + (hb + txb32b2) * TW16N + (wb + txb32m2) * N16 + 6 * N + nb + txm32;
    float* h = res + (hb + txb32b2) * TW16N + (wb + txb32m2) * N16 + 7 * N + nb + txm32;
    float* i = res + (hb + txb32b2) * TW16N + (wb + txb32m2) * N16 + 8 * N + nb + txm32;
    float* j = res + (hb + txb32b2) * TW16N + (wb + txb32m2) * N16 + 9 * N + nb + txm32;
    float* k = res + (hb + txb32b2) * TW16N + (wb + txb32m2) * N16 + 10 * N + nb + txm32;
    float* l = res + (hb + txb32b2) * TW16N + (wb + txb32m2) * N16 + 11 * N + nb + txm32;
    float* m = res + (hb + txb32b2) * TW16N + (wb + txb32m2) * N16 + 12 * N + nb + txm32;
    float* n = res + (hb + txb32b2) * TW16N + (wb + txb32m2) * N16 + 13 * N + nb + txm32;
    float* o = res + (hb + txb32b2) * TW16N + (wb + txb32m2) * N16 + 14 * N + nb + txm32;
    float* p = res + (hb + txb32b2) * TW16N + (wb + txb32m2) * N16 + 15 * N + nb + txm32;

    *a = o1;
    *b = o1 + o2;
    *c = o1 - o2;
    *d = -o2;
    *e = o1 + o3;
    *f = o1 + o2 + o3 + o4;
    *g = o1 - o2 + o3 - o4;
    *h = -o2 - o4;
    *i = o1 - o3;
    *j = o1 + o2 - o3 - o4;
    *k = o1 - o2 - o3 + o4;
    *l = -o2 + o4;
    *m = -o3;
    *n = -o3 - o4;
    *o = -o3 + o4;
    *p = o4;
}

__global__ void conv2d_wino_4_3_do_2x2x16(float* res, float* out, int H, int W, int N, int ldh, int ldw, int ldn) {

    // res is WIxWIx16xN

    int tx = threadIdx.x;

    int nb = blockIdx.x * 16;
    int hb = blockIdx.y * 2;
    int wb = blockIdx.z * 2;

    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txb16m2 = txb16 % 2;
    int txb16b2 = txb16 / 2;

    int N16 = N * 16;
    int TW16N = ((W - 4) / 2 + 1) * N16;
    int NWN = ldw * ldn;

    float o1, o2, o3, o4;
    o1 = o2 = o3 = o4 = 0;
    if ((hb + txb16b2) * 2 < ldh && (wb + txb16m2) * 2 < ldw && (nb + txm16) < ldn) {
        o1 = out[((hb + txb16b2) * 2 + 0) * NWN + ((wb + txb16m2) * 2 + 0) * ldn + nb + txm16];
        o2 = out[((hb + txb16b2) * 2 + 0) * NWN + ((wb + txb16m2) * 2 + 1) * ldn + nb + txm16];
        o3 = out[((hb + txb16b2) * 2 + 1) * NWN + ((wb + txb16m2) * 2 + 0) * ldn + nb + txm16];
        o4 = out[((hb + txb16b2) * 2 + 1) * NWN + ((wb + txb16m2) * 2 + 1) * ldn + nb + txm16];
    }

    float* a = res + (hb + txb16b2) * TW16N + (wb + txb16m2) * N16 + 0 * N + nb + txm16;
    float* b = res + (hb + txb16b2) * TW16N + (wb + txb16m2) * N16 + 1 * N + nb + txm16;
    float* c = res + (hb + txb16b2) * TW16N + (wb + txb16m2) * N16 + 2 * N + nb + txm16;
    float* d = res + (hb + txb16b2) * TW16N + (wb + txb16m2) * N16 + 3 * N + nb + txm16;
    float* e = res + (hb + txb16b2) * TW16N + (wb + txb16m2) * N16 + 4 * N + nb + txm16;
    float* f = res + (hb + txb16b2) * TW16N + (wb + txb16m2) * N16 + 5 * N + nb + txm16;
    float* g = res + (hb + txb16b2) * TW16N + (wb + txb16m2) * N16 + 6 * N + nb + txm16;
    float* h = res + (hb + txb16b2) * TW16N + (wb + txb16m2) * N16 + 7 * N + nb + txm16;
    float* i = res + (hb + txb16b2) * TW16N + (wb + txb16m2) * N16 + 8 * N + nb + txm16;
    float* j = res + (hb + txb16b2) * TW16N + (wb + txb16m2) * N16 + 9 * N + nb + txm16;
    float* k = res + (hb + txb16b2) * TW16N + (wb + txb16m2) * N16 + 10 * N + nb + txm16;
    float* l = res + (hb + txb16b2) * TW16N + (wb + txb16m2) * N16 + 11 * N + nb + txm16;
    float* m = res + (hb + txb16b2) * TW16N + (wb + txb16m2) * N16 + 12 * N + nb + txm16;
    float* n = res + (hb + txb16b2) * TW16N + (wb + txb16m2) * N16 + 13 * N + nb + txm16;
    float* o = res + (hb + txb16b2) * TW16N + (wb + txb16m2) * N16 + 14 * N + nb + txm16;
    float* p = res + (hb + txb16b2) * TW16N + (wb + txb16m2) * N16 + 15 * N + nb + txm16;

    *a = o1;
    *b = o1 + o2;
    *c = o1 - o2;
    *d = -o2;
    *e = o1 + o3;
    *f = o1 + o2 + o3 + o4;
    *g = o1 - o2 + o3 - o4;
    *h = -o2 - o4;
    *i = o1 - o3;
    *j = o1 + o2 - o3 - o4;
    *k = o1 - o2 - o3 + o4;
    *l = -o2 + o4;
    *m = -o3;
    *n = -o3 - o4;
    *o = -o3 + o4;
    *p = o4;
}




DLLEXPORT void cuda_conv2d_wino_i_transform_3x3(float* dimg, float* douti, int H, int W, int C, int ldw, int ldc, int padding) {

    int TH = (int)ceil(((float)H + 2 * padding - 4) / 2 + 1);
    int TW = (int)ceil(((float)W + 2 * padding - 4) / 2 + 1);

    if (C >= 32) {
        dim3 gridDimsi((int)ceil((float)C / 32), (int)ceil((float)TH / 4), (int)ceil((float)TW / 4));
        dim3 blockDimsi(128, 1, 1);
        conv2d_wino_4_3_i_4x4x32 << <gridDimsi, blockDimsi >> > (dimg, douti, H, W, C, ldw, ldc, padding);
    }
    else {
        dim3 gridDimsi((int)ceil((float)C / 16), (int)ceil((float)TH / 4), (int)ceil((float)TW / 4));
        dim3 blockDimsi(64, 1, 1);
        conv2d_wino_4_3_i_4x4x16 << <gridDimsi, blockDimsi >> > (dimg, douti, H, W, C, ldw, ldc, padding);
    }
}

DLLEXPORT void cuda_conv2d_wino_k_transform_3x3(float* dkern, float* doutk, int C, int N, int ldc) {

    dim3 gridDimsk((int)ceil((float)C / 16), (int)ceil((float)N / 16), 1);
    dim3 blockDimsk(256, 1, 1);
    conv2d_wino_4_3_k_16x16 << <gridDimsk, blockDimsk >> > (dkern, doutk, N, C, ldc);
}

DLLEXPORT void cuda_conv2d_wino_o_transform_3x3(float* dres, float* dout, int H, int W, int N, int ldh, int ldw, int ldn) {

    if (N >= 32) {
        dim3 gridDimsk((int)ceil((float)N / 32), (int)ceil((float)H / 2), (int)ceil((float)W / 2));
        dim3 blockDimsk(128, 1, 1);
        conv2d_wino_4_3_o_2x2x32 << <gridDimsk, blockDimsk >> > (dres, dout, H, W, N, ldh, ldw, ldn);
    }
    else {
        dim3 gridDimsk((int)ceil((float)N / 16), (int)ceil((float)H / 2), (int)ceil((float)W / 2));
        dim3 blockDimsk(64, 1, 1);
        conv2d_wino_4_3_o_2x2x16 << <gridDimsk, blockDimsk >> > (dres, dout, H, W, N, ldh, ldw, ldn);
    }
}



DLLEXPORT void cuda_conv2d_der_wino_i_transform_3x3(float* dimg, float* douti, int H, int W, int C, int ldh, int ldw, int ldc, int padding) {

    if (C >= 32) {
        dim3 gridDimsi((int)ceil((float)C / 32), (int)ceil((float)H / 2), (int)ceil((float)W / 2));
        dim3 blockDimsi(128, 1, 1);
        conv2d_wino_4_3_di_2x2x32 << <gridDimsi, blockDimsi >> > (dimg, douti, H, W, C, ldh, ldw, ldc, padding);
    }
    else {
        dim3 gridDimsi((int)ceil((float)C / 16), (int)ceil((float)H / 2), (int)ceil((float)W / 2));
        dim3 blockDimsi(64, 1, 1);
        conv2d_wino_4_3_di_2x2x16 << <gridDimsi, blockDimsi >> > (dimg, douti, H, W, C, ldh, ldw, ldc, padding);
    }
}

DLLEXPORT void cuda_conv2d_der_wino_k_transform_3x3(float* dkern, float* doutk, int C, int N, int ldc, int ldn) {

    dim3 gridDimsk((int)ceil((float)C / 16), (int)ceil((float)N / 16), 1);
    dim3 blockDimsk(256, 1, 1);
    conv2d_wino_4_3_dk_16x16 << <gridDimsk, blockDimsk >> > (dkern, doutk, N, C, ldc, ldn);
}

DLLEXPORT void cuda_conv2d_der_wino_o_transform_3x3(float* dres, float* dout, int H, int W, int N, int ldh, int ldw, int ldn) {

    if (N >= 32) {
        dim3 gridDimsk((int)ceil((float)N / 32), (int)ceil((float)H / 2), (int)ceil((float)W / 2));
        dim3 blockDimsk(128, 1, 1);
        conv2d_wino_4_3_do_2x2x32 << <gridDimsk, blockDimsk >> > (dres, dout, H, W, N, ldh, ldw, ldn);
    }
    else {
        dim3 gridDimsk((int)ceil((float)N / 16), (int)ceil((float)H / 2), (int)ceil((float)W / 2));
        dim3 blockDimsk(64, 1, 1);
        conv2d_wino_4_3_do_2x2x16 << <gridDimsk, blockDimsk >> > (dres, dout, H, W, N, ldh, ldw, ldn);
    }
}
