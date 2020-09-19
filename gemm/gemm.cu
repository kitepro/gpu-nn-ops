#include<time.h>
#include<iostream>

#define DLLEXPORT extern "C" __declspec(dllexport)

#define FULL_MASK 0xFFFFFFFF

struct float8 {
    float a, b, c, d, e, f, g, h;
};

__global__ void sgemm_128x128x8_NN(float* A, float* B, float* C, int M, int N, int K, int aligned) {

    __shared__ __align__(16) float sA[8][132];
    __shared__ __align__(16) float sB[8][128];
    __shared__ __align__(16) float sAb[8][132];
    __shared__ __align__(16) float sBb[8][128];

    int tx = threadIdx.x;
    int mb = blockIdx.z * 256 + (blockIdx.x / 2) * 128;
    int nb = blockIdx.y * 256 + (blockIdx.x % 2) * 128;

    if (mb >= M || nb >= N) {
        return;
    }

    float rC[8][8];
    float4 rA;
    float4 rB;
    float rAs[8];
    float rBs[8];
    float rAsb[8];
    float rBsb[8];

    float(*psA)[8][132] = &sA;
    float(*psB)[8][128] = &sB;

    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm2 = tx % 2;
    int txb2 = tx / 2;
    int txm8 = tx % 8;
    int txb8 = tx / 8;
    int txm128 = tx % 128;
    int txb128 = tx / 128;
    int txm32b4 = txm32 / 4;

    int wnb = (txb32 / 4) * 64;
    int wmb = (txb32 % 4) * 32;

    int m, n, k, kb, kbb;

    float4 f4;
    float8 f8;

    int sAi1 = wmb + (txm32 % 2) * 8 + (txm32 / 16) * 16;
    int sBi1 = wnb + (txm32 / 2) * 4;
    int sBi2 = wnb + (32 + (txm32 / 2) * 4) % 64;

#pragma unroll
    for (m = 0; m < 8; m++) {
#pragma unroll
        for (n = 0; n < 8; n++) {
            rC[m][n] = 0;
        }
    }

    if (aligned == 1) {
        if (mb + txm128 < M && txb128 * 4 < K) {
            rA = *reinterpret_cast<float4*>(A + (mb + txm128) * K + txb128 * 4);
        }
        else {
            rA.x = rA.y = rA.z = rA.w = 0;
        }
        if (nb + txm32 * 4 < N && txb32 < K) {
            rB = *reinterpret_cast<float4*>(B + txb32 * N + nb + txm32 * 4);
        }
        else {
            rB.x = rB.y = rB.z = rB.w = 0;
        }
    }
    else {
        if (mb + txm128 < M) {
            int t = K - txb128 * 4;
            if (4 <= t) {
                rA.x = A[(mb + txm128) * K + txb128 * 4 + 0];
                rA.y = A[(mb + txm128) * K + txb128 * 4 + 1];
                rA.z = A[(mb + txm128) * K + txb128 * 4 + 2];
                rA.w = A[(mb + txm128) * K + txb128 * 4 + 3];
            }
            else if (3 <= t) {
                rA.x = A[(mb + txm128) * K + txb128 * 4 + 0];
                rA.y = A[(mb + txm128) * K + txb128 * 4 + 1];
                rA.z = A[(mb + txm128) * K + txb128 * 4 + 2];
                rA.w = 0;
            }
            else if (2 <= t) {
                rA.x = A[(mb + txm128) * K + txb128 * 4 + 0];
                rA.y = A[(mb + txm128) * K + txb128 * 4 + 1];
                rA.z = rA.w = 0;
            }
            else if (1 <= t) {
                rA.x = A[(mb + txm128) * K + txb128 * 4 + 0];
                rA.y = rA.z = rA.w = 0;
            }
            else {
                rA.x = rA.y = rA.z = rA.w = 0;
            }
        }
        else {
            rA.x = rA.y = rA.z = rA.w = 0;
        }
        if (txb32 < K) {
            int t = N - nb - txm32 * 4;
            if (4 <= t) {
                rB.x = B[txb32 * N + nb + txm32 * 4 + 0];
                rB.y = B[txb32 * N + nb + txm32 * 4 + 1];
                rB.z = B[txb32 * N + nb + txm32 * 4 + 2];
                rB.w = B[txb32 * N + nb + txm32 * 4 + 3];
            }
            else if (3 <= t) {
                rB.x = B[txb32 * N + nb + txm32 * 4 + 0];
                rB.y = B[txb32 * N + nb + txm32 * 4 + 1];
                rB.z = B[txb32 * N + nb + txm32 * 4 + 2];
                rB.w = 0;
            }
            else if (2 <= t) {
                rB.x = B[txb32 * N + nb + txm32 * 4 + 0];
                rB.y = B[txb32 * N + nb + txm32 * 4 + 1];
                rB.z = rB.w = 0;
            }
            else if (1 <= t) {
                rB.x = B[txb32 * N + nb + txm32 * 4 + 0];
                rB.y = rB.z = rB.w = 0;
            }
            else {
                rB.x = rB.y = rB.z = rB.w = 0;
            }
        }
        else {
            rB.x = rB.y = rB.z = rB.w = 0;
        }
    }

    for (kb = 0; kb < K; kb += 8) {

        (*psA)[txb128 * 4 + 0][txm128] = rA.x;
        (*psA)[txb128 * 4 + 1][txm128] = rA.y;
        (*psA)[txb128 * 4 + 2][txm128] = rA.z;
        (*psA)[txb128 * 4 + 3][txm128] = rA.w;
        (*psB)[txb32][txm32 * 4 + 0] = rB.x;
        (*psB)[txb32][txm32 * 4 + 1] = rB.y;
        (*psB)[txb32][txm32 * 4 + 2] = rB.z;
        (*psB)[txb32][txm32 * 4 + 3] = rB.w;

        __syncthreads();

        if (kb + 8 < K) {
            if (aligned == 1) {
                if (mb + txm128 < M && kb + 8 + txb128 * 4 < K) {
                    rA = *reinterpret_cast<float4*>(A + (mb + txm128) * K + kb + 8 + txb128 * 4);
                }
                else {
                    rA.x = rA.y = rA.z = rA.w = 0;
                }
                if (nb + txm32 * 4 < N && kb + 8 + txb32 < K) {
                    rB = *reinterpret_cast<float4*>(B + (kb + 8 + txb32) * N + nb + txm32 * 4);
                }
                else {
                    rB.x = rB.y = rB.z = rB.w = 0;
                }
            }
            else {
                if (mb + txm128 < M) {
                    int t = K - kb - 8 - txb128 * 4;
                    if (4 <= t) {
                        rA.x = A[(mb + txm128) * K + kb + 8 + txb128 * 4 + 0];
                        rA.y = A[(mb + txm128) * K + kb + 8 + txb128 * 4 + 1];
                        rA.z = A[(mb + txm128) * K + kb + 8 + txb128 * 4 + 2];
                        rA.w = A[(mb + txm128) * K + kb + 8 + txb128 * 4 + 3];
                    }
                    else if (3 <= t) {
                        rA.x = A[(mb + txm128) * K + kb + 8 + txb128 * 4 + 0];
                        rA.y = A[(mb + txm128) * K + kb + 8 + txb128 * 4 + 1];
                        rA.z = A[(mb + txm128) * K + kb + 8 + txb128 * 4 + 2];
                        rA.w = 0;
                    }
                    else if (2 <= t) {
                        rA.x = A[(mb + txm128) * K + kb + 8 + txb128 * 4 + 0];
                        rA.y = A[(mb + txm128) * K + kb + 8 + txb128 * 4 + 1];
                        rA.z = rA.w = 0;
                    }
                    else if (1 <= t) {
                        rA.x = A[(mb + txm128) * K + kb + 8 + txb128 * 4 + 0];
                        rA.y = rA.z = rA.w = 0;
                    }
                    else {
                        rA.x = rA.y = rA.z = rA.w = 0;
                    }
                }
                else {
                    rA.x = rA.y = rA.z = rA.w = 0;
                }
                if (kb + 8 + txb32 < K) {
                    int t = N - nb - txm32 * 4;
                    if (4 <= t) {
                        rB.x = B[(kb + 8 + txb32) * N + nb + txm32 * 4 + 0];
                        rB.y = B[(kb + 8 + txb32) * N + nb + txm32 * 4 + 1];
                        rB.z = B[(kb + 8 + txb32) * N + nb + txm32 * 4 + 2];
                        rB.w = B[(kb + 8 + txb32) * N + nb + txm32 * 4 + 3];
                    }
                    else if (3 <= t) {
                        rB.x = B[(kb + 8 + txb32) * N + nb + txm32 * 4 + 0];
                        rB.y = B[(kb + 8 + txb32) * N + nb + txm32 * 4 + 1];
                        rB.z = B[(kb + 8 + txb32) * N + nb + txm32 * 4 + 2];
                        rB.w = 0;
                    }
                    else if (2 <= t) {
                        rB.x = B[(kb + 8 + txb32) * N + nb + txm32 * 4 + 0];
                        rB.y = B[(kb + 8 + txb32) * N + nb + txm32 * 4 + 1];
                        rB.z = rB.w = 0;
                    }
                    else if (1 <= t) {
                        rB.x = B[(kb + 8 + txb32) * N + nb + txm32 * 4 + 0];
                        rB.y = rB.z = rB.w = 0;
                    }
                    else {
                        rB.x = rB.y = rB.z = rB.w = 0;
                    }
                }
                else {
                    rB.x = rB.y = rB.z = rB.w = 0;
                }
            }
        }

        // COMPUTE -------------------

        f8 = *reinterpret_cast<float8*>((*psA)[0] + sAi1);
        rAs[0] = f8.a;
        rAs[1] = f8.b;
        rAs[2] = f8.c;
        rAs[3] = f8.d;
        rAs[4] = f8.e;
        rAs[5] = f8.f;
        rAs[6] = f8.g;
        rAs[7] = f8.h;
        f4 = *reinterpret_cast<float4*>((*psB)[0] + sBi1);
        rBs[0] = f4.x;
        rBs[1] = f4.y;
        rBs[2] = f4.z;
        rBs[3] = f4.w;
        f4 = *reinterpret_cast<float4*>((*psB)[0] + sBi2);
        rBs[4] = f4.x;
        rBs[5] = f4.y;
        rBs[6] = f4.z;
        rBs[7] = f4.w;

#pragma unroll
        for (k = 0; k < 8; k++) {

            if (k < 7) {
                f8 = *reinterpret_cast<float8*>((*psA)[k + 1] + sAi1);
                rAsb[0] = f8.a;
                rAsb[1] = f8.b;
                rAsb[2] = f8.c;
                rAsb[3] = f8.d;
                rAsb[4] = f8.e;
                rAsb[5] = f8.f;
                rAsb[6] = f8.g;
                rAsb[7] = f8.h;
                f4 = *reinterpret_cast<float4*>((*psB)[k + 1] + sBi1);
                rBsb[0] = f4.x;
                rBsb[1] = f4.y;
                rBsb[2] = f4.z;
                rBsb[3] = f4.w;
                f4 = *reinterpret_cast<float4*>((*psB)[k + 1] + sBi2);
                rBsb[4] = f4.x;
                rBsb[5] = f4.y;
                rBsb[6] = f4.z;
                rBsb[7] = f4.w;
            }

#pragma unroll
            for (m = 0; m < 8; m++) {
#pragma unroll
                for (n = 0; n < 8; n++) {
                    rC[m][n] += rAs[m] * rBs[n];
                }
            }

#pragma unroll
            for (m = 0; m < 8; m++) {
                rAs[m] = rAsb[m];
                rBs[m] = rBsb[m];
            }

        }

        // ---------------------------

        if (psA == &sA) {
            psA = &sAb;
            psB = &sBb;
        }
        else {
            psA = &sA;
            psB = &sB;
        }

    }

#pragma unroll
    for (m = 0; m < 8; m++) {
        float4 t;
        if (aligned) {
            if (mb + sAi1 + m < M) {
                if (nb + sBi1 < N) {
                    t.x = rC[m][0];
                    t.y = rC[m][1];
                    t.z = rC[m][2];
                    t.w = rC[m][3];
                    *reinterpret_cast<float4*>(C + (mb + sAi1 + m) * N + nb + sBi1) = t;
                }
                if (nb + sBi2 < N) {
                    t.x = rC[m][4];
                    t.y = rC[m][5];
                    t.z = rC[m][6];
                    t.w = rC[m][7];
                    *reinterpret_cast<float4*>(C + (mb + sAi1 + m) * N + nb + sBi2) = t;
                }
            }
        }
        else {
            if (mb + sAi1 + m < M) {
                int t = N - nb - sBi1;
                if (4 <= t) {
                    C[(mb + sAi1 + m) * N + nb + sBi1 + 0] = rC[m][0];
                    C[(mb + sAi1 + m) * N + nb + sBi1 + 1] = rC[m][1];
                    C[(mb + sAi1 + m) * N + nb + sBi1 + 2] = rC[m][2];
                    C[(mb + sAi1 + m) * N + nb + sBi1 + 3] = rC[m][3];
                }
                else if (3 <= t) {
                    C[(mb + sAi1 + m) * N + nb + sBi1 + 0] = rC[m][0];
                    C[(mb + sAi1 + m) * N + nb + sBi1 + 1] = rC[m][1];
                    C[(mb + sAi1 + m) * N + nb + sBi1 + 2] = rC[m][2];
                }
                else if (2 <= t) {
                    C[(mb + sAi1 + m) * N + nb + sBi1 + 0] = rC[m][0];
                    C[(mb + sAi1 + m) * N + nb + sBi1 + 1] = rC[m][1];
                }
                else if (1 <= t) {
                    C[(mb + sAi1 + m) * N + nb + sBi1 + 0] = rC[m][0];
                }
                t = N - nb - sBi2;
                if (4 <= t) {
                    C[(mb + sAi1 + m) * N + nb + sBi2 + 0] = rC[m][4];
                    C[(mb + sAi1 + m) * N + nb + sBi2 + 1] = rC[m][5];
                    C[(mb + sAi1 + m) * N + nb + sBi2 + 2] = rC[m][6];
                    C[(mb + sAi1 + m) * N + nb + sBi2 + 3] = rC[m][7];
                }
                else if (3 <= t) {
                    C[(mb + sAi1 + m) * N + nb + sBi2 + 0] = rC[m][4];
                    C[(mb + sAi1 + m) * N + nb + sBi2 + 1] = rC[m][5];
                    C[(mb + sAi1 + m) * N + nb + sBi2 + 2] = rC[m][6];
                }
                else if (2 <= t) {
                    C[(mb + sAi1 + m) * N + nb + sBi2 + 0] = rC[m][4];
                    C[(mb + sAi1 + m) * N + nb + sBi2 + 1] = rC[m][5];
                }
                else if (1 <= t) {
                    C[(mb + sAi1 + m) * N + nb + sBi2 + 0] = rC[m][4];
                }
            }
        }
    }
}

__global__ void sgemm_64x64x16_NN(float* A, float* B, float* C, int M, int N, int K, int aligned) {

    __shared__ __align__(16) float sA[16][64];
    __shared__ __align__(16) float sB[16][64];
    __shared__ __align__(16) float sAb[16][64];
    __shared__ __align__(16) float sBb[16][64];

    int tx = threadIdx.x;
    int mb = blockIdx.z * 128 + (blockIdx.x / 2) * 64;
    int nb = blockIdx.y * 128 + (blockIdx.x % 2) * 64;

    if (mb >= M || nb >= N) {
        return;
    }

    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm64 = tx % 64;
    int txb64 = tx / 64;
    int txm8 = tx % 8;
    int txm32b8 = txm32 / 8;

    int wmb = (txb32 / 2) * 32;
    int wnb = (txb32 % 2) * 32;

    int kb, k, m, n;

    float4 rA[2];
    float4 rB[2];
    float rC[8][4];
    float rAs[8];
    float rBs[4];
    float rAsb[8];
    float rBsb[4];

    float(*psA)[16][64] = &sA;
    float(*psB)[16][64] = &sB;

#pragma unroll
    for (m = 0; m < 8; m++) {
#pragma unroll
        for (n = 0; n < 4; n++) {
            rC[m][n] = 0;
        }
    }

    if (aligned == 1) {
        if (txb32 * 4 < K) {
            if (mb + txm32 < M) {
                rA[0] = *reinterpret_cast<float4*>(A + (mb + txm32) * K + txb32 * 4);
            }
            else {
                rA[0].x = rA[0].y = rA[0].z = rA[0].w = 0;
            }
            if (mb + 32 + txm32 < M) {
                rA[1] = *reinterpret_cast<float4*>(A + (mb + 32 + txm32) * K + txb32 * 4);
            }
            else {
                rA[1].x = rA[1].y = rA[1].z = rA[1].w = 0;
            }
        }
        else {
            rA[0].x = rA[0].y = rA[0].z = rA[0].w = 0;
            rA[1].x = rA[1].y = rA[1].z = rA[1].w = 0;
        }

        if (nb + txm16 * 4 < N) {
            if (txb16 < K) {
                rB[0] = *reinterpret_cast<float4*>(B + txb16 * N + nb + txm16 * 4);
            }
            else {
                rB[0].x = rB[0].y = rB[0].z = rB[0].w = 0;
            }
            if (8 + txb16 < K) {
                rB[1] = *reinterpret_cast<float4*>(B + (8 + txb16) * N + nb + txm16 * 4);
            }
            else {
                rB[1].x = rB[1].y = rB[1].z = rB[1].w = 0;
            }
        }
        else {
            rB[0].x = rB[0].y = rB[0].z = rB[0].w = 0;
            rB[1].x = rB[1].y = rB[1].z = rB[1].w = 0;
        }
    }
    else {
        int t = K - txb32 * 4;
        if (mb + txm32) {
            if (4 <= t) {
                rA[0].x = A[(mb + txm32) * K + txb32 * 4 + 0];
                rA[0].y = A[(mb + txm32) * K + txb32 * 4 + 1];
                rA[0].z = A[(mb + txm32) * K + txb32 * 4 + 2];
                rA[0].w = A[(mb + txm32) * K + txb32 * 4 + 3];
            }
            else if (3 <= t) {
                rA[0].x = A[(mb + txm32) * K + txb32 * 4 + 0];
                rA[0].y = A[(mb + txm32) * K + txb32 * 4 + 1];
                rA[0].z = A[(mb + txm32) * K + txb32 * 4 + 2];
                rA[0].x = 0;
            }
            else if (2 <= t) {
                rA[0].x = A[(mb + txm32) * K + txb32 * 4 + 0];
                rA[0].y = A[(mb + txm32) * K + txb32 * 4 + 1];
                rA[0].z = rA[0].x = 0;
            }
            else if (1 <= t) {
                rA[0].x = A[(mb + txm32) * K + txb32 * 4 + 0];
                rA[0].y = rA[0].z = rA[0].x = 0;
            }
            else {
                rA[0].x = rA[0].y = rA[0].z = rA[0].w = 0;
            }
        }
        else {
            rA[0].x = rA[0].y = rA[0].z = rA[0].w = 0;
        }
        if (mb + 32 + txm32 < M) {
            if (4 <= t) {
                rA[1].x = A[(mb + 32 + txm32) * K + txb32 * 4 + 0];
                rA[1].y = A[(mb + 32 + txm32) * K + txb32 * 4 + 1];
                rA[1].z = A[(mb + 32 + txm32) * K + txb32 * 4 + 2];
                rA[1].w = A[(mb + 32 + txm32) * K + txb32 * 4 + 3];
            }
            else if (3 <= t) {
                rA[1].x = A[(mb + 32 + txm32) * K + txb32 * 4 + 0];
                rA[1].y = A[(mb + 32 + txm32) * K + txb32 * 4 + 1];
                rA[1].z = A[(mb + 32 + txm32) * K + txb32 * 4 + 2];
                rA[1].x = 0;
            }
            else if (2 <= t) {
                rA[1].x = A[(mb + 32 + txm32) * K + txb32 * 4 + 0];
                rA[1].y = A[(mb + 32 + txm32) * K + txb32 * 4 + 1];
                rA[1].z = rA[1].x = 0;
            }
            else if (1 <= t) {
                rA[1].x = A[(mb + 32 + txm32) * K + txb32 * 4 + 0];
                rA[1].y = rA[1].z = rA[1].x = 0;
            }
            else {
                rA[1].x = rA[1].y = rA[1].z = rA[1].w = 0;
            }
        }
        else {
            rA[1].x = rA[1].y = rA[1].z = rA[1].w = 0;
        }

        if (txb16 < K) {
            if (4 <= t) {
                rB[0].x = B[txb16 * N + nb + txm16 * 4 + 0];
                rB[0].y = B[txb16 * N + nb + txm16 * 4 + 1];
                rB[0].z = B[txb16 * N + nb + txm16 * 4 + 2];
                rB[0].w = B[txb16 * N + nb + txm16 * 4 + 3];
            }
            else if (3 <= t) {
                rB[0].x = B[txb16 * N + nb + txm16 * 4 + 0];
                rB[0].y = B[txb16 * N + nb + txm16 * 4 + 1];
                rB[0].z = B[txb16 * N + nb + txm16 * 4 + 2];
                rB[0].w = 0;
            }
            else if (2 <= t) {
                rB[0].x = B[txb16 * N + nb + txm16 * 4 + 0];
                rB[0].y = B[txb16 * N + nb + txm16 * 4 + 1];
                rB[0].z = rB[0].w = 0;
            }
            else if (1 <= t) {
                rB[0].x = B[txb16 * N + nb + txm16 * 4 + 0];
                rB[0].y = rB[0].z = rB[0].w = 0;
            }
            else {
                rB[0].x = rB[0].y = rB[0].z = rB[0].w = 0;
            }
        }
        else {
            rB[0].x = rB[0].y = rB[0].z = rB[0].w = 0;
        }
        t = N - nb - txm16 * 4;
        if (8 + txb16 < K) {
            if (4 <= t) {
                rB[1].x = B[(8 + txb16) * N + nb + txm16 * 4 + 0];
                rB[1].y = B[(8 + txb16) * N + nb + txm16 * 4 + 1];
                rB[1].z = B[(8 + txb16) * N + nb + txm16 * 4 + 2];
                rB[1].w = B[(8 + txb16) * N + nb + txm16 * 4 + 3];
            }
            else if (3 <= t) {
                rB[1].x = B[(8 + txb16) * N + nb + txm16 * 4 + 0];
                rB[1].y = B[(8 + txb16) * N + nb + txm16 * 4 + 1];
                rB[1].z = B[(8 + txb16) * N + nb + txm16 * 4 + 2];
                rB[1].w = 0;
            }
            else if (2 <= t) {
                rB[1].x = B[(8 + txb16) * N + nb + txm16 * 4 + 0];
                rB[1].y = B[(8 + txb16) * N + nb + txm16 * 4 + 1];
                rB[1].z = rB[1].w = 0;
            }
            else if (1 <= t) {
                rB[1].x = B[(8 + txb16) * N + nb + txm16 * 4 + 0];
                rB[1].y = rB[1].z = rB[1].w = 0;
            }
            else {
                rB[1].x = rB[1].y = rB[1].z = rB[1].w = 0;
            }
        }
        else {
            rB[1].x = rB[1].y = rB[1].z = rB[1].w = 0;
        }
    }

    for (kb = 0; kb < K; kb += 16) {

        (*psA)[txb32 * 4 + 0][txm32] = rA[0].x;
        (*psA)[txb32 * 4 + 1][txm32] = rA[0].y;
        (*psA)[txb32 * 4 + 2][txm32] = rA[0].z;
        (*psA)[txb32 * 4 + 3][txm32] = rA[0].w;
        (*psA)[txb32 * 4 + 0][32 + txm32] = rA[1].x;
        (*psA)[txb32 * 4 + 1][32 + txm32] = rA[1].y;
        (*psA)[txb32 * 4 + 2][32 + txm32] = rA[1].z;
        (*psA)[txb32 * 4 + 3][32 + txm32] = rA[1].w;
        (*psB)[txb16][txm16 * 4 + 0] = rB[0].x;
        (*psB)[txb16][txm16 * 4 + 1] = rB[0].y;
        (*psB)[txb16][txm16 * 4 + 2] = rB[0].z;
        (*psB)[txb16][txm16 * 4 + 3] = rB[0].w;
        (*psB)[8 + txb16][txm16 * 4 + 0] = rB[1].x;
        (*psB)[8 + txb16][txm16 * 4 + 1] = rB[1].y;
        (*psB)[8 + txb16][txm16 * 4 + 2] = rB[1].z;
        (*psB)[8 + txb16][txm16 * 4 + 3] = rB[1].w;

        __syncthreads();

        if (kb + 16 < K) {

            if (aligned == 1) {
                if (kb + 16 + txb32 * 4 < K) {
                    if (mb + txm32 < M) {
                        rA[0] = *reinterpret_cast<float4*>(A + (mb + txm32) * K + kb + 16 + txb32 * 4);
                    }
                    else {
                        rA[0].x = rA[0].y = rA[0].z = rA[0].w = 0;
                    }
                    if (mb + 32 + txm32 < M) {
                        rA[1] = *reinterpret_cast<float4*>(A + (mb + 32 + txm32) * K + kb + 16 + txb32 * 4);
                    }
                    else {
                        rA[1].x = rA[1].y = rA[1].z = rA[1].w = 0;
                    }
                }
                else {
                    rA[0].x = rA[0].y = rA[0].z = rA[0].w = 0;
                    rA[1].x = rA[1].y = rA[1].z = rA[1].w = 0;
                }

                if (nb + txm16 * 4 < N) {
                    if (kb + 16 + txb16 < K) {
                        rB[0] = *reinterpret_cast<float4*>(B + (kb + 16 + txb16) * N + nb + txm16 * 4);
                    }
                    else {
                        rB[0].x = rB[0].y = rB[0].z = rB[0].w = 0;
                    }
                    if (8 + txb16 < K) {
                        rB[1] = *reinterpret_cast<float4*>(B + (kb + 16 + 8 + txb16) * N + nb + txm16 * 4);
                    }
                    else {
                        rB[1].x = rB[1].y = rB[1].z = rB[1].w = 0;
                    }
                }
                else {
                    rB[0].x = rB[0].y = rB[0].z = rB[0].w = 0;
                    rB[1].x = rB[1].y = rB[1].z = rB[1].w = 0;
                }
            }
            else {
                int t = K - kb - 16 - txb32 * 4;
                if (mb + txm32) {
                    if (4 <= t) {
                        rA[0].x = A[(mb + txm32) * K + kb + 16 + txb32 * 4 + 0];
                        rA[0].y = A[(mb + txm32) * K + kb + 16 + txb32 * 4 + 1];
                        rA[0].z = A[(mb + txm32) * K + kb + 16 + txb32 * 4 + 2];
                        rA[0].w = A[(mb + txm32) * K + kb + 16 + txb32 * 4 + 3];
                    }
                    else if (3 <= t) {
                        rA[0].x = A[(mb + txm32) * K + kb + 16 + txb32 * 4 + 0];
                        rA[0].y = A[(mb + txm32) * K + kb + 16 + txb32 * 4 + 1];
                        rA[0].z = A[(mb + txm32) * K + kb + 16 + txb32 * 4 + 2];
                        rA[0].x = 0;
                    }
                    else if (2 <= t) {
                        rA[0].x = A[(mb + txm32) * K + kb + 16 + txb32 * 4 + 0];
                        rA[0].y = A[(mb + txm32) * K + kb + 16 + txb32 * 4 + 1];
                        rA[0].z = rA[0].x = 0;
                    }
                    else if (1 <= t) {
                        rA[0].x = A[(mb + txm32) * K + kb + 16 + txb32 * 4 + 0];
                        rA[0].y = rA[0].z = rA[0].x = 0;
                    }
                    else {
                        rA[0].x = rA[0].y = rA[0].z = rA[0].w = 0;
                    }
                }
                else {
                    rA[0].x = rA[0].y = rA[0].z = rA[0].w = 0;
                }
                if (mb + 32 + txm32 < M) {
                    if (4 <= t) {
                        rA[1].x = A[(mb + 32 + txm32) * K + kb + 16 + txb32 * 4 + 0];
                        rA[1].y = A[(mb + 32 + txm32) * K + kb + 16 + txb32 * 4 + 1];
                        rA[1].z = A[(mb + 32 + txm32) * K + kb + 16 + txb32 * 4 + 2];
                        rA[1].w = A[(mb + 32 + txm32) * K + kb + 16 + txb32 * 4 + 3];
                    }
                    else if (3 <= t) {
                        rA[1].x = A[(mb + 32 + txm32) * K + kb + 16 + txb32 * 4 + 0];
                        rA[1].y = A[(mb + 32 + txm32) * K + kb + 16 + txb32 * 4 + 1];
                        rA[1].z = A[(mb + 32 + txm32) * K + kb + 16 + txb32 * 4 + 2];
                        rA[1].x = 0;
                    }
                    else if (2 <= t) {
                        rA[1].x = A[(mb + 32 + txm32) * K + kb + 16 + txb32 * 4 + 0];
                        rA[1].y = A[(mb + 32 + txm32) * K + kb + 16 + txb32 * 4 + 1];
                        rA[1].z = rA[1].x = 0;
                    }
                    else if (1 <= t) {
                        rA[1].x = A[(mb + 32 + txm32) * K + kb + 16 + txb32 * 4 + 0];
                        rA[1].y = rA[1].z = rA[1].x = 0;
                    }
                    else {
                        rA[1].x = rA[1].y = rA[1].z = rA[1].w = 0;
                    }
                }
                else {
                    rA[1].x = rA[1].y = rA[1].z = rA[1].w = 0;
                }

                if (kb + 16 + txb16 < K) {
                    if (4 <= t) {
                        rB[0].x = B[(kb + 16 + txb16) * N + nb + txm16 * 4 + 0];
                        rB[0].y = B[(kb + 16 + txb16) * N + nb + txm16 * 4 + 1];
                        rB[0].z = B[(kb + 16 + txb16) * N + nb + txm16 * 4 + 2];
                        rB[0].w = B[(kb + 16 + txb16) * N + nb + txm16 * 4 + 3];
                    }
                    else if (3 <= t) {
                        rB[0].x = B[(kb + 16 + txb16) * N + nb + txm16 * 4 + 0];
                        rB[0].y = B[(kb + 16 + txb16) * N + nb + txm16 * 4 + 1];
                        rB[0].z = B[(kb + 16 + txb16) * N + nb + txm16 * 4 + 2];
                        rB[0].w = 0;
                    }
                    else if (2 <= t) {
                        rB[0].x = B[(kb + 16 + txb16) * N + nb + txm16 * 4 + 0];
                        rB[0].y = B[(kb + 16 + txb16) * N + nb + txm16 * 4 + 1];
                        rB[0].z = rB[0].w = 0;
                    }
                    else if (1 <= t) {
                        rB[0].x = B[(kb + 16 + txb16) * N + nb + txm16 * 4 + 0];
                        rB[0].y = rB[0].z = rB[0].w = 0;
                    }
                    else {
                        rB[0].x = rB[0].y = rB[0].z = rB[0].w = 0;
                    }
                }
                else {
                    rB[0].x = rB[0].y = rB[0].z = rB[0].w = 0;
                }
                t = N - nb - txm16 * 4;
                if (kb + 16 + 8 + txb16 < K) {
                    if (4 <= t) {
                        rB[1].x = B[(kb + 16 + 8 + txb16) * N + nb + txm16 * 4 + 0];
                        rB[1].y = B[(kb + 16 + 8 + txb16) * N + nb + txm16 * 4 + 1];
                        rB[1].z = B[(kb + 16 + 8 + txb16) * N + nb + txm16 * 4 + 2];
                        rB[1].w = B[(kb + 16 + 8 + txb16) * N + nb + txm16 * 4 + 3];
                    }
                    else if (3 <= t) {
                        rB[1].x = B[(kb + 16 + 8 + txb16) * N + nb + txm16 * 4 + 0];
                        rB[1].y = B[(kb + 16 + 8 + txb16) * N + nb + txm16 * 4 + 1];
                        rB[1].z = B[(kb + 16 + 8 + txb16) * N + nb + txm16 * 4 + 2];
                        rB[1].w = 0;
                    }
                    else if (2 <= t) {
                        rB[1].x = B[(kb + 16 + 8 + txb16) * N + nb + txm16 * 4 + 0];
                        rB[1].y = B[(kb + 16 + 8 + txb16) * N + nb + txm16 * 4 + 1];
                        rB[1].z = rB[1].w = 0;
                    }
                    else if (1 <= t) {
                        rB[1].x = B[(kb + 16 + 8 + txb16) * N + nb + txm16 * 4 + 0];
                        rB[1].y = rB[1].z = rB[1].w = 0;
                    }
                    else {
                        rB[1].x = rB[1].y = rB[1].z = rB[1].w = 0;
                    }
                }
                else {
                    rB[1].x = rB[1].y = rB[1].z = rB[1].w = 0;
                }
            }

        }

        for (m = 0; m < 8; m += 4) {
            float4 t = *reinterpret_cast<float4*>((*psA)[0] + wmb + m * 4 + txm32b8 * 4);
            rAs[m + 0] = t.x;
            rAs[m + 1] = t.y;
            rAs[m + 2] = t.z;
            rAs[m + 3] = t.w;
        }
        float4 t = *reinterpret_cast<float4*>((*psB)[0] + wnb + txm8 * 4);
        rBs[0] = t.x;
        rBs[1] = t.y;
        rBs[2] = t.z;
        rBs[3] = t.w;

#pragma unroll
        for (k = 0; k < 16; k++) {
            if (k < 15) {
#pragma unroll
                for (m = 0; m < 8; m += 4) {
                    float4 t = *reinterpret_cast<float4*>((*psA)[k + 1] + wmb + m * 4 + txm32b8 * 4);
                    rAsb[m + 0] = t.x;
                    rAsb[m + 1] = t.y;
                    rAsb[m + 2] = t.z;
                    rAsb[m + 3] = t.w;
                }
                float4 t = *reinterpret_cast<float4*>((*psB)[k + 1] + wnb + txm8 * 4);
                rBsb[0] = t.x;
                rBsb[1] = t.y;
                rBsb[2] = t.z;
                rBsb[3] = t.w;
            }

#pragma unroll
            for (m = 0; m < 8; m++) {
#pragma unroll
                for (n = 0; n < 4; n++) {
                    rC[m][n] += rAs[m] * rBs[n];
                }
            }

#pragma unroll
            for (m = 0; m < 8; m++) {
                rAs[m] = rAsb[m];
            }
#pragma unroll
            for (n = 0; n < 4; n++) {
                rBs[n] = rBsb[n];
            }

        }

        psA = (psA == &sA) ? &sAb : &sA;
        psB = (psB == &sB) ? &sBb : &sB;

    }

#pragma unroll
    for (m = 0; m < 8; m += 4) {
#pragma unroll
        for (n = 0; n < 4; n++) {
            int p = mb + wmb + m * 4 + txm32b8 * 4 + n;
            if (p < M) {
                if (aligned) {
                    if (nb + wnb + txm8 * 4 < N) {
                        float4 t;
                        t.x = rC[m + n][0];
                        t.y = rC[m + n][1];
                        t.z = rC[m + n][2];
                        t.w = rC[m + n][3];
                        *reinterpret_cast<float4*>(C + (mb + wmb + m * 4 + txm32b8 * 4 + n) * N + nb + wnb + txm8 * 4) = t;
                    }
                }
                else {
                    int t = N - nb - wnb - txm8 * 4;
                    if (4 <= t) {
                        C[(mb + wmb + m * 4 + txm32b8 * 4 + n) * N + nb + wnb + txm8 * 4 + 0] = rC[m + n][0];
                        C[(mb + wmb + m * 4 + txm32b8 * 4 + n) * N + nb + wnb + txm8 * 4 + 1] = rC[m + n][1];
                        C[(mb + wmb + m * 4 + txm32b8 * 4 + n) * N + nb + wnb + txm8 * 4 + 2] = rC[m + n][2];
                        C[(mb + wmb + m * 4 + txm32b8 * 4 + n) * N + nb + wnb + txm8 * 4 + 3] = rC[m + n][3];
                    }
                    else if (3 <= t) {
                        C[(mb + wmb + m * 4 + txm32b8 * 4 + n) * N + nb + wnb + txm8 * 4 + 0] = rC[m + n][0];
                        C[(mb + wmb + m * 4 + txm32b8 * 4 + n) * N + nb + wnb + txm8 * 4 + 1] = rC[m + n][1];
                        C[(mb + wmb + m * 4 + txm32b8 * 4 + n) * N + nb + wnb + txm8 * 4 + 2] = rC[m + n][2];
                    }
                    else if (2 <= t) {
                        C[(mb + wmb + m * 4 + txm32b8 * 4 + n) * N + nb + wnb + txm8 * 4 + 0] = rC[m + n][0];
                        C[(mb + wmb + m * 4 + txm32b8 * 4 + n) * N + nb + wnb + txm8 * 4 + 1] = rC[m + n][1];
                    }
                    else if (1 <= t) {
                        C[(mb + wmb + m * 4 + txm32b8 * 4 + n) * N + nb + wnb + txm8 * 4 + 0] = rC[m + n][0];
                    }
                }
            }
        }
    }


}

__global__ void sgemm_32x32x16_NN(float* A, float* B, float* C, int M, int N, int K, int aligned) {

    __shared__ __align__(16) float sA[16][32];
    __shared__ __align__(16) float sB[16][32];
    __shared__ __align__(16) float sAb[16][32];
    __shared__ __align__(16) float sBb[16][32];

    int tx = threadIdx.x;
    int mb = blockIdx.z * 64 + (blockIdx.x / 2) * 32;
    int nb = blockIdx.y * 64 + (blockIdx.x % 2) * 32;

    if (mb >= M || nb >= N) {
        return;
    }

    int txm8 = tx % 8;
    int txb8 = tx / 8;
    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm32m4 = txm32 % 4;
    int txm32b4 = txm32 / 4;
    int txm16b2 = txm16 / 2;
    int txm32m2 = txm32 % 2;
    int txm32b16 = txm32 / 16;
    int wmb = 0;
    int wnb = txb32 * 16;

    float rC[4][4];
    float4 rA[2];
    float4 rB[2];
    float rAs[4];
    float rBs[4];
    float rAsb[4];
    float rBsb[4];

    float(*psA)[16][32] = &sA;
    float(*psB)[16][32] = &sB;

    int m, n, k, kb;

#pragma unroll
    for (m = 0; m < 4; m++) {
#pragma unroll
        for (n = 0; n < 4; n++) {
            rC[m][n] = 0;
        }
    }

#pragma unroll
    for (m = 0; m < 2; m++) {
        if (aligned) {
            if (mb + m * 16 + txm16 < M && txb16 * 4 < K) {
                rA[m] = *reinterpret_cast<float4*>(A + (mb + m * 16 + txm16) * K + txb16 * 4);
            }
            else {
                rA[m].x = rA[m].y = rA[m].z = rA[m].w = 0;
            }
            if (nb + txm8 * 4 < N && m * 8 + txb8 < K) {
                rB[m] = *reinterpret_cast<float4*>(B + (m * 8 + txb8) * N + nb + txm8 * 4);
            }
            else {
                rB[m].x = rB[m].y = rB[m].z = rB[m].w = 0;
            }
        }
        else {
            if (mb + m * 16 + txm16 < M) {
                int t = K - txb16 * 4;
                if (4 <= t) {
                    rA[m].x = A[(mb + m * 16 + txm16) * K + txb16 * 4 + 0];
                    rA[m].y = A[(mb + m * 16 + txm16) * K + txb16 * 4 + 1];
                    rA[m].z = A[(mb + m * 16 + txm16) * K + txb16 * 4 + 2];
                    rA[m].w = A[(mb + m * 16 + txm16) * K + txb16 * 4 + 3];
                }
                else if (3 <= t) {
                    rA[m].x = A[(mb + m * 16 + txm16) * K + txb16 * 4 + 0];
                    rA[m].y = A[(mb + m * 16 + txm16) * K + txb16 * 4 + 1];
                    rA[m].z = A[(mb + m * 16 + txm16) * K + txb16 * 4 + 2];
                    rA[m].w = 0;
                }
                else if (2 <= t) {
                    rA[m].x = A[(mb + m * 16 + txm16) * K + txb16 * 4 + 0];
                    rA[m].y = A[(mb + m * 16 + txm16) * K + txb16 * 4 + 1];
                    rA[m].z = rA[m].w = 0;
                }
                else if (1 <= t) {
                    rA[m].x = A[(mb + m * 16 + txm16) * K + txb16 * 4 + 0];
                    rA[m].y = rA[m].z = rA[m].w = 0;
                }
                else {
                    rA[m].x = rA[m].y = rA[m].z = rA[m].w = 0;
                }
            }
            else {
                rA[m].x = rA[m].y = rA[m].z = rA[m].w = 0;
            }
            if (m * 8 + txb8 < K) {
                int t = N - nb - txm8 * 4;
                if (4 <= t) {
                    rB[m].x = B[(m * 8 + txb8) * N + nb + txm8 * 4 + 0];
                    rB[m].y = B[(m * 8 + txb8) * N + nb + txm8 * 4 + 1];
                    rB[m].z = B[(m * 8 + txb8) * N + nb + txm8 * 4 + 2];
                    rB[m].w = B[(m * 8 + txb8) * N + nb + txm8 * 4 + 3];
                }
                else if (3 <= t) {
                    rB[m].x = B[(m * 8 + txb8) * N + nb + txm8 * 4 + 0];
                    rB[m].y = B[(m * 8 + txb8) * N + nb + txm8 * 4 + 1];
                    rB[m].z = B[(m * 8 + txb8) * N + nb + txm8 * 4 + 2];
                    rB[m].w = 0;
                }
                else if (2 <= t) {
                    rB[m].x = B[(m * 8 + txb8) * N + nb + txm8 * 4 + 0];
                    rB[m].y = B[(m * 8 + txb8) * N + nb + txm8 * 4 + 1];
                    rB[m].z = rB[m].w = 0;
                }
                else if (1 <= t) {
                    rB[m].x = B[(m * 8 + txb8) * N + nb + txm8 * 4 + 0];
                    rB[m].y = rB[m].z = rB[m].w = 0;
                }
                else {
                    rB[m].x = rB[m].y = rB[m].z = rB[m].w = 0;
                }
            }
            else {
                rB[m].x = rB[m].y = rB[m].z = rB[m].w = 0;
            }
        }
    }

    for (kb = 0; kb < K; kb += 16) {

#pragma unroll
        for (m = 0; m < 2; m++) {
            (*psA)[txb16 * 4 + 0][m * 16 + txm16] = rA[m].x;
            (*psA)[txb16 * 4 + 1][m * 16 + txm16] = rA[m].y;
            (*psA)[txb16 * 4 + 2][m * 16 + txm16] = rA[m].z;
            (*psA)[txb16 * 4 + 3][m * 16 + txm16] = rA[m].w;

            (*psB)[m * 8 + txb8][txm8 * 4 + 0] = rB[m].x;
            (*psB)[m * 8 + txb8][txm8 * 4 + 1] = rB[m].y;
            (*psB)[m * 8 + txb8][txm8 * 4 + 2] = rB[m].z;
            (*psB)[m * 8 + txb8][txm8 * 4 + 3] = rB[m].w;
        }

        __syncthreads();

        if (kb + 16 < K) {
#pragma unroll
            for (m = 0; m < 2; m++) {
                if (aligned) {
                    if (mb + m * 16 + txm16 < M && kb + 16 + txb16 * 4 < K) {
                        rA[m] = *reinterpret_cast<float4*>(A + (mb + m * 16 + txm16) * K + kb + 16 + txb16 * 4);
                    }
                    else {
                        rA[m].x = rA[m].y = rA[m].z = rA[m].w = 0;
                    }
                    if (nb + txm8 * 4 < N && kb + 16 + m * 8 + txb8 < K) {
                        rB[m] = *reinterpret_cast<float4*>(B + (kb + 16 + m * 8 + txb8) * N + nb + txm8 * 4);
                    }
                    else {
                        rB[m].x = rB[m].y = rB[m].z = rB[m].w = 0;
                    }
                }
                else {
                    if (mb + m * 16 + txm16 < M) {
                        int t = K - kb - 16 - txb16 * 4;
                        if (4 <= t) {
                            rA[m].x = A[(mb + m * 16 + txm16) * K + kb + 16 + txb16 * 4 + 0];
                            rA[m].y = A[(mb + m * 16 + txm16) * K + kb + 16 + txb16 * 4 + 1];
                            rA[m].z = A[(mb + m * 16 + txm16) * K + kb + 16 + txb16 * 4 + 2];
                            rA[m].w = A[(mb + m * 16 + txm16) * K + kb + 16 + txb16 * 4 + 3];
                        }
                        else if (3 <= t) {
                            rA[m].x = A[(mb + m * 16 + txm16) * K + kb + 16 + txb16 * 4 + 0];
                            rA[m].y = A[(mb + m * 16 + txm16) * K + kb + 16 + txb16 * 4 + 1];
                            rA[m].z = A[(mb + m * 16 + txm16) * K + kb + 16 + txb16 * 4 + 2];
                            rA[m].w = 0;
                        }
                        else if (2 <= t) {
                            rA[m].x = A[(mb + m * 16 + txm16) * K + kb + 16 + txb16 * 4 + 0];
                            rA[m].y = A[(mb + m * 16 + txm16) * K + kb + 16 + txb16 * 4 + 1];
                            rA[m].z = rA[m].w = 0;
                        }
                        else if (1 <= t) {
                            rA[m].x = A[(mb + m * 16 + txm16) * K + kb + 16 + txb16 * 4 + 0];
                            rA[m].y = rA[m].z = rA[m].w = 0;
                        }
                        else {
                            rA[m].x = rA[m].y = rA[m].z = rA[m].w = 0;
                        }
                    }
                    else {
                        rA[m].x = rA[m].y = rA[m].z = rA[m].w = 0;
                    }
                    if (kb + 16 + m * 8 + txb8 < K) {
                        int t = N - nb - txm8 * 4;
                        if (4 <= t) {
                            rB[m].x = B[(kb + 16 + m * 8 + txb8) * N + nb + txm8 * 4 + 0];
                            rB[m].y = B[(kb + 16 + m * 8 + txb8) * N + nb + txm8 * 4 + 1];
                            rB[m].z = B[(kb + 16 + m * 8 + txb8) * N + nb + txm8 * 4 + 2];
                            rB[m].w = B[(kb + 16 + m * 8 + txb8) * N + nb + txm8 * 4 + 3];
                        }
                        else if (3 <= t) {
                            rB[m].x = B[(kb + 16 + m * 8 + txb8) * N + nb + txm8 * 4 + 0];
                            rB[m].y = B[(kb + 16 + m * 8 + txb8) * N + nb + txm8 * 4 + 1];
                            rB[m].z = B[(kb + 16 + m * 8 + txb8) * N + nb + txm8 * 4 + 2];
                            rB[m].w = 0;
                        }
                        else if (2 <= t) {
                            rB[m].x = B[(kb + 16 + m * 8 + txb8) * N + nb + txm8 * 4 + 0];
                            rB[m].y = B[(kb + 16 + m * 8 + txb8) * N + nb + txm8 * 4 + 1];
                            rB[m].z = rB[m].w = 0;
                        }
                        else if (1 <= t) {
                            rB[m].x = B[(kb + 16 + m * 8 + txb8) * N + nb + txm8 * 4 + 0];
                            rB[m].y = rB[m].z = rB[m].w = 0;
                        }
                        else {
                            rB[m].x = rB[m].y = rB[m].z = rB[m].w = 0;
                        }
                    }
                    else {
                        rB[m].x = rB[m].y = rB[m].z = rB[m].w = 0;
                    }
                }
            }
        }

        // COMPUTE -------------------------

        float4 tA = *reinterpret_cast<float4*>((*psA)[0] + wmb + txm16b2 * 4);
        rAs[0] = tA.x;
        rAs[1] = tA.y;
        rAs[2] = tA.z;
        rAs[3] = tA.w;
        float4 tB = *reinterpret_cast<float4*>((*psB)[0] + wnb + txm32b16 * 8 + txm32m2 * 4);
        rBs[0] = tB.x;
        rBs[1] = tB.y;
        rBs[2] = tB.z;
        rBs[3] = tB.w;

#pragma unroll
        for (k = 0; k < 16; k++) {

            if (k < 15) {
                float4 tA = *reinterpret_cast<float4*>((*psA)[k + 1] + wmb + txm16b2 * 4);
                rAsb[0] = tA.x;
                rAsb[1] = tA.y;
                rAsb[2] = tA.z;
                rAsb[3] = tA.w;
                float4 tB = *reinterpret_cast<float4*>((*psB)[k + 1] + wnb + txm32b16 * 8 + txm32m2 * 4);
                rBsb[0] = tB.x;
                rBsb[1] = tB.y;
                rBsb[2] = tB.z;
                rBsb[3] = tB.w;
            }

#pragma unroll
            for (m = 0; m < 4; m++) {
#pragma unroll
                for (n = 0; n < 4; n++) {
                    rC[m][n] += rAs[m] * rBs[n];
                }
            }

#pragma unroll
            for (m = 0; m < 4; m++) {
                rAs[m] = rAsb[m];
                rBs[m] = rBsb[m];
            }
        }

        // ---------------------------------

        psA = (psA == &sA) ? &sAb : &sA;
        psB = (psB == &sB) ? &sBb : &sB;

    }

#pragma unroll
    for (m = 0; m < 4; m++) {
#pragma unroll
        for (n = 0; n < 4; n += 4) {
            int p = mb + wmb + txm16b2 * 4 + m;
            if (p < M) {
                if (aligned) {
                    if (nb + wnb + txm32b16 * 8 + txm32m2 * 4 + n < N) {
                        float4 t;
                        t.x = rC[m][n + 0];
                        t.y = rC[m][n + 1];
                        t.z = rC[m][n + 2];
                        t.w = rC[m][n + 3];
                        *reinterpret_cast<float4*>(C + p * N + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + n) = t;
                    }
                }
                else {
                    int t = N - nb - wnb - txm32b16 * 8 - txm32m2 * 4 - n;
                    if (4 <= t) {
                        C[p * N + (N - t) + 0] = rC[m][n + 0];
                        C[p * N + (N - t) + 1] = rC[m][n + 1];
                        C[p * N + (N - t) + 2] = rC[m][n + 2];
                        C[p * N + (N - t) + 3] = rC[m][n + 3];
                    }
                    else if (3 <= t) {
                        C[p * N + (N - t) + 0] = rC[m][n + 0];
                        C[p * N + (N - t) + 1] = rC[m][n + 1];
                        C[p * N + (N - t) + 2] = rC[m][n + 2];
                    }
                    else if (2 <= t) {
                        C[p * N + (N - t) + 0] = rC[m][n + 0];
                        C[p * N + (N - t) + 1] = rC[m][n + 1];
                    }
                    else if (1 <= t) {
                        C[p * N + (N - t) + 0] = rC[m][n + 0];
                    }
                }
            }
        }
    }

}

__global__ void sgemm_4x4x256_NN(float* A, float* B, float* C, int M, int N, int K, int aligned) {

    __shared__ float sA[4][256];
    __shared__ float sAb[4][256];
    __shared__ float sC[8][4][4];

    int tx = threadIdx.x;
    int mb = blockIdx.x * 4;
    int nb = blockIdx.y * 4;

    int txm8 = tx % 8;
    int txb8 = tx / 8;
    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm64 = tx % 64;
    int txb64 = tx / 64;
    int txm32m4 = txm32 % 4;
    int txm32b4 = txm32 / 4;
    int txm16b2 = txm16 / 2;
    int txm32m2 = txm32 % 2;
    int txm32b16 = txm32 / 16;
    int wmb = 0;
    int wnb = txb32 * 16;

    float rC[4][4];
    float4 rA, rAb;
    float4 rB, rBb;

    float (*psA)[4][256] = &sA;

    int m, n, k, kb;

#pragma unroll
    for (m = 0; m < 4; m++) {
#pragma unroll
        for (n = 0; n < 4; n++) {
            rC[m][n] = 0;
        }
    }

    if (mb + txb64 < M) {
        if (aligned) {
            if (txm64 * 4 < K) {
                rA = *reinterpret_cast<float4*>(A + (mb + txb64) * K + txm64 * 4);
            }
            else {
                rA.x = rA.y = rA.z = rA.w = 0;
            }
        }
        else {
            int t = K - txm64 * 4;
            if (4 <= t) {
                rA.x = A[(mb + txb64) * K + K - t + 0];
                rA.y = A[(mb + txb64) * K + K - t + 1];
                rA.z = A[(mb + txb64) * K + K - t + 2];
                rA.w = A[(mb + txb64) * K + K - t + 3];
            }
            else if (3 <= t) {
                rA.x = A[(mb + txb64) * K + K - t + 0];
                rA.y = A[(mb + txb64) * K + K - t + 1];
                rA.z = A[(mb + txb64) * K + K - t + 2];
                rA.w = 0;
            }
            else if (2 <= t) {
                rA.x = A[(mb + txb64) * K + K - t + 0];
                rA.y = A[(mb + txb64) * K + K - t + 1];
                rA.z = rA.w = 0;
            }
            else if (1 <= t) {
                rA.x = A[(mb + txb64) * K + K - t + 0];
                rA.y = rA.z = rA.w = 0;
            }
            else {
                rA.x = rA.y = rA.z = rA.w = 0;
            }
        }
    }
    else {
        rA.x = rA.y = rA.z = rA.w = 0;
    }
    if (tx < K) {
        if (aligned) {
            rB = *reinterpret_cast<float4*>(B + tx * N + nb);
        }
        else {
            int t = N - nb;
            if (4 <= t) {
                rB.x = B[tx * N + nb + 0];
                rB.y = B[tx * N + nb + 1];
                rB.z = B[tx * N + nb + 2];
                rB.w = B[tx * N + nb + 3];
            }
            else if (3 <= t) {
                rB.x = B[tx * N + nb + 0];
                rB.y = B[tx * N + nb + 1];
                rB.z = B[tx * N + nb + 2];
                rB.w = 0;
            }
            else if (2 <= t) {
                rB.x = B[tx * N + nb + 0];
                rB.y = B[tx * N + nb + 1];
                rB.z = rB.w = 0;
            }
            else if (1 <= t) {
                rB.x = B[tx * N + nb + 0];
                rB.y = rB.z = rB.w = 0;
            }
            else {
                rB.x = rB.y = rB.z = rB.w = 0;
            }
        }
    }
    else {
        rB.x = rB.y = rB.z = rB.w = 0;
    }

    for (kb = 0; kb < K; kb += 256) {

        //*reinterpret_cast<float4*>((*psA)[txb64] + txm64 * 4) = rA;
        (*psA)[txb64][txm64 * 4 + 0] = rA.x;
        (*psA)[txb64][txm64 * 4 + 1] = rA.y;
        (*psA)[txb64][txm64 * 4 + 2] = rA.z;
        (*psA)[txb64][txm64 * 4 + 3] = rA.w;

        __syncthreads();

        if (kb + 256 < K) {
            if (mb + txb64 < M) {
                if (aligned) {
                    if (kb + 256 + txm64 * 4 < K) {
                        rA = *reinterpret_cast<float4*>(A + (mb + txb64) * K + kb + 256 + txm64 * 4);
                    }
                    else {
                        rA.x = rA.y = rA.z = rA.w = 0;
                    }
                }
                else {
                    int t = K - kb - 256 - txm64 * 4;
                    if (4 <= t) {
                        rA.x = A[(mb + txb64) * K + K - t + 0];
                        rA.y = A[(mb + txb64) * K + K - t + 1];
                        rA.z = A[(mb + txb64) * K + K - t + 2];
                        rA.w = A[(mb + txb64) * K + K - t + 3];
                    }
                    else if (3 <= t) {
                        rA.x = A[(mb + txb64) * K + K - t + 0];
                        rA.y = A[(mb + txb64) * K + K - t + 1];
                        rA.z = A[(mb + txb64) * K + K - t + 2];
                        rA.w = 0;
                    }
                    else if (2 <= t) {
                        rA.x = A[(mb + txb64) * K + K - t + 0];
                        rA.y = A[(mb + txb64) * K + K - t + 1];
                        rA.z = rA.w = 0;
                    }
                    else if (1 <= t) {
                        rA.x = A[(mb + txb64) * K + K - t + 0];
                        rA.y = rA.z = rA.w = 0;
                    }
                    else {
                        rA.x = rA.y = rA.z = rA.w = 0;
                    }
                }
            }
            else {
                rA.x = rA.y = rA.z = rA.w = 0;
            }
            if (kb + 256 + tx < K) {
                if (aligned) {
                    rBb = *reinterpret_cast<float4*>(B + (kb + 256 + tx) * N + nb);
                }
                else {
                    int t = N - nb;
                    if (4 <= t) {
                        rBb.x = B[(kb + 256 + tx) * N + nb + 0];
                        rBb.y = B[(kb + 256 + tx) * N + nb + 1];
                        rBb.z = B[(kb + 256 + tx) * N + nb + 2];
                        rBb.w = B[(kb + 256 + tx) * N + nb + 3];
                    }
                    else if (3 <= t) {
                        rBb.x = B[(kb + 256 + tx) * N + nb + 0];
                        rBb.y = B[(kb + 256 + tx) * N + nb + 1];
                        rBb.z = B[(kb + 256 + tx) * N + nb + 2];
                        rBb.w = 0;
                    }
                    else if (2 <= t) {
                        rBb.x = B[(kb + 256 + tx) * N + nb + 0];
                        rBb.y = B[(kb + 256 + tx) * N + nb + 1];
                        rBb.z = rBb.w = 0;
                    }
                    else if (1 <= t) {
                        rBb.x = B[(kb + 256 + tx) * N + nb + 0];
                        rBb.y = rBb.z = rBb.w = 0;
                    }
                    else {
                        rBb.x = rBb.y = rBb.z = rBb.w = 0;
                    }
                }
            }
            else {
                rBb.x = rBb.y = rBb.z = rBb.w = 0;
            }
        }

        // COMPUTE -------------------------

#pragma unroll
        for (m = 0; m < 4; m++) {
            rC[m][0] += (*psA)[m][tx] * rB.x;
            rC[m][1] += (*psA)[m][tx] * rB.y;
            rC[m][2] += (*psA)[m][tx] * rB.z;
            rC[m][3] += (*psA)[m][tx] * rB.w;
        }

        rB.x = rBb.x;
        rB.y = rBb.y;
        rB.z = rBb.z;
        rB.w = rBb.w;

        psA = (psA == &sA) ? &sAb : &sA;

    }

#pragma unroll
    for (m = 0; m < 4; m++) {
#pragma unroll
        for (n = 0; n < 4; n++) {
            float t = __shfl_sync(FULL_MASK, rC[m][n], txm32 + 16, 32);
            rC[m][n] += t;
            t = __shfl_sync(FULL_MASK, rC[m][n], txm32 + 8, 32);
            rC[m][n] += t;
            t = __shfl_sync(FULL_MASK, rC[m][n], txm32 + 4, 32);
            rC[m][n] += t;
            t = __shfl_sync(FULL_MASK, rC[m][n], txm32 + 2, 32);
            rC[m][n] += t;
            t = __shfl_sync(FULL_MASK, rC[m][n], txm32 + 1, 32);
            rC[m][n] += t;
            if (txm32 == 0) {
                sC[txb32][m][n] = rC[m][n];
            }
        }
    }

    __syncthreads();

    if (tx == 0) {
        if (aligned) {
#pragma unroll
            for (m = 0; m < 4; m++) {
                if (mb + m < M) {
                    float4 t;
                    t.x = rC[m][0] + sC[1][m][0] + sC[2][m][0] + sC[3][m][0] + sC[4][m][0] + sC[5][m][0] + sC[6][m][0] + sC[7][m][0];
                    t.y = rC[m][1] + sC[1][m][1] + sC[2][m][1] + sC[3][m][1] + sC[4][m][1] + sC[5][m][1] + sC[6][m][1] + sC[7][m][1];
                    t.z = rC[m][2] + sC[1][m][2] + sC[2][m][2] + sC[3][m][2] + sC[4][m][2] + sC[5][m][2] + sC[6][m][2] + sC[7][m][2];
                    t.w = rC[m][3] + sC[1][m][3] + sC[2][m][3] + sC[3][m][3] + sC[4][m][3] + sC[5][m][3] + sC[6][m][3] + sC[7][m][3];
                    *reinterpret_cast<float4*>(C + (mb + m) * N + nb) = t;
                }
            }
        }
        else {
            int t = N - nb;
#pragma unroll
            for (m = 0; m < 4; m++) {
                if (mb + m < M) {
                    if (4 <= t) {
                        C[(mb + m) * N + nb + 0] = rC[m][0] + sC[1][m][0] + sC[2][m][0] + sC[3][m][0] + sC[4][m][0] + sC[5][m][0] + sC[6][m][0] + sC[7][m][0];
                        C[(mb + m) * N + nb + 1] = rC[m][1] + sC[1][m][1] + sC[2][m][1] + sC[3][m][1] + sC[4][m][1] + sC[5][m][1] + sC[6][m][1] + sC[7][m][1];
                        C[(mb + m) * N + nb + 2] = rC[m][2] + sC[1][m][2] + sC[2][m][2] + sC[3][m][2] + sC[4][m][2] + sC[5][m][2] + sC[6][m][2] + sC[7][m][2];
                        C[(mb + m) * N + nb + 3] = rC[m][3] + sC[1][m][3] + sC[2][m][3] + sC[3][m][3] + sC[4][m][3] + sC[5][m][3] + sC[6][m][3] + sC[7][m][3];
                    }
                    else if (3 <= t) {
                        C[(mb + m) * N + nb + 0] = rC[m][0] + sC[1][m][0] + sC[2][m][0] + sC[3][m][0] + sC[4][m][0] + sC[5][m][0] + sC[6][m][0] + sC[7][m][0];
                        C[(mb + m) * N + nb + 1] = rC[m][1] + sC[1][m][1] + sC[2][m][1] + sC[3][m][1] + sC[4][m][1] + sC[5][m][1] + sC[6][m][1] + sC[7][m][1];
                        C[(mb + m) * N + nb + 2] = rC[m][2] + sC[1][m][2] + sC[2][m][2] + sC[3][m][2] + sC[4][m][2] + sC[5][m][2] + sC[6][m][2] + sC[7][m][2];
                    }
                    else if (2 <= t) {
                        C[(mb + m) * N + nb + 0] = rC[m][0] + sC[1][m][0] + sC[2][m][0] + sC[3][m][0] + sC[4][m][0] + sC[5][m][0] + sC[6][m][0] + sC[7][m][0];
                        C[(mb + m) * N + nb + 1] = rC[m][1] + sC[1][m][1] + sC[2][m][1] + sC[3][m][1] + sC[4][m][1] + sC[5][m][1] + sC[6][m][1] + sC[7][m][1];
                    }
                    else if (1 <= t) {
                        C[(mb + m) * N + nb + 0] = rC[m][0] + sC[1][m][0] + sC[2][m][0] + sC[3][m][0] + sC[4][m][0] + sC[5][m][0] + sC[6][m][0] + sC[7][m][0];
                    }
                }
            }
        }
    }

}

__global__ void sgemm_1x4x256_NN(float* A, float* B, float* C, int M, int N, int K, int aligned) {

    __shared__ float sC[8][4];

    int tx = threadIdx.x;
    int nb = blockIdx.x * 4;

    int txm8 = tx % 8;
    int txb8 = tx / 8;
    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm32m4 = txm32 % 4;
    int txm32b4 = txm32 / 4;
    int txm16b2 = txm16 / 2;
    int txm32m2 = txm32 % 2;
    int txm32b16 = txm32 / 16;
    int wmb = 0;
    int wnb = txb32 * 16;

    float rC[4];
    float rA, rAb;
    float4 rB, rBb;

    int m, n, k, kb;

#pragma unroll
    for (m = 0; m < 4; m++) {
        rC[m] = 0;
    }

    if (tx < K) {
        rA = A[tx];
        if (aligned) {
            rB = *reinterpret_cast<float4*>(B + tx * N + nb);
        }
        else {
            int t = N - nb;
            if (4 <= t) {
                rB.x = B[tx * N + nb + 0];
                rB.y = B[tx * N + nb + 1];
                rB.z = B[tx * N + nb + 2];
                rB.w = B[tx * N + nb + 3];
            }
            else if (3 <= t) {
                rB.x = B[tx * N + nb + 0];
                rB.y = B[tx * N + nb + 1];
                rB.z = B[tx * N + nb + 2];
                rB.w = 0;
            }
            else if (2 <= t) {
                rB.x = B[tx * N + nb + 0];
                rB.y = B[tx * N + nb + 1];
                rB.z = rB.w = 0;
            }
            else if (1 <= t) {
                rB.x = B[tx * N + nb + 0];
                rB.y = rB.z = rB.w = 0;
            }
            else {
                rB.x = rB.y = rB.z = rB.w = 0;
            }
        }
    }
    else {
        rA = 0;
        rB.x = rB.y = rB.z = rB.w = 0;
    }

    for (kb = 0; kb < K; kb += 256) {

        if (kb + 256 < K) {

            if (kb + 256 + tx < K) {
                rAb = A[kb + 256 + tx];
                if (aligned) {
                    rBb = *reinterpret_cast<float4*>(B + (kb + 256 + tx) * N + nb);
                }
                else {
                    int t = N - nb;
                    if (4 <= t) {
                        rBb.x = B[(kb + 256 + tx) * N + nb + 0];
                        rBb.y = B[(kb + 256 + tx) * N + nb + 1];
                        rBb.z = B[(kb + 256 + tx) * N + nb + 2];
                        rBb.w = B[(kb + 256 + tx) * N + nb + 3];
                    }
                    else if (3 <= t) {
                        rBb.x = B[(kb + 256 + tx) * N + nb + 0];
                        rBb.y = B[(kb + 256 + tx) * N + nb + 1];
                        rBb.z = B[(kb + 256 + tx) * N + nb + 2];
                        rBb.w = 0;
                    }
                    else if (2 <= t) {
                        rBb.x = B[(kb + 256 + tx) * N + nb + 0];
                        rBb.y = B[(kb + 256 + tx) * N + nb + 1];
                        rBb.z = rBb.w = 0;
                    }
                    else if (1 <= t) {
                        rBb.x = B[(kb + 256 + tx) * N + nb + 0];
                        rBb.y = rBb.z = rBb.w = 0;
                    }
                    else {
                        rBb.x = rBb.y = rBb.z = rBb.w = 0;
                    }
                }
            }
            else {
                rAb = 0;
                rBb.x = rBb.y = rBb.z = rBb.w = 0;
            }
        }

        // COMPUTE -------------------------

        rC[0] += rA * rB.x;
        rC[1] += rA * rB.y;
        rC[2] += rA * rB.z;
        rC[3] += rA * rB.w;

        rA = rAb;
        rB = rBb;

    }

#pragma unroll
    for (m = 0; m < 4; m++) {
        float t = __shfl_sync(FULL_MASK, rC[m], txm32 + 16, 32);
        rC[m] += t;
        t = __shfl_sync(FULL_MASK, rC[m], txm32 + 8, 32);
        rC[m] += t;
        t = __shfl_sync(FULL_MASK, rC[m], txm32 + 4, 32);
        rC[m] += t;
        t = __shfl_sync(FULL_MASK, rC[m], txm32 + 2, 32);
        rC[m] += t;
        t = __shfl_sync(FULL_MASK, rC[m], txm32 + 1, 32);
        rC[m] += t;
        if (txm32 == 0) {
            sC[txb32][m] = rC[m];
        }
    }

    __syncthreads();

    if (tx == 0) {
        if (aligned) {
            float4 t;
            t.x = rC[0] + sC[1][0] + sC[2][0] + sC[3][0] + sC[4][0] + sC[5][0] + sC[6][0] + sC[7][0];
            t.y = rC[1] + sC[1][1] + sC[2][1] + sC[3][1] + sC[4][1] + sC[5][1] + sC[6][1] + sC[7][1];
            t.z = rC[2] + sC[1][2] + sC[2][2] + sC[3][2] + sC[4][2] + sC[5][2] + sC[6][2] + sC[7][2];
            t.w = rC[3] + sC[1][3] + sC[2][3] + sC[3][3] + sC[4][3] + sC[5][3] + sC[6][3] + sC[7][3];
            *reinterpret_cast<float4*>(C + nb) = t;
        }
        else {
            int t = N - nb;
            if (4 <= t) {
                C[nb + 0] = rC[0] + sC[1][0] + sC[2][0] + sC[3][0] + sC[4][0] + sC[5][0] + sC[6][0] + sC[7][0];
                C[nb + 1] = rC[1] + sC[1][1] + sC[2][1] + sC[3][1] + sC[4][1] + sC[5][1] + sC[6][1] + sC[7][1];
                C[nb + 2] = rC[2] + sC[1][2] + sC[2][2] + sC[3][2] + sC[4][2] + sC[5][2] + sC[6][2] + sC[7][2];
                C[nb + 3] = rC[3] + sC[1][3] + sC[2][3] + sC[3][3] + sC[4][3] + sC[5][3] + sC[6][3] + sC[7][3];
            }
            else if (3 <= t) {
                C[nb + 0] = rC[0] + sC[1][0] + sC[2][0] + sC[3][0] + sC[4][0] + sC[5][0] + sC[6][0] + sC[7][0];
                C[nb + 1] = rC[1] + sC[1][1] + sC[2][1] + sC[3][1] + sC[4][1] + sC[5][1] + sC[6][1] + sC[7][1];
                C[nb + 2] = rC[2] + sC[1][2] + sC[2][2] + sC[3][2] + sC[4][2] + sC[5][2] + sC[6][2] + sC[7][2];
            }
            else if (2 <= t) {
                C[nb + 0] = rC[0] + sC[1][0] + sC[2][0] + sC[3][0] + sC[4][0] + sC[5][0] + sC[6][0] + sC[7][0];
                C[nb + 1] = rC[1] + sC[1][1] + sC[2][1] + sC[3][1] + sC[4][1] + sC[5][1] + sC[6][1] + sC[7][1];
            }
            else if (1 <= t) {
                C[nb + 0] = rC[0] + sC[1][0] + sC[2][0] + sC[3][0] + sC[4][0] + sC[5][0] + sC[6][0] + sC[7][0];
            }
        }
    }

}

__global__ void sgemm_128x128x1(float* A, float* B, float* C, int M, int N, int K, int aligned) {

    __shared__ __align__(16) float sA[128];
    __shared__ __align__(16) float sB[128];

    int tx = threadIdx.x;
    int mb = blockIdx.z * 256 + (blockIdx.x / 2) * 128;
    int nb = blockIdx.y * 256 + (blockIdx.x % 2) * 128;

    if (mb >= M || nb >= N) {
        return;
    }

    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm32 = tx % 32;
    int txb32 = tx / 32;

    int m, n;
    float rsA[8];
    float rsB[8];

    if (txb32 == 0) {
        if (aligned) {
            if (mb + txm32 * 4 < M) {
                *reinterpret_cast<float4*>(sA + txm32 * 4) = *reinterpret_cast<float4*>(A + mb + txm32 * 4);
            }
            else {
                sA[txm32 * 4 + 0] = sA[txm32 * 4 + 1] = sA[txm32 * 4 + 2] = sA[txm32 * 4 + 3] = 0;
            }
        }
        else {
            int t = M - mb - txm32 * 4;
            if (4 <= t) {
                sA[txm32 * 4 + 0] = A[M - t + 0];
                sA[txm32 * 4 + 1] = A[M - t + 1];
                sA[txm32 * 4 + 2] = A[M - t + 2];
                sA[txm32 * 4 + 3] = A[M - t + 3];
            }
            else if (3 <= t) {
                sA[txm32 * 4 + 0] = A[M - t + 0];
                sA[txm32 * 4 + 1] = A[M - t + 1];
                sA[txm32 * 4 + 2] = A[M - t + 2];
                sA[txm32 * 4 + 3] = 0;
            }
            else if (2 <= t) {
                sA[txm32 * 4 + 0] = A[M - t + 0];
                sA[txm32 * 4 + 1] = A[M - t + 1];
                sA[txm32 * 4 + 2] = sA[txm32 * 4 + 3] = 0;
            }
            else if (1 <= t) {
                sA[txm32 * 4 + 0] = A[M - t + 0];
                sA[txm32 * 4 + 1] = sA[txm32 * 4 + 2] = sA[txm32 * 4 + 3] = 0;
            }
            else {
                sA[txm32 * 4 + 0] = sA[txm32 * 4 + 1] = sA[txm32 * 4 + 2] = sA[txm32 * 4 + 3] = 0;
            }
        }
    }
    if (txb32 == 1) {
        if (aligned) {
            if (nb + txm32 * 4 < N) {
                *reinterpret_cast<float4*>(sB + txm32 * 4) = *reinterpret_cast<float4*>(B + nb + txm32 * 4);
            }
            else {
                sB[txm32 * 4 + 0] = sB[txm32 * 4 + 1] = sB[txm32 * 4 + 2] = sB[txm32 * 4 + 3] = 0;
            }
        }
        else {
            int t = N - nb - txm32 * 4;
            if (4 <= t) {
                sB[txm32 * 4 + 0] = B[N - t + 0];
                sB[txm32 * 4 + 1] = B[N - t + 1];
                sB[txm32 * 4 + 2] = B[N - t + 2];
                sB[txm32 * 4 + 3] = B[N - t + 3];
            }
            else if (3 <= t) {
                sB[txm32 * 4 + 0] = B[N - t + 0];
                sB[txm32 * 4 + 1] = B[N - t + 1];
                sB[txm32 * 4 + 2] = B[N - t + 2];
                sB[txm32 * 4 + 3] = 0;
            }
            else if (2 <= t) {
                sB[txm32 * 4 + 0] = B[N - t + 0];
                sB[txm32 * 4 + 1] = B[N - t + 1];
                sB[txm32 * 4 + 2] = sB[txm32 * 4 + 3] = 0;
            }
            else if (1 <= t) {
                sB[txm32 * 4 + 0] = B[N - t + 0];
                sB[txm32 * 4 + 1] = sB[txm32 * 4 + 2] = sB[txm32 * 4 + 3] = 0;
            }
            else {
                sB[txm32 * 4 + 0] = sB[txm32 * 4 + 1] = sB[txm32 * 4 + 2] = sB[txm32 * 4 + 3] = 0;
            }
        }
    }

    __syncthreads();

    float4 t = *reinterpret_cast<float4*>(sA + txb16 * 8);
    rsA[0] = t.x;
    rsA[1] = t.y;
    rsA[2] = t.z;
    rsA[3] = t.w;
    t = *reinterpret_cast<float4*>(sA + txb16 * 8 + 4);
    rsA[4] = t.x;
    rsA[5] = t.y;
    rsA[6] = t.z;
    rsA[7] = t.w;
    t = *reinterpret_cast<float4*>(sB + txm16 * 4);
    rsB[0] = t.x;
    rsB[1] = t.y;
    rsB[2] = t.z;
    rsB[3] = t.w;
    t = *reinterpret_cast<float4*>(sB + 64 + txm16 * 4);
    rsB[4] = t.x;
    rsB[5] = t.y;
    rsB[6] = t.z;
    rsB[7] = t.w;

#pragma unroll
    for (m = 0; m < 8; m++) {
        int p1 = mb + txb16 * 8 + m;
        if (p1 < M) {
            int p2 = nb + txm16 * 4;
            int p3 = nb + 64 + txm16 * 4;
            if (aligned) {
                if (p2 < N) {
                    float4 t;
                    t.x = rsA[m] * rsB[0];
                    t.y = rsA[m] * rsB[1];
                    t.z = rsA[m] * rsB[2];
                    t.w = rsA[m] * rsB[3];
                    *reinterpret_cast<float4*>(C + p1 * N + p2) = t;
                }
                if (p3 < N) {
                    t.x = rsA[m] * rsB[4];
                    t.y = rsA[m] * rsB[5];
                    t.z = rsA[m] * rsB[6];
                    t.w = rsA[m] * rsB[7];
                    *reinterpret_cast<float4*>(C + p1 * N + p3) = t;
                }
            }
            else {
                int t = N - p2;
                if (4 <= t) {
                    C[p1 * N + p2 + 0] = rsA[m] * rsB[0];
                    C[p1 * N + p2 + 1] = rsA[m] * rsB[1];
                    C[p1 * N + p2 + 2] = rsA[m] * rsB[2];
                    C[p1 * N + p2 + 3] = rsA[m] * rsB[3];
                }
                else if (3 <= t) {
                    C[p1 * N + p2 + 0] = rsA[m] * rsB[0];
                    C[p1 * N + p2 + 1] = rsA[m] * rsB[1];
                    C[p1 * N + p2 + 2] = rsA[m] * rsB[2];
                }
                else if (2 <= t) {
                    C[p1 * N + p2 + 0] = rsA[m] * rsB[0];
                    C[p1 * N + p2 + 1] = rsA[m] * rsB[1];
                }
                else if (1 <= t) {
                    C[p1 * N + p2 + 0] = rsA[m] * rsB[0];
                }
                t = N - p3;
                if (4 <= t) {
                    C[p1 * N + p3 + 0] = rsA[m] * rsB[4];
                    C[p1 * N + p3 + 1] = rsA[m] * rsB[5];
                    C[p1 * N + p3 + 2] = rsA[m] * rsB[6];
                    C[p1 * N + p3 + 3] = rsA[m] * rsB[7];
                }
                else if (3 <= t) {
                    C[p1 * N + p3 + 0] = rsA[m] * rsB[4];
                    C[p1 * N + p3 + 1] = rsA[m] * rsB[5];
                    C[p1 * N + p3 + 2] = rsA[m] * rsB[6];
                }
                else if (2 <= t) {
                    C[p1 * N + p3 + 0] = rsA[m] * rsB[4];
                    C[p1 * N + p3 + 1] = rsA[m] * rsB[5];
                }
                else if (1 <= t) {
                    C[p1 * N + p3 + 0] = rsA[m] * rsB[4];
                }
            }
        }
    }


}


__global__ void sgemm_128x128x8_NT(float* A, float* B, float* C, int M, int N, int K, int aligned) {

    __shared__ __align__(16) float sA[8][132];
    __shared__ __align__(16) float sB[8][132];
    __shared__ __align__(16) float sAb[8][132];
    __shared__ __align__(16) float sBb[8][132];

    int tx = threadIdx.x;
    int mb = blockIdx.z * 256 + (blockIdx.x / 2) * 128;
    int nb = blockIdx.y * 256 + (blockIdx.x % 2) * 128;

    if (mb >= M || nb >= N) {
        return;
    }

    float rC[8][8];
    float4 rA;
    float4 rB;
    float rAs[8];
    float rBs[8];
    float rAsb[8];
    float rBsb[8];

    float(*psA)[8][132] = &sA;
    float(*psB)[8][132] = &sB;

    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm2 = tx % 2;
    int txb2 = tx / 2;
    int txm8 = tx % 8;
    int txb8 = tx / 8;
    int txm128 = tx % 128;
    int txb128 = tx / 128;
    int txm32b4 = txm32 / 4;

    int wnb = (txb32 / 4) * 64;
    int wmb = (txb32 % 4) * 32;

    int m, n, k, kb, kbb;

    float4 f4;
    float8 f8;

    int sAi1 = wmb + (txm32 % 2) * 8 + (txm32 / 16) * 16;
    int sBi1 = wnb + (txm32 / 2) * 4;
    int sBi2 = wnb + (32 + (txm32 / 2) * 4) % 64;

#pragma unroll
    for (m = 0; m < 8; m++) {
#pragma unroll
        for (n = 0; n < 8; n++) {
            rC[m][n] = 0;
        }
    }

    if (aligned) {
        if (txb128 * 4 < K) {
            if (mb + txm128 < M) {
                rA = *reinterpret_cast<float4*>(A + (mb + txm128) * K + txb128 * 4);
            }
            else {
                rA.x = rA.y = rA.z = rA.w = 0;
            }
            if (nb + txm128 < N) {
                rB = *reinterpret_cast<float4*>(B + (nb + txm128) * K + txb128 * 4);
            }
            else {
                rB.x = rB.y = rB.z = rB.w = 0;
            }
        }
        else {
            rB.x = rB.y = rB.z = rB.w = 0;
        }
    }
    else {
        int t = K - txb128 * 4;
        if (mb + txm128 < M) {
            if (4 <= t) {
                rA.x = A[(mb + txm128) * K + txb128 * 4 + 0];
                rA.y = A[(mb + txm128) * K + txb128 * 4 + 1];
                rA.z = A[(mb + txm128) * K + txb128 * 4 + 2];
                rA.w = A[(mb + txm128) * K + txb128 * 4 + 3];
            }
            else if (3 <= t) {
                rA.x = A[(mb + txm128) * K + txb128 * 4 + 0];
                rA.y = A[(mb + txm128) * K + txb128 * 4 + 1];
                rA.z = A[(mb + txm128) * K + txb128 * 4 + 2];
                rA.w = 0;
            }
            else if (2 <= t) {
                rA.x = A[(mb + txm128) * K + txb128 * 4 + 0];
                rA.y = A[(mb + txm128) * K + txb128 * 4 + 1];
                rA.z = rA.w = 0;
            }
            else if (1 <= t) {
                rA.x = A[(mb + txm128) * K + txb128 * 4 + 0];
                rA.y = rA.z = rA.w = 0;
            }
            else {
                rA.x = rA.y = rA.z = rA.w = 0;
            }
        }
        else {
            rA.x = rA.y = rA.z = rA.w = 0;
        }
        if (nb + txm128 < N) {
            if (4 <= t) {
                rB.x = B[(nb + txm128) * K + txb128 * 4 + 0];
                rB.y = B[(nb + txm128) * K + txb128 * 4 + 1];
                rB.z = B[(nb + txm128) * K + txb128 * 4 + 2];
                rB.w = B[(nb + txm128) * K + txb128 * 4 + 3];
            }
            else if (3 <= t) {
                rB.x = B[(nb + txm128) * K + txb128 * 4 + 0];
                rB.y = B[(nb + txm128) * K + txb128 * 4 + 1];
                rB.z = B[(nb + txm128) * K + txb128 * 4 + 2];
                rB.w = 0;
            }
            else if (2 <= t) {
                rB.x = B[(nb + txm128) * K + txb128 * 4 + 0];
                rB.y = B[(nb + txm128) * K + txb128 * 4 + 1];
                rB.z = rB.w = 0;
            }
            else if (1 <= t) {
                rB.x = B[(nb + txm128) * K + txb128 * 4 + 0];
                rB.y = rB.z = rB.w = 0;
            }
            else {
                rB.x = rB.y = rB.z = rB.w = 0;
            }
        }
        else {
            rB.x = rB.y = rB.z = rB.w = 0;
        }
    }

    for (kb = 0; kb < K; kb += 8) {

        (*psA)[txb128 * 4 + 0][txm128] = rA.x;
        (*psA)[txb128 * 4 + 1][txm128] = rA.y;
        (*psA)[txb128 * 4 + 2][txm128] = rA.z;
        (*psA)[txb128 * 4 + 3][txm128] = rA.w;
        (*psB)[txb128 * 4 + 0][txm128] = rB.x;
        (*psB)[txb128 * 4 + 1][txm128] = rB.y;
        (*psB)[txb128 * 4 + 2][txm128] = rB.z;
        (*psB)[txb128 * 4 + 3][txm128] = rB.w;

        __syncthreads();

        if (kb + 8 < K) {
            if (aligned) {
                if (kb + 8 + txb128 * 4 < K) {
                    if (mb + txm128 < M) {
                        rA = *reinterpret_cast<float4*>(A + (mb + txm128) * K + kb + 8 + txb128 * 4);
                    }
                    else {
                        rA.x = rA.y = rA.z = rA.w = 0;
                    }
                    if (nb + txm128 < N) {
                        rB = *reinterpret_cast<float4*>(B + (nb + txm128) * K + kb + 8 + txb128 * 4);
                    }
                    else {
                        rB.x = rB.y = rB.z = rB.w = 0;
                    }
                }
                else {
                    rB.x = rB.y = rB.z = rB.w = 0;
                }
            }
            else {
                int t = K - txb128 * 4 - kb - 8;
                if (mb + txm128 < M) {
                    if (4 <= t) {
                        rA.x = A[(mb + txm128) * K + kb + 8 + txb128 * 4 + 0];
                        rA.y = A[(mb + txm128) * K + kb + 8 + txb128 * 4 + 1];
                        rA.z = A[(mb + txm128) * K + kb + 8 + txb128 * 4 + 2];
                        rA.w = A[(mb + txm128) * K + kb + 8 + txb128 * 4 + 3];
                    }
                    else if (3 <= t) {
                        rA.x = A[(mb + txm128) * K + kb + 8 + txb128 * 4 + 0];
                        rA.y = A[(mb + txm128) * K + kb + 8 + txb128 * 4 + 1];
                        rA.z = A[(mb + txm128) * K + kb + 8 + txb128 * 4 + 2];
                        rA.w = 0;
                    }
                    else if (2 <= t) {
                        rA.x = A[(mb + txm128) * K + kb + 8 + txb128 * 4 + 0];
                        rA.y = A[(mb + txm128) * K + kb + 8 + txb128 * 4 + 1];
                        rA.z = rA.w = 0;
                    }
                    else if (1 <= t) {
                        rA.x = A[(mb + txm128) * K + kb + 8 + txb128 * 4 + 0];
                        rA.y = rA.z = rA.w = 0;
                    }
                    else {
                        rA.x = rA.y = rA.z = rA.w = 0;
                    }
                }
                else {
                    rA.x = rA.y = rA.z = rA.w = 0;
                }
                if (nb + txm128 < N) {
                    if (4 <= t) {
                        rB.x = B[(nb + txm128) * K + kb + 8 + txb128 * 4 + 0];
                        rB.y = B[(nb + txm128) * K + kb + 8 + txb128 * 4 + 1];
                        rB.z = B[(nb + txm128) * K + kb + 8 + txb128 * 4 + 2];
                        rB.w = B[(nb + txm128) * K + kb + 8 + txb128 * 4 + 3];
                    }
                    else if (3 <= t) {
                        rB.x = B[(nb + txm128) * K + kb + 8 + txb128 * 4 + 0];
                        rB.y = B[(nb + txm128) * K + kb + 8 + txb128 * 4 + 1];
                        rB.z = B[(nb + txm128) * K + kb + 8 + txb128 * 4 + 2];
                        rB.w = 0;
                    }
                    else if (2 <= t) {
                        rB.x = B[(nb + txm128) * K + kb + 8 + txb128 * 4 + 0];
                        rB.y = B[(nb + txm128) * K + kb + 8 + txb128 * 4 + 1];
                        rB.z = rB.w = 0;
                    }
                    else if (1 <= t) {
                        rB.x = B[(nb + txm128) * K + kb + 8 + txb128 * 4 + 0];
                        rB.y = rB.z = rB.w = 0;
                    }
                    else {
                        rB.x = rB.y = rB.z = rB.w = 0;
                    }
                }
                else {
                    rB.x = rB.y = rB.z = rB.w = 0;
                }
            }
        }

        // COMPUTE -------------------

        f8 = *reinterpret_cast<float8*>((*psA)[0] + sAi1);
        rAs[0] = f8.a;
        rAs[1] = f8.b;
        rAs[2] = f8.c;
        rAs[3] = f8.d;
        rAs[4] = f8.e;
        rAs[5] = f8.f;
        rAs[6] = f8.g;
        rAs[7] = f8.h;
        f4 = *reinterpret_cast<float4*>((*psB)[0] + sBi1);
        rBs[0] = f4.x;
        rBs[1] = f4.y;
        rBs[2] = f4.z;
        rBs[3] = f4.w;
        f4 = *reinterpret_cast<float4*>((*psB)[0] + sBi2);
        rBs[4] = f4.x;
        rBs[5] = f4.y;
        rBs[6] = f4.z;
        rBs[7] = f4.w;

#pragma unroll
        for (k = 0; k < 8; k++) {

            if (k < 7) {
                f8 = *reinterpret_cast<float8*>((*psA)[k + 1] + sAi1);
                rAsb[0] = f8.a;
                rAsb[1] = f8.b;
                rAsb[2] = f8.c;
                rAsb[3] = f8.d;
                rAsb[4] = f8.e;
                rAsb[5] = f8.f;
                rAsb[6] = f8.g;
                rAsb[7] = f8.h;
                f4 = *reinterpret_cast<float4*>((*psB)[k + 1] + sBi1);
                rBsb[0] = f4.x;
                rBsb[1] = f4.y;
                rBsb[2] = f4.z;
                rBsb[3] = f4.w;
                f4 = *reinterpret_cast<float4*>((*psB)[k + 1] + sBi2);
                rBsb[4] = f4.x;
                rBsb[5] = f4.y;
                rBsb[6] = f4.z;
                rBsb[7] = f4.w;
            }

#pragma unroll
            for (m = 0; m < 8; m++) {
#pragma unroll
                for (n = 0; n < 8; n++) {
                    rC[m][n] += rAs[m] * rBs[n];
                }
            }

#pragma unroll
            for (m = 0; m < 8; m++) {
                rAs[m] = rAsb[m];
                rBs[m] = rBsb[m];
            }

        }

        // ---------------------------

        if (psA == &sA) {
            psA = &sAb;
            psB = &sBb;
        }
        else {
            psA = &sA;
            psB = &sB;
        }

    }

#pragma unroll
    for (m = 0; m < 8; m++) {
        if (mb + sAi1 + m < M) {
            if (aligned) {
                if (nb + sBi1 < N) {
                    float4 t;
                    t.x = rC[m][0];
                    t.y = rC[m][1];
                    t.z = rC[m][2];
                    t.w = rC[m][3];
                    *reinterpret_cast<float4*>(C + (mb + sAi1 + m) * N + nb + sBi1) = t;
                }
                if (nb + sBi2 < N) {
                    float4 t;
                    t.x = rC[m][4];
                    t.y = rC[m][5];
                    t.z = rC[m][6];
                    t.w = rC[m][7];
                    *reinterpret_cast<float4*>(C + (mb + sAi1 + m) * N + nb + sBi2) = t;
                }
            }
            else {
                int t = N - nb - sBi1;
                if (4 <= t) {
                    C[(mb + sAi1 + m) * N + nb + sBi1 + 0] = rC[m][0];
                    C[(mb + sAi1 + m) * N + nb + sBi1 + 1] = rC[m][1];
                    C[(mb + sAi1 + m) * N + nb + sBi1 + 2] = rC[m][2];
                    C[(mb + sAi1 + m) * N + nb + sBi1 + 3] = rC[m][3];
                }
                else if (3 <= t) {
                    C[(mb + sAi1 + m) * N + nb + sBi1 + 0] = rC[m][0];
                    C[(mb + sAi1 + m) * N + nb + sBi1 + 1] = rC[m][1];
                    C[(mb + sAi1 + m) * N + nb + sBi1 + 2] = rC[m][2];
                }
                else if (2 <= t) {
                    C[(mb + sAi1 + m) * N + nb + sBi1 + 0] = rC[m][0];
                    C[(mb + sAi1 + m) * N + nb + sBi1 + 1] = rC[m][1];
                }
                else if (1 <= t) {
                    C[(mb + sAi1 + m) * N + nb + sBi1 + 0] = rC[m][0];
                }
                t = N - nb - sBi2;
                if (4 <= t) {
                    C[(mb + sAi1 + m) * N + nb + sBi2 + 0] = rC[m][4];
                    C[(mb + sAi1 + m) * N + nb + sBi2 + 1] = rC[m][5];
                    C[(mb + sAi1 + m) * N + nb + sBi2 + 2] = rC[m][6];
                    C[(mb + sAi1 + m) * N + nb + sBi2 + 3] = rC[m][7];
                }
                else if (3 <= t) {
                    C[(mb + sAi1 + m) * N + nb + sBi2 + 0] = rC[m][4];
                    C[(mb + sAi1 + m) * N + nb + sBi2 + 1] = rC[m][5];
                    C[(mb + sAi1 + m) * N + nb + sBi2 + 2] = rC[m][6];
                }
                else if (2 <= t) {
                    C[(mb + sAi1 + m) * N + nb + sBi2 + 0] = rC[m][4];
                    C[(mb + sAi1 + m) * N + nb + sBi2 + 1] = rC[m][5];
                }
                else if (1 <= t) {
                    C[(mb + sAi1 + m) * N + nb + sBi2 + 0] = rC[m][4];
                }
            }
        }
    }
}

__global__ void sgemm_64x64x16_NT(float* A, float* B, float* C, int M, int N, int K, int aligned) {

    __shared__ __align__(16) float sA[16][64];
    __shared__ __align__(16) float sB[16][64];
    __shared__ __align__(16) float sAb[16][64];
    __shared__ __align__(16) float sBb[16][64];

    int tx = threadIdx.x;
    int mb = blockIdx.z * 128 + (blockIdx.x / 2) * 64;
    int nb = blockIdx.y * 128 + (blockIdx.x % 2) * 64;

    if (mb >= M || nb >= N) {
        return;
    }

    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm64 = tx % 64;
    int txb64 = tx / 64;
    int txm8 = tx % 8;
    int txm32b8 = txm32 / 8;

    int wmb = (txb32 / 2) * 32;
    int wnb = (txb32 % 2) * 32;

    int kb, k, m, n;

    float4 rA[2];
    float4 rB[2];
    float rC[8][4];
    float rAs[8];
    float rBs[4];
    float rAsb[8];
    float rBsb[4];

    float(*psA)[16][64] = &sA;
    float(*psB)[16][64] = &sB;

#pragma unroll
    for (m = 0; m < 8; m++) {
#pragma unroll
        for (n = 0; n < 4; n++) {
            rC[m][n] = 0;
        }
    }

    if (aligned) {
        if (txb32 * 4 < K) {
            if (mb + txm32 < M) {
                rA[0] = *reinterpret_cast<float4*>(A + (mb + txm32) * K + txb32 * 4);
            }
            else {
                rA[0].x = rA[0].y = rA[0].z = rA[0].w = 0;
            }
            if (mb + 32 + txm32 < M) {
                rA[1] = *reinterpret_cast<float4*>(A + (mb + 32 + txm32) * K + txb32 * 4);
            }
            else {
                rA[1].x = rA[1].y = rA[1].z = rA[1].w = 0;
            }
            if (nb + txm32 < N) {
                rB[0] = *reinterpret_cast<float4*>(B + (nb + txm32) * K + txb32 * 4);
            }
            else {
                rB[0].x = rB[0].y = rB[0].z = rB[0].w = 0;
            }
            if (nb + 32 + txm32 < N) {
                rB[1] = *reinterpret_cast<float4*>(B + (nb + 32 + txm32) * K + txb32 * 4);
            }
            else {
                rB[1].x = rB[1].y = rB[1].z = rB[1].w = 0;
            }
        }
        else {
            rA[0].x = rA[0].y = rA[0].z = rA[0].w = 0;
            rA[1].x = rA[1].y = rA[1].z = rA[1].w = 0;
            rB[0].x = rB[0].y = rB[0].z = rB[0].w = 0;
            rB[1].x = rB[1].y = rB[1].z = rB[1].w = 0;
        }
    }
    else {
        int t = K - txb32 * 4;
        if (4 <= t) {
            if (mb + txm32 < M) {
                rA[0].x = A[(mb + txm32) * K + txb32 * 4 + 0];
                rA[0].y = A[(mb + txm32) * K + txb32 * 4 + 1];
                rA[0].z = A[(mb + txm32) * K + txb32 * 4 + 2];
                rA[0].w = A[(mb + txm32) * K + txb32 * 4 + 3];
            }
            else {
                rA[0].x = rA[0].y = rA[0].z = rA[0].w = 0;
            }
            if (mb + 32 + txm32 < M) {
                rA[1].x = A[(mb + 32 + txm32) * K + txb32 * 4 + 0];
                rA[1].y = A[(mb + 32 + txm32) * K + txb32 * 4 + 1];
                rA[1].z = A[(mb + 32 + txm32) * K + txb32 * 4 + 2];
                rA[1].w = A[(mb + 32 + txm32) * K + txb32 * 4 + 3];
            }
            else {
                rA[1].x = rA[1].y = rA[1].z = rA[1].w = 0;
            }
            if (nb + txm32 < N) {
                rB[0].x = B[(nb + txm32) * K + txb32 * 4 + 0];
                rB[0].y = B[(nb + txm32) * K + txb32 * 4 + 1];
                rB[0].z = B[(nb + txm32) * K + txb32 * 4 + 2];
                rB[0].w = B[(nb + txm32) * K + txb32 * 4 + 3];
            }
            else {
                rB[0].x = rB[0].y = rB[0].z = rB[0].w = 0;
            }
            if (nb + 32 + txm32 < N) {
                rB[1].x = B[(nb + 32 + txm32) * K + txb32 * 4 + 0];
                rB[1].y = B[(nb + 32 + txm32) * K + txb32 * 4 + 1];
                rB[1].z = B[(nb + 32 + txm32) * K + txb32 * 4 + 2];
                rB[1].w = B[(nb + 32 + txm32) * K + txb32 * 4 + 3];
            }
            else {
                rB[1].x = rB[1].y = rB[1].z = rB[1].w = 0;
            }
        }
        else if (3 <= t) {
            if (mb + txm32 < M) {
                rA[0].x = A[(mb + txm32) * K + txb32 * 4 + 0];
                rA[0].y = A[(mb + txm32) * K + txb32 * 4 + 1];
                rA[0].z = A[(mb + txm32) * K + txb32 * 4 + 2];
                rA[0].w = 0;
            }
            else {
                rA[0].x = rA[0].y = rA[0].z = rA[0].w = 0;
            }
            if (mb + 32 + txm32 < M) {
                rA[1].x = A[(mb + 32 + txm32) * K + txb32 * 4 + 0];
                rA[1].y = A[(mb + 32 + txm32) * K + txb32 * 4 + 1];
                rA[1].z = A[(mb + 32 + txm32) * K + txb32 * 4 + 2];
                rA[1].w = 0;
            }
            else {
                rA[1].x = rA[1].y = rA[1].z = rA[1].w = 0;
            }
            if (nb + txm32 < N) {
                rB[0].x = B[(nb + txm32) * K + txb32 * 4 + 0];
                rB[0].y = B[(nb + txm32) * K + txb32 * 4 + 1];
                rB[0].z = B[(nb + txm32) * K + txb32 * 4 + 2];
                rB[0].w = 0;
            }
            else {
                rB[0].x = rB[0].y = rB[0].z = rB[0].w = 0;
            }
            if (nb + 32 + txm32 < N) {
                rB[1].x = B[(nb + 32 + txm32) * K + txb32 * 4 + 0];
                rB[1].y = B[(nb + 32 + txm32) * K + txb32 * 4 + 1];
                rB[1].z = B[(nb + 32 + txm32) * K + txb32 * 4 + 2];
                rB[1].w = 0;
            }
            else {
                rB[1].x = rB[1].y = rB[1].z = rB[1].w = 0;
            }
        }
        else if (2 <= t) {
            if (mb + txm32 < M) {
                rA[0].x = A[(mb + txm32) * K + txb32 * 4 + 0];
                rA[0].y = A[(mb + txm32) * K + txb32 * 4 + 1];
                rA[0].z = rA[0].w = 0;
            }
            else {
                rA[0].x = rA[0].y = rA[0].z = rA[0].w = 0;
            }
            if (mb + 32 + txm32 < M) {
                rA[1].x = A[(mb + 32 + txm32) * K + txb32 * 4 + 0];
                rA[1].y = A[(mb + 32 + txm32) * K + txb32 * 4 + 1];
                rA[1].z = rA[1].w = 0;
            }
            else {
                rA[1].x = rA[1].y = rA[1].z = rA[1].w = 0;
            }
            if (nb + txm32 < N) {
                rB[0].x = B[(nb + txm32) * K + txb32 * 4 + 0];
                rB[0].y = B[(nb + txm32) * K + txb32 * 4 + 1];
                rB[0].z = rB[0].w = 0;
            }
            else {
                rB[0].x = rB[0].y = rB[0].z = rB[0].w = 0;
            }
            if (nb + 32 + txm32 < N) {
                rB[1].x = B[(nb + 32 + txm32) * K + txb32 * 4 + 0];
                rB[1].y = B[(nb + 32 + txm32) * K + txb32 * 4 + 1];
                rB[1].z = rB[1].w = 0;
            }
            else {
                rB[1].x = rB[1].y = rB[1].z = rB[1].w = 0;
            }
        }
        else if (1 <= t) {
            if (mb + txm32 < M) {
                rA[0].x = A[(mb + txm32) * K + txb32 * 4 + 0];
                rA[0].y = rA[0].z = rA[0].w = 0;
            }
            else {
                rA[0].x = rA[0].y = rA[0].z = rA[0].w = 0;
            }
            if (mb + 32 + txm32 < M) {
                rA[1].x = A[(mb + 32 + txm32) * K + txb32 * 4 + 0];
                rA[1].y = rA[1].z = rA[1].w = 0;
            }
            else {
                rA[1].x = rA[1].y = rA[1].z = rA[1].w = 0;
            }
            if (nb + txm32 < N) {
                rB[0].x = B[(nb + txm32) * K + txb32 * 4 + 0];
                rB[0].y = rB[0].z = rB[0].w = 0;
            }
            else {
                rB[0].x = rB[0].y = rB[0].z = rB[0].w = 0;
            }
            if (nb + 32 + txm32 < N) {
                rB[1].x = B[(nb + 32 + txm32) * K + txb32 * 4 + 0];
                rB[1].y = rB[1].z = rB[1].w = 0;
            }
            else {
                rB[1].x = rB[1].y = rB[1].z = rB[1].w = 0;
            }
        }
        else {
            rA[0].x = rA[0].y = rA[0].z = rA[0].w = 0;
            rA[1].x = rA[1].y = rA[1].z = rA[1].w = 0;
            rB[0].x = rB[0].y = rB[0].z = rB[0].w = 0;
            rB[1].x = rB[1].y = rB[1].z = rB[1].w = 0;
        }
    }

    for (kb = 0; kb < K; kb += 16) {

        (*psA)[txb32 * 4 + 0][txm32] = rA[0].x;
        (*psA)[txb32 * 4 + 1][txm32] = rA[0].y;
        (*psA)[txb32 * 4 + 2][txm32] = rA[0].z;
        (*psA)[txb32 * 4 + 3][txm32] = rA[0].w;
        (*psA)[txb32 * 4 + 0][32 + txm32] = rA[1].x;
        (*psA)[txb32 * 4 + 1][32 + txm32] = rA[1].y;
        (*psA)[txb32 * 4 + 2][32 + txm32] = rA[1].z;
        (*psA)[txb32 * 4 + 3][32 + txm32] = rA[1].w;
        (*psB)[txb32 * 4 + 0][txm32] = rB[0].x;
        (*psB)[txb32 * 4 + 1][txm32] = rB[0].y;
        (*psB)[txb32 * 4 + 2][txm32] = rB[0].z;
        (*psB)[txb32 * 4 + 3][txm32] = rB[0].w;
        (*psB)[txb32 * 4 + 0][32 + txm32] = rB[1].x;
        (*psB)[txb32 * 4 + 1][32 + txm32] = rB[1].y;
        (*psB)[txb32 * 4 + 2][32 + txm32] = rB[1].z;
        (*psB)[txb32 * 4 + 3][32 + txm32] = rB[1].w;

        __syncthreads();

        if (kb + 16 < K) {
            if (aligned) {
                if (kb + 16 + txb32 * 4 < K) {
                    if (mb + txm32 < M) {
                        rA[0] = *reinterpret_cast<float4*>(A + (mb + txm32) * K + kb + 16 + txb32 * 4);
                    }
                    else {
                        rA[0].x = rA[0].y = rA[0].z = rA[0].w = 0;
                    }
                    if (mb + 32 + txm32 < M) {
                        rA[1] = *reinterpret_cast<float4*>(A + (mb + 32 + txm32) * K + kb + 16 + txb32 * 4);
                    }
                    else {
                        rA[1].x = rA[1].y = rA[1].z = rA[1].w = 0;
                    }
                    if (nb + txm32 < N) {
                        rB[0] = *reinterpret_cast<float4*>(B + (nb + txm32) * K + kb + 16 + txb32 * 4);
                    }
                    else {
                        rB[0].x = rB[0].y = rB[0].z = rB[0].w = 0;
                    }
                    if (nb + 32 + txm32 < N) {
                        rB[1] = *reinterpret_cast<float4*>(B + (nb + 32 + txm32) * K + kb + 16 + txb32 * 4);
                    }
                    else {
                        rB[1].x = rB[1].y = rB[1].z = rB[1].w = 0;
                    }
                }
                else {
                    rA[0].x = rA[0].y = rA[0].z = rA[0].w = 0;
                    rA[1].x = rA[1].y = rA[1].z = rA[1].w = 0;
                    rB[0].x = rB[0].y = rB[0].z = rB[0].w = 0;
                    rB[1].x = rB[1].y = rB[1].z = rB[1].w = 0;
                }
            }
            else {
                int t = K - kb - 16 - txb32 * 4;
                if (4 <= t) {
                    if (mb + txm32 < M) {
                        rA[0].x = A[(mb + txm32) * K + kb + 16 + txb32 * 4 + 0];
                        rA[0].y = A[(mb + txm32) * K + kb + 16 + txb32 * 4 + 1];
                        rA[0].z = A[(mb + txm32) * K + kb + 16 + txb32 * 4 + 2];
                        rA[0].w = A[(mb + txm32) * K + kb + 16 + txb32 * 4 + 3];
                    }
                    else {
                        rA[0].x = rA[0].y = rA[0].z = rA[0].w = 0;
                    }
                    if (mb + 32 + txm32 < M) {
                        rA[1].x = A[(mb + 32 + txm32) * K + kb + 16 + txb32 * 4 + 0];
                        rA[1].y = A[(mb + 32 + txm32) * K + kb + 16 + txb32 * 4 + 1];
                        rA[1].z = A[(mb + 32 + txm32) * K + kb + 16 + txb32 * 4 + 2];
                        rA[1].w = A[(mb + 32 + txm32) * K + kb + 16 + txb32 * 4 + 3];
                    }
                    else {
                        rA[1].x = rA[1].y = rA[1].z = rA[1].w = 0;
                    }
                    if (nb + txm32 < N) {
                        rB[0].x = B[(nb + txm32) * K + kb + 16 + txb32 * 4 + 0];
                        rB[0].y = B[(nb + txm32) * K + kb + 16 + txb32 * 4 + 1];
                        rB[0].z = B[(nb + txm32) * K + kb + 16 + txb32 * 4 + 2];
                        rB[0].w = B[(nb + txm32) * K + kb + 16 + txb32 * 4 + 3];
                    }
                    else {
                        rB[0].x = rB[0].y = rB[0].z = rB[0].w = 0;
                    }
                    if (nb + 32 + txm32 < N) {
                        rB[1].x = B[(nb + 32 + txm32) * K + kb + 16 + txb32 * 4 + 0];
                        rB[1].y = B[(nb + 32 + txm32) * K + kb + 16 + txb32 * 4 + 1];
                        rB[1].z = B[(nb + 32 + txm32) * K + kb + 16 + txb32 * 4 + 2];
                        rB[1].w = B[(nb + 32 + txm32) * K + kb + 16 + txb32 * 4 + 3];
                    }
                    else {
                        rB[1].x = rB[1].y = rB[1].z = rB[1].w = 0;
                    }
                }
                else if (3 <= t) {
                    if (mb + txm32 < M) {
                        rA[0].x = A[(mb + txm32) * K + kb + 16 + txb32 * 4 + 0];
                        rA[0].y = A[(mb + txm32) * K + kb + 16 + txb32 * 4 + 1];
                        rA[0].z = A[(mb + txm32) * K + kb + 16 + txb32 * 4 + 2];
                        rA[0].w = 0;
                    }
                    else {
                        rA[0].x = rA[0].y = rA[0].z = rA[0].w = 0;
                    }
                    if (mb + 32 + txm32 < M) {
                        rA[1].x = A[(mb + 32 + txm32) * K + kb + 16 + txb32 * 4 + 0];
                        rA[1].y = A[(mb + 32 + txm32) * K + kb + 16 + txb32 * 4 + 1];
                        rA[1].z = A[(mb + 32 + txm32) * K + kb + 16 + txb32 * 4 + 2];
                        rA[1].w = 0;
                    }
                    else {
                        rA[1].x = rA[1].y = rA[1].z = rA[1].w = 0;
                    }
                    if (nb + txm32 < N) {
                        rB[0].x = B[(nb + txm32) * K + kb + 16 + txb32 * 4 + 0];
                        rB[0].y = B[(nb + txm32) * K + kb + 16 + txb32 * 4 + 1];
                        rB[0].z = B[(nb + txm32) * K + kb + 16 + txb32 * 4 + 2];
                        rB[0].w = 0;
                    }
                    else {
                        rB[0].x = rB[0].y = rB[0].z = rB[0].w = 0;
                    }
                    if (nb + 32 + txm32 < N) {
                        rB[1].x = B[(nb + 32 + txm32) * K + kb + 16 + txb32 * 4 + 0];
                        rB[1].y = B[(nb + 32 + txm32) * K + kb + 16 + txb32 * 4 + 1];
                        rB[1].z = B[(nb + 32 + txm32) * K + kb + 16 + txb32 * 4 + 2];
                        rB[1].w = 0;
                    }
                    else {
                        rB[1].x = rB[1].y = rB[1].z = rB[1].w = 0;
                    }
                }
                else if (2 <= t) {
                    if (mb + txm32 < M) {
                        rA[0].x = A[(mb + txm32) * K + kb + 16 + txb32 * 4 + 0];
                        rA[0].y = A[(mb + txm32) * K + kb + 16 + txb32 * 4 + 1];
                        rA[0].z = rA[0].w = 0;
                    }
                    else {
                        rA[0].x = rA[0].y = rA[0].z = rA[0].w = 0;
                    }
                    if (mb + 32 + txm32 < M) {
                        rA[1].x = A[(mb + 32 + txm32) * K + kb + 16 + txb32 * 4 + 0];
                        rA[1].y = A[(mb + 32 + txm32) * K + kb + 16 + txb32 * 4 + 1];
                        rA[1].z = rA[1].w = 0;
                    }
                    else {
                        rA[1].x = rA[1].y = rA[1].z = rA[1].w = 0;
                    }
                    if (nb + txm32 < N) {
                        rB[0].x = B[(nb + txm32) * K + kb + 16 + txb32 * 4 + 0];
                        rB[0].y = B[(nb + txm32) * K + kb + 16 + txb32 * 4 + 1];
                        rB[0].z = rB[0].w = 0;
                    }
                    else {
                        rB[0].x = rB[0].y = rB[0].z = rB[0].w = 0;
                    }
                    if (nb + 32 + txm32 < N) {
                        rB[1].x = B[(nb + 32 + txm32) * K + kb + 16 + txb32 * 4 + 0];
                        rB[1].y = B[(nb + 32 + txm32) * K + kb + 16 + txb32 * 4 + 1];
                        rB[1].z = rB[1].w = 0;
                    }
                    else {
                        rB[1].x = rB[1].y = rB[1].z = rB[1].w = 0;
                    }
                }
                else if (1 <= t) {
                    if (mb + txm32 < M) {
                        rA[0].x = A[(mb + txm32) * K + kb + 16 + txb32 * 4 + 0];
                        rA[0].y = rA[0].z = rA[0].w = 0;
                    }
                    else {
                        rA[0].x = rA[0].y = rA[0].z = rA[0].w = 0;
                    }
                    if (mb + 32 + txm32 < M) {
                        rA[1].x = A[(mb + 32 + txm32) * K + kb + 16 + txb32 * 4 + 0];
                        rA[1].y = rA[1].z = rA[1].w = 0;
                    }
                    else {
                        rA[1].x = rA[1].y = rA[1].z = rA[1].w = 0;
                    }
                    if (nb + txm32 < N) {
                        rB[0].x = B[(nb + txm32) * K + kb + 16 + txb32 * 4 + 0];
                        rB[0].y = rB[0].z = rB[0].w = 0;
                    }
                    else {
                        rB[0].x = rB[0].y = rB[0].z = rB[0].w = 0;
                    }
                    if (nb + 32 + txm32 < N) {
                        rB[1].x = B[(nb + 32 + txm32) * K + kb + 16 + txb32 * 4 + 0];
                        rB[1].y = rB[1].z = rB[1].w = 0;
                    }
                    else {
                        rB[1].x = rB[1].y = rB[1].z = rB[1].w = 0;
                    }
                }
                else {
                    rA[0].x = rA[0].y = rA[0].z = rA[0].w = 0;
                    rA[1].x = rA[1].y = rA[1].z = rA[1].w = 0;
                    rB[0].x = rB[0].y = rB[0].z = rB[0].w = 0;
                    rB[1].x = rB[1].y = rB[1].z = rB[1].w = 0;
                }
            }
        }

        for (m = 0; m < 8; m += 4) {
            float4 t = *reinterpret_cast<float4*>((*psA)[0] + wmb + m * 4 + txm32b8 * 4);
            rAs[m + 0] = t.x;
            rAs[m + 1] = t.y;
            rAs[m + 2] = t.z;
            rAs[m + 3] = t.w;
        }
        float4 t = *reinterpret_cast<float4*>((*psB)[0] + wnb + txm8 * 4);
        rBs[0] = t.x;
        rBs[1] = t.y;
        rBs[2] = t.z;
        rBs[3] = t.w;

#pragma unroll
        for (k = 0; k < 16; k++) {
            if (k < 15) {
#pragma unroll
                for (m = 0; m < 8; m += 4) {
                    float4 t = *reinterpret_cast<float4*>((*psA)[k + 1] + wmb + m * 4 + txm32b8 * 4);
                    rAsb[m + 0] = t.x;
                    rAsb[m + 1] = t.y;
                    rAsb[m + 2] = t.z;
                    rAsb[m + 3] = t.w;
                }
                float4 t = *reinterpret_cast<float4*>((*psB)[k + 1] + wnb + txm8 * 4);
                rBsb[0] = t.x;
                rBsb[1] = t.y;
                rBsb[2] = t.z;
                rBsb[3] = t.w;
            }

#pragma unroll
            for (m = 0; m < 8; m++) {
#pragma unroll
                for (n = 0; n < 4; n++) {
                    rC[m][n] += rAs[m] * rBs[n];
                }
            }

#pragma unroll
            for (m = 0; m < 8; m++) {
                rAs[m] = rAsb[m];
            }
#pragma unroll
            for (n = 0; n < 4; n++) {
                rBs[n] = rBsb[n];
            }

        }

        psA = (psA == &sA) ? &sAb : &sA;
        psB = (psB == &sB) ? &sBb : &sB;

    }

#pragma unroll
    for (m = 0; m < 8; m += 4) {
#pragma unroll
        for (n = 0; n < 4; n++) {
            int t1 = mb + wmb + m * 4 + txm32b8 * 4 + n;
            int t2 = nb + wnb + txm8 * 4;
            if (t1 < M) {
                if (aligned) {
                    if (t2 < N) {
                        float4 t;
                        t.x = rC[m + n][0];
                        t.y = rC[m + n][1];
                        t.z = rC[m + n][2];
                        t.w = rC[m + n][3];
                        *reinterpret_cast<float4*>(C + t1 * N + t2) = t;
                    }
                }
                else {
                    if (4 <= N - t2) {
                        C[t1 * N + t2 + 0] = rC[m + n][0];
                        C[t1 * N + t2 + 1] = rC[m + n][1];
                        C[t1 * N + t2 + 2] = rC[m + n][2];
                        C[t1 * N + t2 + 3] = rC[m + n][3];
                    }
                    else if (3 <= N - t2) {
                        C[t1 * N + t2 + 0] = rC[m + n][0];
                        C[t1 * N + t2 + 1] = rC[m + n][1];
                        C[t1 * N + t2 + 2] = rC[m + n][2];
                    }
                    else if (2 <= N - t2) {
                        C[t1 * N + t2 + 0] = rC[m + n][0];
                        C[t1 * N + t2 + 1] = rC[m + n][1];
                    }
                    else if (1 <= N - t2) {
                        C[t1 * N + t2 + 0] = rC[m + n][0];
                    }
                }
            }
        }
    }


}

__global__ void sgemm_32x32x16_NT(float* A, float* B, float* C, int M, int N, int K, int aligned) {

    __shared__ __align__(16) float sA[16][32];
    __shared__ __align__(16) float sB[16][32];
    __shared__ __align__(16) float sAb[16][32];
    __shared__ __align__(16) float sBb[16][32];

    int tx = threadIdx.x;
    int mb = blockIdx.z * 64 + (blockIdx.x / 2) * 32;
    int nb = blockIdx.y * 64 + (blockIdx.x % 2) * 32;

    if (mb >= M || nb >= N) {
        return;
    }

    int txm8 = tx % 8;
    int txb8 = tx / 8;
    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm32m4 = txm32 % 4;
    int txm32b4 = txm32 / 4;
    int txm16b2 = txm16 / 2;
    int txm32m2 = txm32 % 2;
    int txm32b16 = txm32 / 16;
    int wmb = 0;
    int wnb = txb32 * 16;

    float rC[4][4];
    float4 rA[2];
    float4 rB[2];
    float rAs[4];
    float rBs[4];
    float rAsb[4];
    float rBsb[4];

    float(*psA)[16][32] = &sA;
    float(*psB)[16][32] = &sB;

    int m, n, k, kb;

#pragma unroll
    for (m = 0; m < 4; m++) {
#pragma unroll
        for (n = 0; n < 4; n++) {
            rC[m][n] = 0;
        }
    }

#pragma unroll
    for (m = 0; m < 2; m++) {
        if (aligned) {
            if (txb16 * 4 < K) {
                if (mb + m * 16 + txm16 < M) {
                    rA[m] = *reinterpret_cast<float4*>(A + (mb + m * 16 + txm16) * K + txb16 * 4);
                }
                else {
                    rA[m].x = rA[m].y = rA[m].z = rA[m].w = 0;
                }
                if (nb + m * 16 + txm16 < N) {
                    rB[m] = *reinterpret_cast<float4*>(B + (nb + m * 16 + txm16) * K + txb16 * 4);
                }
                else {
                    rB[m].x = rB[m].y = rB[m].z = rB[m].w = 0;
                }
            }
            else {
                rA[m].x = rA[m].y = rA[m].z = rA[m].w = 0;
                rB[m].x = rB[m].y = rB[m].z = rB[m].w = 0;
            }
        }
        else {
            int t = K - txb16 * 4;
            if (mb + m * 16 + txm16 < M) {
                if (4 <= t) {
                    rA[m].x = A[(mb + m * 16 + txm16) * K + txb16 * 4 + 0];
                    rA[m].y = A[(mb + m * 16 + txm16) * K + txb16 * 4 + 1];
                    rA[m].z = A[(mb + m * 16 + txm16) * K + txb16 * 4 + 2];
                    rA[m].w = A[(mb + m * 16 + txm16) * K + txb16 * 4 + 3];
                }
                else if (3 <= t) {
                    rA[m].x = A[(mb + m * 16 + txm16) * K + txb16 * 4 + 0];
                    rA[m].y = A[(mb + m * 16 + txm16) * K + txb16 * 4 + 1];
                    rA[m].z = A[(mb + m * 16 + txm16) * K + txb16 * 4 + 2];
                    rA[m].w = 0;
                }
                else if (2 <= t) {
                    rA[m].x = A[(mb + m * 16 + txm16) * K + txb16 * 4 + 0];
                    rA[m].y = A[(mb + m * 16 + txm16) * K + txb16 * 4 + 1];
                    rA[m].z = rA[m].w = 0;
                }
                else if (1 <= t) {
                    rA[m].x = A[(mb + m * 16 + txm16) * K + txb16 * 4 + 0];
                    rA[m].y = rA[m].z = rA[m].w = 0;
                }
                else {
                    rA[m].x = rA[m].y = rA[m].z = rA[m].w = 0;
                }
            }
            else {
                rA[m].x = rA[m].y = rA[m].z = rA[m].w = 0;
            }
            if (nb + m * 16 + txm16 < N) {
                if (4 <= t) {
                    rB[m].x = B[(nb + m * 16 + txm16) * K + txb16 * 4 + 0];
                    rB[m].y = B[(nb + m * 16 + txm16) * K + txb16 * 4 + 1];
                    rB[m].z = B[(nb + m * 16 + txm16) * K + txb16 * 4 + 2];
                    rB[m].w = B[(nb + m * 16 + txm16) * K + txb16 * 4 + 3];
                }
                else if (3 <= t) {
                    rB[m].x = B[(nb + m * 16 + txm16) * K + txb16 * 4 + 0];
                    rB[m].y = B[(nb + m * 16 + txm16) * K + txb16 * 4 + 1];
                    rB[m].z = B[(nb + m * 16 + txm16) * K + txb16 * 4 + 2];
                    rB[m].w = 0;
                }
                else if (2 <= t) {
                    rB[m].x = B[(nb + m * 16 + txm16) * K + txb16 * 4 + 0];
                    rB[m].y = B[(nb + m * 16 + txm16) * K + txb16 * 4 + 1];
                    rB[m].z = rB[m].w = 0;
                }
                else if (1 <= t) {
                    rB[m].x = B[(nb + m * 16 + txm16) * K + txb16 * 4 + 0];
                    rB[m].y = rB[m].z = rB[m].w = 0;
                }
                else {
                    rB[m].x = rB[m].y = rB[m].z = rB[m].w = 0;
                }
            }
            else {
                rB[m].x = rB[m].y = rB[m].z = rB[m].w = 0;
            }
        }
    }

    for (kb = 0; kb < K; kb += 16) {

#pragma unroll
        for (m = 0; m < 2; m++) {
            (*psA)[txb16 * 4 + 0][m * 16 + txm16] = rA[m].x;
            (*psA)[txb16 * 4 + 1][m * 16 + txm16] = rA[m].y;
            (*psA)[txb16 * 4 + 2][m * 16 + txm16] = rA[m].z;
            (*psA)[txb16 * 4 + 3][m * 16 + txm16] = rA[m].w;

            (*psB)[txb16 * 4 + 0][m * 16 + txm16] = rB[m].x;
            (*psB)[txb16 * 4 + 1][m * 16 + txm16] = rB[m].y;
            (*psB)[txb16 * 4 + 2][m * 16 + txm16] = rB[m].z;
            (*psB)[txb16 * 4 + 3][m * 16 + txm16] = rB[m].w;
        }

        __syncthreads();

        if (kb + 16 < K) {
#pragma unroll
            for (m = 0; m < 2; m++) {
                if (aligned) {
                    if (kb + 16 + txb16 * 4 < K) {
                        if (mb + m * 16 + txm16 < M) {
                            rA[m] = *reinterpret_cast<float4*>(A + (mb + m * 16 + txm16) * K + kb + 16 + txb16 * 4);
                        }
                        else {
                            rA[m].x = rA[m].y = rA[m].z = rA[m].w = 0;
                        }
                        if (nb + m * 16 + txm16 < N) {
                            rB[m] = *reinterpret_cast<float4*>(B + (nb + m * 16 + txm16) * K + kb + 16 + txb16 * 4);
                        }
                        else {
                            rB[m].x = rB[m].y = rB[m].z = rB[m].w = 0;
                        }
                    }
                    else {
                        rA[m].x = rA[m].y = rA[m].z = rA[m].w = 0;
                        rB[m].x = rB[m].y = rB[m].z = rB[m].w = 0;
                    }
                }
                else {
                    int t = K - kb - 16 - txb16 * 4;
                    if (mb + m * 16 + txm16 < M) {
                        if (4 <= t) {
                            rA[m].x = A[(mb + m * 16 + txm16) * K + kb + 16 + txb16 * 4 + 0];
                            rA[m].y = A[(mb + m * 16 + txm16) * K + kb + 16 + txb16 * 4 + 1];
                            rA[m].z = A[(mb + m * 16 + txm16) * K + kb + 16 + txb16 * 4 + 2];
                            rA[m].w = A[(mb + m * 16 + txm16) * K + kb + 16 + txb16 * 4 + 3];
                        }
                        else if (3 <= t) {
                            rA[m].x = A[(mb + m * 16 + txm16) * K + kb + 16 + txb16 * 4 + 0];
                            rA[m].y = A[(mb + m * 16 + txm16) * K + kb + 16 + txb16 * 4 + 1];
                            rA[m].z = A[(mb + m * 16 + txm16) * K + kb + 16 + txb16 * 4 + 2];
                            rA[m].w = 0;
                        }
                        else if (2 <= t) {
                            rA[m].x = A[(mb + m * 16 + txm16) * K + kb + 16 + txb16 * 4 + 0];
                            rA[m].y = A[(mb + m * 16 + txm16) * K + kb + 16 + txb16 * 4 + 1];
                            rA[m].z = rA[m].w = 0;
                        }
                        else if (1 <= t) {
                            rA[m].x = A[(mb + m * 16 + txm16) * K + kb + 16 + txb16 * 4 + 0];
                            rA[m].y = rA[m].z = rA[m].w = 0;
                        }
                        else {
                            rA[m].x = rA[m].y = rA[m].z = rA[m].w = 0;
                        }
                    }
                    else {
                        rA[m].x = rA[m].y = rA[m].z = rA[m].w = 0;
                    }
                    if (nb + m * 16 + txm16 < N) {
                        if (4 <= t) {
                            rB[m].x = B[(nb + m * 16 + txm16) * K + kb + 16 + txb16 * 4 + 0];
                            rB[m].y = B[(nb + m * 16 + txm16) * K + kb + 16 + txb16 * 4 + 1];
                            rB[m].z = B[(nb + m * 16 + txm16) * K + kb + 16 + txb16 * 4 + 2];
                            rB[m].w = B[(nb + m * 16 + txm16) * K + kb + 16 + txb16 * 4 + 3];
                        }
                        else if (3 <= t) {
                            rB[m].x = B[(nb + m * 16 + txm16) * K + kb + 16 + txb16 * 4 + 0];
                            rB[m].y = B[(nb + m * 16 + txm16) * K + kb + 16 + txb16 * 4 + 1];
                            rB[m].z = B[(nb + m * 16 + txm16) * K + kb + 16 + txb16 * 4 + 2];
                            rB[m].w = 0;
                        }
                        else if (2 <= t) {
                            rB[m].x = B[(nb + m * 16 + txm16) * K + kb + 16 + txb16 * 4 + 0];
                            rB[m].y = B[(nb + m * 16 + txm16) * K + kb + 16 + txb16 * 4 + 1];
                            rB[m].z = rB[m].w = 0;
                        }
                        else if (1 <= t) {
                            rB[m].x = B[(nb + m * 16 + txm16) * K + kb + 16 + txb16 * 4 + 0];
                            rB[m].y = rB[m].z = rB[m].w = 0;
                        }
                        else {
                            rB[m].x = rB[m].y = rB[m].z = rB[m].w = 0;
                        }
                    }
                    else {
                        rB[m].x = rB[m].y = rB[m].z = rB[m].w = 0;
                    }
                }
            }
        }

        // COMPUTE -------------------------

        float4 tA = *reinterpret_cast<float4*>((*psA)[0] + wmb + txm16b2 * 4);
        rAs[0] = tA.x;
        rAs[1] = tA.y;
        rAs[2] = tA.z;
        rAs[3] = tA.w;
        float4 tB = *reinterpret_cast<float4*>((*psB)[0] + wnb + txm32b16 * 8 + txm32m2 * 4);
        rBs[0] = tB.x;
        rBs[1] = tB.y;
        rBs[2] = tB.z;
        rBs[3] = tB.w;

#pragma unroll
        for (k = 0; k < 16; k++) {

            if (k < 15) {
                float4 tA = *reinterpret_cast<float4*>((*psA)[k + 1] + wmb + txm16b2 * 4);
                rAsb[0] = tA.x;
                rAsb[1] = tA.y;
                rAsb[2] = tA.z;
                rAsb[3] = tA.w;
                float4 tB = *reinterpret_cast<float4*>((*psB)[k + 1] + wnb + txm32b16 * 8 + txm32m2 * 4);
                rBsb[0] = tB.x;
                rBsb[1] = tB.y;
                rBsb[2] = tB.z;
                rBsb[3] = tB.w;
            }

#pragma unroll
            for (m = 0; m < 4; m++) {
#pragma unroll
                for (n = 0; n < 4; n++) {
                    rC[m][n] += rAs[m] * rBs[n];
                }
            }

#pragma unroll
            for (m = 0; m < 4; m++) {
                rAs[m] = rAsb[m];
                rBs[m] = rBsb[m];
            }
        }

        // ---------------------------------

        psA = (psA == &sA) ? &sAb : &sA;
        psB = (psB == &sB) ? &sBb : &sB;

    }

#pragma unroll
    for (m = 0; m < 4; m++) {
        if (mb + wmb + txm16b2 * 4 + m < M) {
            if (aligned) {
                if (nb + wnb + txm32b16 * 8 + txm32m2 * 4 < N) {
                    float4 t;
                    t.x = rC[m][0];
                    t.y = rC[m][1];
                    t.z = rC[m][2];
                    t.w = rC[m][3];
                    *reinterpret_cast<float4*>(C + (mb + wmb + txm16b2 * 4 + m) * N + nb + wnb + txm32b16 * 8 + txm32m2 * 4) = t;
                }
            }
            else {
                int t = N - (nb + wnb + txm32b16 * 8 + txm32m2 * 4);
                if (4 <= t) {
                    C[(mb + wmb + txm16b2 * 4 + m) * N + N - t + 0] = rC[m][0];
                    C[(mb + wmb + txm16b2 * 4 + m) * N + N - t + 1] = rC[m][1];
                    C[(mb + wmb + txm16b2 * 4 + m) * N + N - t + 2] = rC[m][2];
                    C[(mb + wmb + txm16b2 * 4 + m) * N + N - t + 3] = rC[m][3];
                }
                else if (3 <= t) {
                    C[(mb + wmb + txm16b2 * 4 + m) * N + N - t + 0] = rC[m][0];
                    C[(mb + wmb + txm16b2 * 4 + m) * N + N - t + 1] = rC[m][1];
                    C[(mb + wmb + txm16b2 * 4 + m) * N + N - t + 2] = rC[m][2];
                }
                else if (2 <= t) {
                    C[(mb + wmb + txm16b2 * 4 + m) * N + N - t + 0] = rC[m][0];
                    C[(mb + wmb + txm16b2 * 4 + m) * N + N - t + 1] = rC[m][1];
                }
                else if (1 <= t) {
                    C[(mb + wmb + txm16b2 * 4 + m) * N + N - t + 0] = rC[m][0];
                }
            }
        }
    }

}

__global__ void sgemm_4x4x256_NT(float* A, float* B, float* C, int M, int N, int K, int aligned) {

    __shared__ float sA[4][256];
    __shared__ float sAb[4][256];
    __shared__ float sB[4][256];
    __shared__ float sBb[4][256];
    __shared__ float sC[8][4][4];

    int tx = threadIdx.x;
    int mb = blockIdx.x * 4;
    int nb = blockIdx.y * 4;

    int txm8 = tx % 8;
    int txb8 = tx / 8;
    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm64 = tx % 64;
    int txb64 = tx / 64;
    int txm32m4 = txm32 % 4;
    int txm32b4 = txm32 / 4;
    int txm16b2 = txm16 / 2;
    int txm32m2 = txm32 % 2;
    int txm32b16 = txm32 / 16;
    int wmb = 0;
    int wnb = txb32 * 16;

    float rC[4][4];
    float4 rA, rAb;
    float4 rB, rBb;

    float(*psA)[4][256] = &sA;
    float(*psB)[4][256] = &sB;

    int m, n, k, kb;

#pragma unroll
    for (m = 0; m < 4; m++) {
#pragma unroll
        for (n = 0; n < 4; n++) {
            rC[m][n] = 0;
        }
    }

    if (mb + txb64 < M) {
        if (aligned) {
            if (txm64 * 4 < K) {
                rA = *reinterpret_cast<float4*>(A + (mb + txb64) * K + txm64 * 4);
            }
            else {
                rA.x = rA.y = rA.z = rA.w = 0;
            }
        }
        else {
            int t = K - txm64 * 4;
            if (4 <= t) {
                rA.x = A[(mb + txb64) * K + K - t + 0];
                rA.y = A[(mb + txb64) * K + K - t + 1];
                rA.z = A[(mb + txb64) * K + K - t + 2];
                rA.w = A[(mb + txb64) * K + K - t + 3];
            }
            else if (3 <= t) {
                rA.x = A[(mb + txb64) * K + K - t + 0];
                rA.y = A[(mb + txb64) * K + K - t + 1];
                rA.z = A[(mb + txb64) * K + K - t + 2];
                rA.w = 0;
            }
            else if (2 <= t) {
                rA.x = A[(mb + txb64) * K + K - t + 0];
                rA.y = A[(mb + txb64) * K + K - t + 1];
                rA.z = rA.w = 0;
            }
            else if (1 <= t) {
                rA.x = A[(mb + txb64) * K + K - t + 0];
                rA.y = rA.z = rA.w = 0;
            }
            else {
                rA.x = rA.y = rA.z = rA.w = 0;
            }
        }
    }
    else {
        rA.x = rA.y = rA.z = rA.w = 0;
    }
    if (nb + txb64 < N) {
        if (aligned) {
            if (txm64 * 4 < K) {
                rB = *reinterpret_cast<float4*>(B + (nb + txb64) * K + txm64 * 4);
            }
            else {
                rB.x = rB.y = rB.z = rB.w = 0;
            }
        }
        else {
            int t = K - txm64 * 4;
            if (4 <= t) {
                rB.x = B[(nb + txb64) * K + K - t + 0];
                rB.y = B[(nb + txb64) * K + K - t + 1];
                rB.z = B[(nb + txb64) * K + K - t + 2];
                rB.w = B[(nb + txb64) * K + K - t + 3];
            }
            else if (3 <= t) {
                rB.x = B[(nb + txb64) * K + K - t + 0];
                rB.y = B[(nb + txb64) * K + K - t + 1];
                rB.z = B[(nb + txb64) * K + K - t + 2];
                rB.w = 0;
            }
            else if (2 <= t) {
                rB.x = B[(nb + txb64) * K + K - t + 0];
                rB.y = B[(nb + txb64) * K + K - t + 1];
                rB.z = rB.w = 0;
            }
            else if (1 <= t) {
                rB.x = B[(nb + txb64) * K + K - t + 0];
                rB.y = rB.z = rB.w = 0;
            }
            else {
                rB.x = rB.y = rB.z = rB.w = 0;
            }
        }
    }
    else {
        rB.x = rB.y = rB.z = rB.w = 0;
    }

    for (kb = 0; kb < K; kb += 256) {

        (*psA)[txb64][txm64 * 4 + 0] = rA.x;
        (*psA)[txb64][txm64 * 4 + 1] = rA.y;
        (*psA)[txb64][txm64 * 4 + 2] = rA.z;
        (*psA)[txb64][txm64 * 4 + 3] = rA.w;

        (*psB)[txb64][txm64 * 4 + 0] = rB.x;
        (*psB)[txb64][txm64 * 4 + 1] = rB.y;
        (*psB)[txb64][txm64 * 4 + 2] = rB.z;
        (*psB)[txb64][txm64 * 4 + 3] = rB.w;

        __syncthreads();

        if (kb + 256 < K) {
            if (mb + txb64 < M) {
                if (aligned) {
                    if (kb + 256 + txm64 * 4 < K) {
                        rA = *reinterpret_cast<float4*>(A + (mb + txb64) * K + kb + 256 + txm64 * 4);
                    }
                    else {
                        rA.x = rA.y = rA.z = rA.w = 0;
                    }
                }
                else {
                    int t = K - kb - 256 - txm64 * 4;
                    if (4 <= t) {
                        rA.x = A[(mb + txb64) * K + K - t + 0];
                        rA.y = A[(mb + txb64) * K + K - t + 1];
                        rA.z = A[(mb + txb64) * K + K - t + 2];
                        rA.w = A[(mb + txb64) * K + K - t + 3];
                    }
                    else if (3 <= t) {
                        rA.x = A[(mb + txb64) * K + K - t + 0];
                        rA.y = A[(mb + txb64) * K + K - t + 1];
                        rA.z = A[(mb + txb64) * K + K - t + 2];
                        rA.w = 0;
                    }
                    else if (2 <= t) {
                        rA.x = A[(mb + txb64) * K + K - t + 0];
                        rA.y = A[(mb + txb64) * K + K - t + 1];
                        rA.z = rA.w = 0;
                    }
                    else if (1 <= t) {
                        rA.x = A[(mb + txb64) * K + K - t + 0];
                        rA.y = rA.z = rA.w = 0;
                    }
                    else {
                        rA.x = rA.y = rA.z = rA.w = 0;
                    }
                }
            }
            else {
                rA.x = rA.y = rA.z = rA.w = 0;
            }
            if (nb + txb64 < N) {
                if (aligned) {
                    if (kb + 256 + txm64 * 4 < K) {
                        rB = *reinterpret_cast<float4*>(B + (nb + txb64) * K + kb + 256 + txm64 * 4);
                    }
                    else {
                        rB.x = rB.y = rB.z = rB.w = 0;
                    }
                }
                else {
                    int t = K - kb - 256 - txm64 * 4;
                    if (4 <= t) {
                        rB.x = B[(nb + txb64) * K + K - t + 0];
                        rB.y = B[(nb + txb64) * K + K - t + 1];
                        rB.z = B[(nb + txb64) * K + K - t + 2];
                        rB.w = B[(nb + txb64) * K + K - t + 3];
                    }
                    else if (3 <= t) {
                        rB.x = B[(nb + txb64) * K + K - t + 0];
                        rB.y = B[(nb + txb64) * K + K - t + 1];
                        rB.z = B[(nb + txb64) * K + K - t + 2];
                        rB.w = 0;
                    }
                    else if (2 <= t) {
                        rB.x = B[(nb + txb64) * K + K - t + 0];
                        rB.y = B[(nb + txb64) * K + K - t + 1];
                        rB.z = rB.w = 0;
                    }
                    else if (1 <= t) {
                        rB.x = B[(nb + txb64) * K + K - t + 0];
                        rB.y = rB.z = rB.w = 0;
                    }
                    else {
                        rB.x = rB.y = rB.z = rB.w = 0;
                    }
                }
            }
            else {
                rB.x = rB.y = rB.z = rB.w = 0;
            }
        }

        // COMPUTE -------------------------

#pragma unroll
        for (m = 0; m < 4; m++) {
#pragma unroll
            for (n = 0; n < 4; n++) {
                rC[m][n] += (*psA)[m][tx] * (*psB)[n][tx];
            }
        }

        psA = (psA == &sA) ? &sAb : &sA;
        psB = (psB == &sB) ? &sBb : &sB;

    }

#pragma unroll
    for (m = 0; m < 4; m++) {
#pragma unroll
        for (n = 0; n < 4; n++) {
            float t = __shfl_sync(FULL_MASK, rC[m][n], txm32 + 16, 32);
            rC[m][n] += t;
            t = __shfl_sync(FULL_MASK, rC[m][n], txm32 + 8, 32);
            rC[m][n] += t;
            t = __shfl_sync(FULL_MASK, rC[m][n], txm32 + 4, 32);
            rC[m][n] += t;
            t = __shfl_sync(FULL_MASK, rC[m][n], txm32 + 2, 32);
            rC[m][n] += t;
            t = __shfl_sync(FULL_MASK, rC[m][n], txm32 + 1, 32);
            rC[m][n] += t;
            if (txm32 == 0) {
                sC[txb32][m][n] = rC[m][n];
            }
        }
    }

    __syncthreads();

    if (tx == 0) {
        if (aligned) {
#pragma unroll
            for (m = 0; m < 4; m++) {
                if (mb + m < M) {
                    float4 t;
                    t.x = rC[m][0] + sC[1][m][0] + sC[2][m][0] + sC[3][m][0] + sC[4][m][0] + sC[5][m][0] + sC[6][m][0] + sC[7][m][0];
                    t.y = rC[m][1] + sC[1][m][1] + sC[2][m][1] + sC[3][m][1] + sC[4][m][1] + sC[5][m][1] + sC[6][m][1] + sC[7][m][1];
                    t.z = rC[m][2] + sC[1][m][2] + sC[2][m][2] + sC[3][m][2] + sC[4][m][2] + sC[5][m][2] + sC[6][m][2] + sC[7][m][2];
                    t.w = rC[m][3] + sC[1][m][3] + sC[2][m][3] + sC[3][m][3] + sC[4][m][3] + sC[5][m][3] + sC[6][m][3] + sC[7][m][3];
                    *reinterpret_cast<float4*>(C + (mb + m) * N + nb) = t;
                }
            }
        }
        else {
            int t = N - nb;
#pragma unroll
            for (m = 0; m < 4; m++) {
                if (mb + m < M) {
                    if (4 <= t) {
                        C[(mb + m) * N + nb + 0] = rC[m][0] + sC[1][m][0] + sC[2][m][0] + sC[3][m][0] + sC[4][m][0] + sC[5][m][0] + sC[6][m][0] + sC[7][m][0];
                        C[(mb + m) * N + nb + 1] = rC[m][1] + sC[1][m][1] + sC[2][m][1] + sC[3][m][1] + sC[4][m][1] + sC[5][m][1] + sC[6][m][1] + sC[7][m][1];
                        C[(mb + m) * N + nb + 2] = rC[m][2] + sC[1][m][2] + sC[2][m][2] + sC[3][m][2] + sC[4][m][2] + sC[5][m][2] + sC[6][m][2] + sC[7][m][2];
                        C[(mb + m) * N + nb + 3] = rC[m][3] + sC[1][m][3] + sC[2][m][3] + sC[3][m][3] + sC[4][m][3] + sC[5][m][3] + sC[6][m][3] + sC[7][m][3];
                    }
                    else if (3 <= t) {
                        C[(mb + m) * N + nb + 0] = rC[m][0] + sC[1][m][0] + sC[2][m][0] + sC[3][m][0] + sC[4][m][0] + sC[5][m][0] + sC[6][m][0] + sC[7][m][0];
                        C[(mb + m) * N + nb + 1] = rC[m][1] + sC[1][m][1] + sC[2][m][1] + sC[3][m][1] + sC[4][m][1] + sC[5][m][1] + sC[6][m][1] + sC[7][m][1];
                        C[(mb + m) * N + nb + 2] = rC[m][2] + sC[1][m][2] + sC[2][m][2] + sC[3][m][2] + sC[4][m][2] + sC[5][m][2] + sC[6][m][2] + sC[7][m][2];
                    }
                    else if (2 <= t) {
                        C[(mb + m) * N + nb + 0] = rC[m][0] + sC[1][m][0] + sC[2][m][0] + sC[3][m][0] + sC[4][m][0] + sC[5][m][0] + sC[6][m][0] + sC[7][m][0];
                        C[(mb + m) * N + nb + 1] = rC[m][1] + sC[1][m][1] + sC[2][m][1] + sC[3][m][1] + sC[4][m][1] + sC[5][m][1] + sC[6][m][1] + sC[7][m][1];
                    }
                    else if (1 <= t) {
                        C[(mb + m) * N + nb + 0] = rC[m][0] + sC[1][m][0] + sC[2][m][0] + sC[3][m][0] + sC[4][m][0] + sC[5][m][0] + sC[6][m][0] + sC[7][m][0];
                    }
                }
            }
        }
    }

}


__global__ void sgemm_128x128x8_TN(float* A, float* B, float* C, int M, int N, int K, int aligned) {

    __shared__ __align__(16) float sA[8][132];
    __shared__ __align__(16) float sB[8][132];
    __shared__ __align__(16) float sAb[8][132];
    __shared__ __align__(16) float sBb[8][132];

    int tx = threadIdx.x;
    int mb = blockIdx.z * 256 + (blockIdx.x / 2) * 128;
    int nb = blockIdx.y * 256 + (blockIdx.x % 2) * 128;

    if (mb >= M || nb >= N) {
        return;
    }

    float rC[8][8];
    float4 rA;
    float4 rB;
    float rAs[8];
    float rBs[8];
    float rAsb[8];
    float rBsb[8];

    float(*psA)[8][132] = &sA;
    float(*psB)[8][132] = &sB;

    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm2 = tx % 2;
    int txb2 = tx / 2;
    int txm8 = tx % 8;
    int txb8 = tx / 8;
    int txm128 = tx % 128;
    int txb128 = tx / 128;
    int txm32b4 = txm32 / 4;

    int wnb = (txb32 / 4) * 64;
    int wmb = (txb32 % 4) * 32;

    int m, n, k, kb, kbb;

    float4 f4;
    float8 f8;

    int sAi1 = wmb + (txm32 % 2) * 8 + (txm32 / 16) * 16;
    int sBi1 = wnb + (txm32 / 2) * 4;
    int sBi2 = wnb + (32 + (txm32 / 2) * 4) % 64;

#pragma unroll
    for (m = 0; m < 8; m++) {
#pragma unroll
        for (n = 0; n < 8; n++) {
            rC[m][n] = 0;
        }
    }

    if (aligned) {
        if (txb32 < K && mb + txm32 * 4 < M) {
            rA = *reinterpret_cast<float4*>(A + txb32 * M + mb + txm32 * 4);
        }
        else {
            rA.x = rA.y = rA.z = rA.w = 0;
        }
        if (txb32 < K && nb + txm32 * 4 < N) {
            rB = *reinterpret_cast<float4*>(B + (txb32) * N + nb + txm32 * 4);
        }
        else {
            rB.x = rB.y = rB.z = rB.w = 0;
        }
    }
    else {
        if (txb32 < K) {
            int t = M - mb - txm32 * 4;
            if (4 <= t) {
                rA.x = A[txb32 * M + mb + txm32 * 4 + 0];
                rA.y = A[txb32 * M + mb + txm32 * 4 + 1];
                rA.z = A[txb32 * M + mb + txm32 * 4 + 2];
                rA.w = A[txb32 * M + mb + txm32 * 4 + 3];
            }
            else if (3 <= t) {
                rA.x = A[txb32 * M + mb + txm32 * 4 + 0];
                rA.y = A[txb32 * M + mb + txm32 * 4 + 1];
                rA.z = A[txb32 * M + mb + txm32 * 4 + 2];
                rA.w = 0;
            }
            else if (2 <= t) {
                rA.x = A[txb32 * M + mb + txm32 * 4 + 0];
                rA.y = A[txb32 * M + mb + txm32 * 4 + 1];
                rA.z = rA.w = 0;
            }
            else if (1 <= t) {
                rA.x = A[txb32 * M + mb + txm32 * 4 + 0];
                rA.y = rA.z = rA.w = 0;
            }
            else {
                rA.x = rA.y = rA.z = rA.w = 0;
            }
            t = N - nb - txm32 * 4;
            if (4 <= t) {
                rB.x = B[txb32 * N + nb + txm32 * 4 + 0];
                rB.y = B[txb32 * N + nb + txm32 * 4 + 1];
                rB.z = B[txb32 * N + nb + txm32 * 4 + 2];
                rB.w = B[txb32 * N + nb + txm32 * 4 + 3];
            }
            else if (3 <= t) {
                rB.x = B[txb32 * N + nb + txm32 * 4 + 0];
                rB.y = B[txb32 * N + nb + txm32 * 4 + 1];
                rB.z = B[txb32 * N + nb + txm32 * 4 + 2];
                rB.w = 0;
            }
            else if (2 <= t) {
                rB.x = B[txb32 * N + nb + txm32 * 4 + 0];
                rB.y = B[txb32 * N + nb + txm32 * 4 + 1];
                rB.z = rB.w = 0;
            }
            else if (1 <= t) {
                rB.x = B[txb32 * N + nb + txm32 * 4 + 0];
                rB.y = rB.z = rB.w = 0;
            }
            else {
                rB.x = rB.y = rB.z = rB.w = 0;
            }
        }
        else {
            rA.x = rA.y = rA.z = rA.w = 0;
            rB.x = rB.y = rB.z = rB.w = 0;
        }
    }

    for (kb = 0; kb < K; kb += 8) {

        (*psA)[txb32][txm32 * 4 + 0] = rA.x;
        (*psA)[txb32][txm32 * 4 + 1] = rA.y;
        (*psA)[txb32][txm32 * 4 + 2] = rA.z;
        (*psA)[txb32][txm32 * 4 + 3] = rA.w;

        (*psB)[txb32][txm32 * 4 + 0] = rB.x;
        (*psB)[txb32][txm32 * 4 + 1] = rB.y;
        (*psB)[txb32][txm32 * 4 + 2] = rB.z;
        (*psB)[txb32][txm32 * 4 + 3] = rB.w;

        __syncthreads();

        if (kb + 8 < K) {
            if (aligned) {
                if (kb + 8 + txb32 < K && mb + txm32 * 4 < M) {
                    rA = *reinterpret_cast<float4*>(A + (kb + 8 + txb32) * M + mb + txm32 * 4);
                }
                else {
                    rA.x = rA.y = rA.z = rA.w = 0;
                }
                if (kb + 8 + txb32 < K && nb + txm32 * 4 < N) {
                    rB = *reinterpret_cast<float4*>(B + (kb + 8 + txb32) * N + nb + txm32 * 4);
                }
                else {
                    rB.x = rB.y = rB.z = rB.w = 0;
                }
            }
            else {
                if (kb + 8 + txb32 < K) {
                    int t = M - mb - txm32 * 4;
                    if (4 <= t) {
                        rA.x = A[(kb + 8 + txb32) * M + mb + txm32 * 4 + 0];
                        rA.y = A[(kb + 8 + txb32) * M + mb + txm32 * 4 + 1];
                        rA.z = A[(kb + 8 + txb32) * M + mb + txm32 * 4 + 2];
                        rA.w = A[(kb + 8 + txb32) * M + mb + txm32 * 4 + 3];
                    }
                    else if (3 <= t) {
                        rA.x = A[(kb + 8 + txb32) * M + mb + txm32 * 4 + 0];
                        rA.y = A[(kb + 8 + txb32) * M + mb + txm32 * 4 + 1];
                        rA.z = A[(kb + 8 + txb32) * M + mb + txm32 * 4 + 2];
                        rA.w = 0;
                    }
                    else if (2 <= t) {
                        rA.x = A[(kb + 8 + txb32) * M + mb + txm32 * 4 + 0];
                        rA.y = A[(kb + 8 + txb32) * M + mb + txm32 * 4 + 1];
                        rA.z = rA.w = 0;
                    }
                    else if (1 <= t) {
                        rA.x = A[(kb + 8 + txb32) * M + mb + txm32 * 4 + 0];
                        rA.y = rA.z = rA.w = 0;
                    }
                    else {
                        rA.x = rA.y = rA.z = rA.w = 0;
                    }
                    t = N - nb - txm32 * 4;
                    if (4 <= t) {
                        rB.x = B[(kb + 8 + txb32) * N + nb + txm32 * 4 + 0];
                        rB.y = B[(kb + 8 + txb32) * N + nb + txm32 * 4 + 1];
                        rB.z = B[(kb + 8 + txb32) * N + nb + txm32 * 4 + 2];
                        rB.w = B[(kb + 8 + txb32) * N + nb + txm32 * 4 + 3];
                    }
                    else if (3 <= t) {
                        rB.x = B[(kb + 8 + txb32) * N + nb + txm32 * 4 + 0];
                        rB.y = B[(kb + 8 + txb32) * N + nb + txm32 * 4 + 1];
                        rB.z = B[(kb + 8 + txb32) * N + nb + txm32 * 4 + 2];
                        rB.w = 0;
                    }
                    else if (2 <= t) {
                        rB.x = B[(kb + 8 + txb32) * N + nb + txm32 * 4 + 0];
                        rB.y = B[(kb + 8 + txb32) * N + nb + txm32 * 4 + 1];
                        rB.z = rB.w = 0;
                    }
                    else if (1 <= t) {
                        rB.x = B[(kb + 8 + txb32) * N + nb + txm32 * 4 + 0];
                        rB.y = rB.z = rB.w = 0;
                    }
                    else {
                        rB.x = rB.y = rB.z = rB.w = 0;
                    }
                }
                else {
                    rA.x = rA.y = rA.z = rA.w = 0;
                    rB.x = rB.y = rB.z = rB.w = 0;
                }
            }
        }

        // COMPUTE -------------------

        f8 = *reinterpret_cast<float8*>((*psA)[0] + sAi1);
        rAs[0] = f8.a;
        rAs[1] = f8.b;
        rAs[2] = f8.c;
        rAs[3] = f8.d;
        rAs[4] = f8.e;
        rAs[5] = f8.f;
        rAs[6] = f8.g;
        rAs[7] = f8.h;
        f4 = *reinterpret_cast<float4*>((*psB)[0] + sBi1);
        rBs[0] = f4.x;
        rBs[1] = f4.y;
        rBs[2] = f4.z;
        rBs[3] = f4.w;
        f4 = *reinterpret_cast<float4*>((*psB)[0] + sBi2);
        rBs[4] = f4.x;
        rBs[5] = f4.y;
        rBs[6] = f4.z;
        rBs[7] = f4.w;

#pragma unroll
        for (k = 0; k < 8; k++) {

            if (k < 7) {
                f8 = *reinterpret_cast<float8*>((*psA)[k + 1] + sAi1);
                rAsb[0] = f8.a;
                rAsb[1] = f8.b;
                rAsb[2] = f8.c;
                rAsb[3] = f8.d;
                rAsb[4] = f8.e;
                rAsb[5] = f8.f;
                rAsb[6] = f8.g;
                rAsb[7] = f8.h;
                f4 = *reinterpret_cast<float4*>((*psB)[k + 1] + sBi1);
                rBsb[0] = f4.x;
                rBsb[1] = f4.y;
                rBsb[2] = f4.z;
                rBsb[3] = f4.w;
                f4 = *reinterpret_cast<float4*>((*psB)[k + 1] + sBi2);
                rBsb[4] = f4.x;
                rBsb[5] = f4.y;
                rBsb[6] = f4.z;
                rBsb[7] = f4.w;
            }

#pragma unroll
            for (m = 0; m < 8; m++) {
#pragma unroll
                for (n = 0; n < 8; n++) {
                    rC[m][n] += rAs[m] * rBs[n];
                }
            }

#pragma unroll
            for (m = 0; m < 8; m++) {
                rAs[m] = rAsb[m];
                rBs[m] = rBsb[m];
            }

        }

        // ---------------------------

        if (psA == &sA) {
            psA = &sAb;
            psB = &sBb;
        }
        else {
            psA = &sA;
            psB = &sB;
        }

    }

#pragma unroll
    for (m = 0; m < 8; m++) {
        if (mb + sAi1 + m < M) {
            if (aligned) {
                if (nb + sBi1 < N) {
                    float4 t;
                    t.x = rC[m][0];
                    t.y = rC[m][1];
                    t.z = rC[m][2];
                    t.w = rC[m][3];
                    *reinterpret_cast<float4*>(C + (mb + sAi1 + m) * N + nb + sBi1) = t;
                }
                if (nb + sBi2 < N) {
                    float4 t;
                    t.x = rC[m][4];
                    t.y = rC[m][5];
                    t.z = rC[m][6];
                    t.w = rC[m][7];
                    *reinterpret_cast<float4*>(C + (mb + sAi1 + m) * N + nb + sBi2) = t;
                }
            }
            else {
                int t = N - nb - sBi1;
                if (4 <= t) {
                    C[(mb + sAi1 + m) * N + nb + sBi1 + 0] = rC[m][0];
                    C[(mb + sAi1 + m) * N + nb + sBi1 + 1] = rC[m][1];
                    C[(mb + sAi1 + m) * N + nb + sBi1 + 2] = rC[m][2];
                    C[(mb + sAi1 + m) * N + nb + sBi1 + 3] = rC[m][3];
                }
                else if (3 <= t) {
                    C[(mb + sAi1 + m) * N + nb + sBi1 + 0] = rC[m][0];
                    C[(mb + sAi1 + m) * N + nb + sBi1 + 1] = rC[m][1];
                    C[(mb + sAi1 + m) * N + nb + sBi1 + 2] = rC[m][2];
                }
                else if (2 <= t) {
                    C[(mb + sAi1 + m) * N + nb + sBi1 + 0] = rC[m][0];
                    C[(mb + sAi1 + m) * N + nb + sBi1 + 1] = rC[m][1];
                }
                else if (1 <= t) {
                    C[(mb + sAi1 + m) * N + nb + sBi1 + 0] = rC[m][0];
                }
                t = N - nb - sBi2;
                if (4 <= t) {
                    C[(mb + sAi1 + m) * N + nb + sBi2 + 0] = rC[m][4];
                    C[(mb + sAi1 + m) * N + nb + sBi2 + 1] = rC[m][5];
                    C[(mb + sAi1 + m) * N + nb + sBi2 + 2] = rC[m][6];
                    C[(mb + sAi1 + m) * N + nb + sBi2 + 3] = rC[m][7];
                }
                else if (3 <= t) {
                    C[(mb + sAi1 + m) * N + nb + sBi2 + 0] = rC[m][4];
                    C[(mb + sAi1 + m) * N + nb + sBi2 + 1] = rC[m][5];
                    C[(mb + sAi1 + m) * N + nb + sBi2 + 2] = rC[m][6];
                }
                else if (2 <= t) {
                    C[(mb + sAi1 + m) * N + nb + sBi2 + 0] = rC[m][4];
                    C[(mb + sAi1 + m) * N + nb + sBi2 + 1] = rC[m][5];
                }
                else if (1 <= t) {
                    C[(mb + sAi1 + m) * N + nb + sBi2 + 0] = rC[m][4];
                }
            }
        }
    }
}

__global__ void sgemm_64x64x16_TN(float* A, float* B, float* C, int M, int N, int K, int aligned) {

    __shared__ __align__(16) float sA[16][64];
    __shared__ __align__(16) float sB[16][64];
    __shared__ __align__(16) float sAb[16][64];
    __shared__ __align__(16) float sBb[16][64];

    int tx = threadIdx.x;
    int mb = blockIdx.z * 128 + (blockIdx.x / 2) * 64;
    int nb = blockIdx.y * 128 + (blockIdx.x % 2) * 64;

    if (mb >= M || nb >= N) {
        return;
    }

    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm64 = tx % 64;
    int txb64 = tx / 64;
    int txm8 = tx % 8;
    int txm32b8 = txm32 / 8;

    int wmb = (txb32 / 2) * 32;
    int wnb = (txb32 % 2) * 32;

    int kb, k, m, n;

    float4 rA[2];
    float4 rB[2];
    float rC[8][4];
    float rAs[8];
    float rBs[4];
    float rAsb[8];
    float rBsb[4];

    float(*psA)[16][64] = &sA;
    float(*psB)[16][64] = &sB;

#pragma unroll
    for (m = 0; m < 8; m++) {
#pragma unroll
        for (n = 0; n < 4; n++) {
            rC[m][n] = 0;
        }
    }

#pragma unroll
    for (m = 0; m < 2; m++) {
        if (aligned) {
            if (mb + txm16 * 4 < M && txb16 + m * 8 < K) {
                rA[m] = *reinterpret_cast<float4*>(A + (txb16 + m * 8) * M + mb + txm16 * 4);
            }
            else {
                rA[m].x = rA[m].y = rA[m].z = rA[m].w = 0;
            }
            if (nb + txm16 * 4 < N && txb16 + m * 8 < K) {
                rB[m] = *reinterpret_cast<float4*>(B + (txb16 + m * 8) * N + nb + txm16 * 4);
            }
            else {
                rB[m].x = rB[m].y = rB[m].z = rB[m].w = 0;
            }
        }
        else {
            if (txb16 + m * 8 < K) {
                int t = M - mb - txm16 * 4;
                if (4 <= t) {
                    rA[m].x = A[(txb16 + m * 8) * M + (M - t) + 0];
                    rA[m].y = A[(txb16 + m * 8) * M + (M - t) + 1];
                    rA[m].z = A[(txb16 + m * 8) * M + (M - t) + 2];
                    rA[m].w = A[(txb16 + m * 8) * M + (M - t) + 3];
                }
                else if (3 <= t) {
                    rA[m].x = A[(txb16 + m * 8) * M + (M - t) + 0];
                    rA[m].y = A[(txb16 + m * 8) * M + (M - t) + 1];
                    rA[m].z = A[(txb16 + m * 8) * M + (M - t) + 2];
                    rA[m].w = 0;
                }
                else if (2 <= t) {
                    rA[m].x = A[(txb16 + m * 8) * M + (M - t) + 0];
                    rA[m].y = A[(txb16 + m * 8) * M + (M - t) + 1];
                    rA[m].z = rA[m].w = 0;
                }
                else if (1 <= t) {
                    rA[m].x = A[(txb16 + m * 8) * M + (M - t) + 0];
                    rA[m].y = rA[m].z = rA[m].w = 0;
                }
                else {
                    rA[m].x = rA[m].y = rA[m].z = rA[m].w = 0;
                }
                t = N - nb - txm16 * 4;
                if (4 <= t) {
                    rB[m].x = B[(txb16 + m * 8) * N + (N - t) + 0];
                    rB[m].y = B[(txb16 + m * 8) * N + (N - t) + 1];
                    rB[m].z = B[(txb16 + m * 8) * N + (N - t) + 2];
                    rB[m].w = B[(txb16 + m * 8) * N + (N - t) + 3];
                }
                else if (3 <= t) {
                    rB[m].x = B[(txb16 + m * 8) * N + (N - t) + 0];
                    rB[m].y = B[(txb16 + m * 8) * N + (N - t) + 1];
                    rB[m].z = B[(txb16 + m * 8) * N + (N - t) + 2];
                    rB[m].w = 0;
                }
                else if (2 <= t) {
                    rB[m].x = B[(txb16 + m * 8) * N + (N - t) + 0];
                    rB[m].y = B[(txb16 + m * 8) * N + (N - t) + 1];
                    rB[m].z = rB[m].w = 0;
                }
                else if (1 <= t) {
                    rB[m].x = B[(txb16 + m * 8) * N + (N - t) + 0];
                    rB[m].y = rB[m].z = rB[m].w = 0;
                }
                else {
                    rB[m].x = rB[m].y = rB[m].z = rB[m].w = 0;
                }
            }
            else {
                rA[m].x = rA[m].y = rA[m].z = rA[m].w = 0;
                rB[m].x = rB[m].y = rB[m].z = rB[m].w = 0;
            }
        }
    }

    for (kb = 0; kb < K; kb += 16) {

        (*psA)[txb16][txm16 * 4 + 0] = rA[0].x;
        (*psA)[txb16][txm16 * 4 + 1] = rA[0].y;
        (*psA)[txb16][txm16 * 4 + 2] = rA[0].z;
        (*psA)[txb16][txm16 * 4 + 3] = rA[0].w;
        (*psA)[8 + txb16][txm16 * 4 + 0] = rA[1].x;
        (*psA)[8 + txb16][txm16 * 4 + 1] = rA[1].y;
        (*psA)[8 + txb16][txm16 * 4 + 2] = rA[1].z;
        (*psA)[8 + txb16][txm16 * 4 + 3] = rA[1].w;

        (*psB)[txb16][txm16 * 4 + 0] = rB[0].x;
        (*psB)[txb16][txm16 * 4 + 1] = rB[0].y;
        (*psB)[txb16][txm16 * 4 + 2] = rB[0].z;
        (*psB)[txb16][txm16 * 4 + 3] = rB[0].w;
        (*psB)[8 + txb16][txm16 * 4 + 0] = rB[1].x;
        (*psB)[8 + txb16][txm16 * 4 + 1] = rB[1].y;
        (*psB)[8 + txb16][txm16 * 4 + 2] = rB[1].z;
        (*psB)[8 + txb16][txm16 * 4 + 3] = rB[1].w;

        __syncthreads();

        if (kb + 16 < K) {
#pragma unroll
            for (m = 0; m < 2; m++) {
                if (aligned) {
                    if (mb + txm16 * 4 < M && kb + 16 + txb16 + m * 8 < K) {
                        rA[m] = *reinterpret_cast<float4*>(A + (kb + 16 + txb16 + m * 8) * M + mb + txm16 * 4);
                    }
                    else {
                        rA[m].x = rA[m].y = rA[m].z = rA[m].w = 0;
                    }
                    if (nb + txm16 * 4 < N && kb + 16 + txb16 + m * 8 < K) {
                        rB[m] = *reinterpret_cast<float4*>(B + (kb + 16 + txb16 + m * 8) * N + nb + txm16 * 4);
                    }
                    else {
                        rB[m].x = rB[m].y = rB[m].z = rB[m].w = 0;
                    }
                }
                else {
                    if (kb + 16 + txb16 + m * 8 < K) {
                        int t = M - mb - txm16 * 4;
                        if (4 <= t) {
                            rA[m].x = A[(kb + 16 + txb16 + m * 8) * M + (M - t) + 0];
                            rA[m].y = A[(kb + 16 + txb16 + m * 8) * M + (M - t) + 1];
                            rA[m].z = A[(kb + 16 + txb16 + m * 8) * M + (M - t) + 2];
                            rA[m].w = A[(kb + 16 + txb16 + m * 8) * M + (M - t) + 3];
                        }
                        else if (3 <= t) {
                            rA[m].x = A[(kb + 16 + txb16 + m * 8) * M + (M - t) + 0];
                            rA[m].y = A[(kb + 16 + txb16 + m * 8) * M + (M - t) + 1];
                            rA[m].z = A[(kb + 16 + txb16 + m * 8) * M + (M - t) + 2];
                            rA[m].w = 0;
                        }
                        else if (2 <= t) {
                            rA[m].x = A[(kb + 16 + txb16 + m * 8) * M + (M - t) + 0];
                            rA[m].y = A[(kb + 16 + txb16 + m * 8) * M + (M - t) + 1];
                            rA[m].z = rA[m].w = 0;
                        }
                        else if (1 <= t) {
                            rA[m].x = A[(kb + 16 + txb16 + m * 8) * M + (M - t) + 0];
                            rA[m].y = rA[m].z = rA[m].w = 0;
                        }
                        else {
                            rA[m].x = rA[m].y = rA[m].z = rA[m].w = 0;
                        }
                        t = N - nb - txm16 * 4;
                        if (4 <= t) {
                            rB[m].x = B[(kb + 16 + txb16 + m * 8) * N + (N - t) + 0];
                            rB[m].y = B[(kb + 16 + txb16 + m * 8) * N + (N - t) + 1];
                            rB[m].z = B[(kb + 16 + txb16 + m * 8) * N + (N - t) + 2];
                            rB[m].w = B[(kb + 16 + txb16 + m * 8) * N + (N - t) + 3];
                        }
                        else if (3 <= t) {
                            rB[m].x = B[(kb + 16 + txb16 + m * 8) * N + (N - t) + 0];
                            rB[m].y = B[(kb + 16 + txb16 + m * 8) * N + (N - t) + 1];
                            rB[m].z = B[(kb + 16 + txb16 + m * 8) * N + (N - t) + 2];
                            rB[m].w = 0;
                        }
                        else if (2 <= t) {
                            rB[m].x = B[(kb + 16 + txb16 + m * 8) * N + (N - t) + 0];
                            rB[m].y = B[(kb + 16 + txb16 + m * 8) * N + (N - t) + 1];
                            rB[m].z = rB[m].w = 0;
                        }
                        else if (1 <= t) {
                            rB[m].x = B[(kb + 16 + txb16 + m * 8) * N + (N - t) + 0];
                            rB[m].y = rB[m].z = rB[m].w = 0;
                        }
                        else {
                            rB[m].x = rB[m].y = rB[m].z = rB[m].w = 0;
                        }
                    }
                    else {
                        rA[m].x = rA[m].y = rA[m].z = rA[m].w = 0;
                        rB[m].x = rB[m].y = rB[m].z = rB[m].w = 0;
                    }
                }
            }
        }

        for (m = 0; m < 8; m += 4) {
            float4 t = *reinterpret_cast<float4*>((*psA)[0] + wmb + m * 4 + txm32b8 * 4);
            rAs[m + 0] = t.x;
            rAs[m + 1] = t.y;
            rAs[m + 2] = t.z;
            rAs[m + 3] = t.w;
        }
        float4 t = *reinterpret_cast<float4*>((*psB)[0] + wnb + txm8 * 4);
        rBs[0] = t.x;
        rBs[1] = t.y;
        rBs[2] = t.z;
        rBs[3] = t.w;

#pragma unroll
        for (k = 0; k < 16; k++) {
            if (k < 15) {
#pragma unroll
                for (m = 0; m < 8; m += 4) {
                    float4 t = *reinterpret_cast<float4*>((*psA)[k + 1] + wmb + m * 4 + txm32b8 * 4);
                    rAsb[m + 0] = t.x;
                    rAsb[m + 1] = t.y;
                    rAsb[m + 2] = t.z;
                    rAsb[m + 3] = t.w;
                }
                float4 t = *reinterpret_cast<float4*>((*psB)[k + 1] + wnb + txm8 * 4);
                rBsb[0] = t.x;
                rBsb[1] = t.y;
                rBsb[2] = t.z;
                rBsb[3] = t.w;
            }

#pragma unroll
            for (m = 0; m < 8; m++) {
#pragma unroll
                for (n = 0; n < 4; n++) {
                    rC[m][n] += rAs[m] * rBs[n];
                }
            }

#pragma unroll
            for (m = 0; m < 8; m++) {
                rAs[m] = rAsb[m];
            }
#pragma unroll
            for (n = 0; n < 4; n++) {
                rBs[n] = rBsb[n];
            }

        }

        psA = (psA == &sA) ? &sAb : &sA;
        psB = (psB == &sB) ? &sBb : &sB;

    }


#pragma unroll
    for (m = 0; m < 8; m += 4) {
#pragma unroll
        for (n = 0; n < 4; n++) {
            int p1 = mb + wmb + m * 4 + txm32b8 * 4 + n;
            if (p1 < M) {
                int p2 = nb + wnb + txm8 * 4;
                if (aligned) {
                    if (p2 < N) {
                        float4 t;
                        t.x = rC[m + n][0];
                        t.y = rC[m + n][1];
                        t.z = rC[m + n][2];
                        t.w = rC[m + n][3];
                        *reinterpret_cast<float4*>(C + p1 * N + p2) = t;
                    }
                }
                else {
                    int t = N - p2;
                    if (4 <= t) {
                        C[p1 * N + p2 + 0] = rC[m + n][0];
                        C[p1 * N + p2 + 1] = rC[m + n][1];
                        C[p1 * N + p2 + 2] = rC[m + n][2];
                        C[p1 * N + p2 + 3] = rC[m + n][3];
                    }
                    else if (3 <= t) {
                        C[p1 * N + p2 + 0] = rC[m + n][0];
                        C[p1 * N + p2 + 1] = rC[m + n][1];
                        C[p1 * N + p2 + 2] = rC[m + n][2];
                    }
                    else if (2 <= t) {
                        C[p1 * N + p2 + 0] = rC[m + n][0];
                        C[p1 * N + p2 + 1] = rC[m + n][1];
                    }
                    else if (1 <= t) {
                        C[p1 * N + p2 + 0] = rC[m + n][0];
                    }
                }
            }
        }
    }


}

__global__ void sgemm_32x32x16_TN(float* A, float* B, float* C, int M, int N, int K, int aligned) {

    __shared__ __align__(16) float sA[16][32];
    __shared__ __align__(16) float sB[16][32];
    __shared__ __align__(16) float sAb[16][32];
    __shared__ __align__(16) float sBb[16][32];

    int tx = threadIdx.x;
    int mb = blockIdx.z * 64 + (blockIdx.x / 2) * 32;
    int nb = blockIdx.y * 64 + (blockIdx.x % 2) * 32;

    if (mb >= M || nb >= N) {
        return;
    }

    int txm8 = tx % 8;
    int txb8 = tx / 8;
    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm32m4 = txm32 % 4;
    int txm32b4 = txm32 / 4;
    int txm16b2 = txm16 / 2;
    int txm32m2 = txm32 % 2;
    int txm32b16 = txm32 / 16;
    int wmb = 0;
    int wnb = txb32 * 16;

    float rC[4][4];
    float4 rA[2];
    float4 rB[2];
    float rAs[4];
    float rBs[4];
    float rAsb[4];
    float rBsb[4];

    float(*psA)[16][32] = &sA;
    float(*psB)[16][32] = &sB;

    int m, n, k, kb;

#pragma unroll
    for (m = 0; m < 4; m++) {
#pragma unroll
        for (n = 0; n < 4; n++) {
            rC[m][n] = 0;
        }
    }

#pragma unroll
    for (m = 0; m < 2; m++) {
        if (aligned) {
            if (m * 8 + txb8 < K && mb + txm8 * 4 < M) {
                rA[m] = *reinterpret_cast<float4*>(A + (m * 8 + txb8) * M + mb + txm8 * 4);
            }
            else {
                rA[m].x = rA[m].y = rA[m].z = rA[m].w = 0;
            }
            if (m * 8 + txb8 < K && nb + txm8 * 4 < N) {
                rB[m] = *reinterpret_cast<float4*>(B + (m * 8 + txb8) * N + nb + txm8 * 4);
            }
            else {
                rB[m].x = rB[m].y = rB[m].z = rB[m].w = 0;
            }
        }
        else {
            if (m * 8 + txb8 < K) {
                int t = M - mb - txm8 * 4;
                if (4 <= t) {
                    rA[m].x = A[(m * 8 + txb8) * M + M - t + 0];
                    rA[m].y = A[(m * 8 + txb8) * M + M - t + 1];
                    rA[m].z = A[(m * 8 + txb8) * M + M - t + 2];
                    rA[m].w = A[(m * 8 + txb8) * M + M - t + 3];
                }
                else if (3 <= t) {
                    rA[m].x = A[(m * 8 + txb8) * M + M - t + 0];
                    rA[m].y = A[(m * 8 + txb8) * M + M - t + 1];
                    rA[m].z = A[(m * 8 + txb8) * M + M - t + 2];
                    rA[m].w = 0;
                }
                else if (2 <= t) {
                    rA[m].x = A[(m * 8 + txb8) * M + M - t + 0];
                    rA[m].y = A[(m * 8 + txb8) * M + M - t + 1];
                    rA[m].z = rA[m].w = 0;
                }
                else if (1 <= t) {
                    rA[m].x = A[(m * 8 + txb8) * M + M - t + 0];
                    rA[m].y = rA[m].z = rA[m].w = 0;
                }
                else {
                    rA[m].x = rA[m].y = rA[m].z = rA[m].w = 0;
                }
                t = N - nb - txm8 * 4;
                if (4 <= t) {
                    rB[m].x = B[(m * 8 + txb8) * N + N - t + 0];
                    rB[m].y = B[(m * 8 + txb8) * N + N - t + 1];
                    rB[m].z = B[(m * 8 + txb8) * N + N - t + 2];
                    rB[m].w = B[(m * 8 + txb8) * N + N - t + 3];
                }
                else if (3 <= t) {
                    rB[m].x = B[(m * 8 + txb8) * N + N - t + 0];
                    rB[m].y = B[(m * 8 + txb8) * N + N - t + 1];
                    rB[m].z = B[(m * 8 + txb8) * N + N - t + 2];
                    rB[m].w = 0;
                }
                else if (2 <= t) {
                    rB[m].x = B[(m * 8 + txb8) * N + N - t + 0];
                    rB[m].y = B[(m * 8 + txb8) * N + N - t + 1];
                    rB[m].z = rB[m].w = 0;
                }
                else if (1 <= t) {
                    rB[m].x = B[(m * 8 + txb8) * N + N - t + 0];
                    rB[m].y = rB[m].z = rB[m].w = 0;
                }
                else {
                    rB[m].x = rB[m].y = rB[m].z = rB[m].w = 0;
                }
            }
            else {
                rA[m].x = rA[m].y = rA[m].z = rA[m].w = 0;
                rB[m].x = rB[m].y = rB[m].z = rB[m].w = 0;
            }
        }
    }

    for (kb = 0; kb < K; kb += 16) {

#pragma unroll
        for (m = 0; m < 2; m++) {
            (*psA)[m * 8 + txb8][txm8 * 4 + 0] = rA[m].x;
            (*psA)[m * 8 + txb8][txm8 * 4 + 1] = rA[m].y;
            (*psA)[m * 8 + txb8][txm8 * 4 + 2] = rA[m].z;
            (*psA)[m * 8 + txb8][txm8 * 4 + 3] = rA[m].w;

            (*psB)[m * 8 + txb8][txm8 * 4 + 0] = rB[m].x;
            (*psB)[m * 8 + txb8][txm8 * 4 + 1] = rB[m].y;
            (*psB)[m * 8 + txb8][txm8 * 4 + 2] = rB[m].z;
            (*psB)[m * 8 + txb8][txm8 * 4 + 3] = rB[m].w;
        }

        __syncthreads();

        if (kb + 16 < K) {
#pragma unroll
            for (m = 0; m < 2; m++) {
                if (aligned) {
                    if (kb + 16 + m * 8 + txb8 < K && mb + txm8 * 4 < M) {
                        rA[m] = *reinterpret_cast<float4*>(A + (kb + 16 + m * 8 + txb8) * M + mb + txm8 * 4);
                    }
                    else {
                        rA[m].x = rA[m].y = rA[m].z = rA[m].w = 0;
                    }
                    if (kb + 16 + m * 8 + txb8 < K && nb + txm8 * 4 < N) {
                        rB[m] = *reinterpret_cast<float4*>(B + (kb + 16 + m * 8 + txb8) * N + nb + txm8 * 4);
                    }
                    else {
                        rB[m].x = rB[m].y = rB[m].z = rB[m].w = 0;
                    }
                }
                else {
                    if (kb + 16 + m * 8 + txb8 < K) {
                        int t = M - mb - txm8 * 4;
                        if (4 <= t) {
                            rA[m].x = A[(kb + 16 + m * 8 + txb8) * M + M - t + 0];
                            rA[m].y = A[(kb + 16 + m * 8 + txb8) * M + M - t + 1];
                            rA[m].z = A[(kb + 16 + m * 8 + txb8) * M + M - t + 2];
                            rA[m].w = A[(kb + 16 + m * 8 + txb8) * M + M - t + 3];
                        }
                        else if (3 <= t) {
                            rA[m].x = A[(kb + 16 + m * 8 + txb8) * M + M - t + 0];
                            rA[m].y = A[(kb + 16 + m * 8 + txb8) * M + M - t + 1];
                            rA[m].z = A[(kb + 16 + m * 8 + txb8) * M + M - t + 2];
                            rA[m].w = 0;
                        }
                        else if (2 <= t) {
                            rA[m].x = A[(kb + 16 + m * 8 + txb8) * M + M - t + 0];
                            rA[m].y = A[(kb + 16 + m * 8 + txb8) * M + M - t + 1];
                            rA[m].z = rA[m].w = 0;
                        }
                        else if (1 <= t) {
                            rA[m].x = A[(kb + 16 + m * 8 + txb8) * M + M - t + 0];
                            rA[m].y = rA[m].z = rA[m].w = 0;
                        }
                        else {
                            rA[m].x = rA[m].y = rA[m].z = rA[m].w = 0;
                        }
                        t = N - nb - txm8 * 4;
                        if (4 <= t) {
                            rB[m].x = B[(kb + 16 + m * 8 + txb8) * N + N - t + 0];
                            rB[m].y = B[(kb + 16 + m * 8 + txb8) * N + N - t + 1];
                            rB[m].z = B[(kb + 16 + m * 8 + txb8) * N + N - t + 2];
                            rB[m].w = B[(kb + 16 + m * 8 + txb8) * N + N - t + 3];
                        }
                        else if (3 <= t) {
                            rB[m].x = B[(kb + 16 + m * 8 + txb8) * N + N - t + 0];
                            rB[m].y = B[(kb + 16 + m * 8 + txb8) * N + N - t + 1];
                            rB[m].z = B[(kb + 16 + m * 8 + txb8) * N + N - t + 2];
                            rB[m].w = 0;
                        }
                        else if (2 <= t) {
                            rB[m].x = B[(kb + 16 + m * 8 + txb8) * N + N - t + 0];
                            rB[m].y = B[(kb + 16 + m * 8 + txb8) * N + N - t + 1];
                            rB[m].z = rB[m].w = 0;
                        }
                        else if (1 <= t) {
                            rB[m].x = B[(kb + 16 + m * 8 + txb8) * N + N - t + 0];
                            rB[m].y = rB[m].z = rB[m].w = 0;
                        }
                        else {
                            rB[m].x = rB[m].y = rB[m].z = rB[m].w = 0;
                        }
                    }
                    else {
                        rA[m].x = rA[m].y = rA[m].z = rA[m].w = 0;
                        rB[m].x = rB[m].y = rB[m].z = rB[m].w = 0;
                    }
                }
            }
        }

        // COMPUTE -------------------------

        float4 tA = *reinterpret_cast<float4*>((*psA)[0] + wmb + txm16b2 * 4);
        rAs[0] = tA.x;
        rAs[1] = tA.y;
        rAs[2] = tA.z;
        rAs[3] = tA.w;
        float4 tB = *reinterpret_cast<float4*>((*psB)[0] + wnb + txm32b16 * 8 + txm32m2 * 4);
        rBs[0] = tB.x;
        rBs[1] = tB.y;
        rBs[2] = tB.z;
        rBs[3] = tB.w;

#pragma unroll
        for (k = 0; k < 16; k++) {

            if (k < 15) {
                float4 tA = *reinterpret_cast<float4*>((*psA)[k + 1] + wmb + txm16b2 * 4);
                rAsb[0] = tA.x;
                rAsb[1] = tA.y;
                rAsb[2] = tA.z;
                rAsb[3] = tA.w;
                float4 tB = *reinterpret_cast<float4*>((*psB)[k + 1] + wnb + txm32b16 * 8 + txm32m2 * 4);
                rBsb[0] = tB.x;
                rBsb[1] = tB.y;
                rBsb[2] = tB.z;
                rBsb[3] = tB.w;
            }

#pragma unroll
            for (m = 0; m < 4; m++) {
#pragma unroll
                for (n = 0; n < 4; n++) {
                    rC[m][n] += rAs[m] * rBs[n];
                }
            }

#pragma unroll
            for (m = 0; m < 4; m++) {
                rAs[m] = rAsb[m];
                rBs[m] = rBsb[m];
            }
        }

        // ---------------------------------

        psA = (psA == &sA) ? &sAb : &sA;
        psB = (psB == &sB) ? &sBb : &sB;

    }

#pragma unroll
    for (m = 0; m < 4; m++) {
        int p1 = mb + wmb + txm16b2 * 4 + m;
        if (p1 < M) {
            int p2 = nb + wnb + txm32b16 * 8 + txm32m2 * 4;
            if (aligned) {
                if (p2 < N) {
                    float4 t;
                    t.x = rC[m][0];
                    t.y = rC[m][1];
                    t.z = rC[m][2];
                    t.w = rC[m][3];
                    *reinterpret_cast<float4*>(C + p1 * N + p2) = t;
                }
            }
            else {
                int t = N - p2;
                if (4 <= t) {
                    C[p1 * N + p2 + 0] = rC[m][0];
                    C[p1 * N + p2 + 1] = rC[m][1];
                    C[p1 * N + p2 + 2] = rC[m][2];
                    C[p1 * N + p2 + 3] = rC[m][3];
                }
                else if (3 <= t) {
                    C[p1 * N + p2 + 0] = rC[m][0];
                    C[p1 * N + p2 + 1] = rC[m][1];
                    C[p1 * N + p2 + 2] = rC[m][2];
                }
                else if (2 <= t) {
                    C[p1 * N + p2 + 0] = rC[m][0];
                    C[p1 * N + p2 + 1] = rC[m][1];
                }
                else if (1 <= t) {
                    C[p1 * N + p2 + 0] = rC[m][0];
                }
            }
        }
    }

}

__global__ void sgemm_4x4x256_TN(float* A, float* B, float* C, int M, int N, int K, int aligned) {

    __shared__ float sA[4][256];
    __shared__ float sAb[4][256];
    __shared__ float sC[8][4][4];

    int tx = threadIdx.x;
    int mb = blockIdx.x * 4;
    int nb = blockIdx.y * 4;

    int txm8 = tx % 8;
    int txb8 = tx / 8;
    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm64 = tx % 64;
    int txb64 = tx / 64;
    int txm32m4 = txm32 % 4;
    int txm32b4 = txm32 / 4;
    int txm16b2 = txm16 / 2;
    int txm32m2 = txm32 % 2;
    int txm32b16 = txm32 / 16;
    int wmb = 0;
    int wnb = txb32 * 16;

    float rC[4][4];
    float4 rA, rAb;
    float4 rB, rBb;

    float(*psA)[4][256] = &sA;

    int m, n, k, kb;

#pragma unroll
    for (m = 0; m < 4; m++) {
#pragma unroll
        for (n = 0; n < 4; n++) {
            rC[m][n] = 0;
        }
    }

    if (tx < K) {
        if (aligned) {
            rA = *reinterpret_cast<float4*>(A + tx * M + mb);
            rB = *reinterpret_cast<float4*>(B + tx * N + nb);
        }
        else {
            int t = M - mb;
            if (4 <= t) {
                rA.x = A[tx * M + mb + 0];
                rA.y = A[tx * M + mb + 1];
                rA.z = A[tx * M + mb + 2];
                rA.w = A[tx * M + mb + 3];
            }
            else if (3 <= t) {
                rA.x = A[tx * M + mb + 0];
                rA.y = A[tx * M + mb + 1];
                rA.z = A[tx * M + mb + 2];
                rA.w = 0;
            }
            else if (2 <= t) {
                rA.x = A[tx * M + mb + 0];
                rA.y = A[tx * M + mb + 1];
                rA.z = rA.w = 0;
            }
            else if (1 <= t) {
                rA.x = A[tx * M + mb + 0];
                rA.y = rA.z = rA.w = 0;
            }
            else {
                rA.x = rA.y = rA.z = rA.w = 0;
            }
            t = N - nb;
            if (4 <= t) {
                rB.x = B[tx * N + nb + 0];
                rB.y = B[tx * N + nb + 1];
                rB.z = B[tx * N + nb + 2];
                rB.w = B[tx * N + nb + 3];
            }
            else if (3 <= t) {
                rB.x = B[tx * N + nb + 0];
                rB.y = B[tx * N + nb + 1];
                rB.z = B[tx * N + nb + 2];
                rB.w = 0;
            }
            else if (2 <= t) {
                rB.x = B[tx * N + nb + 0];
                rB.y = B[tx * N + nb + 1];
                rB.z = rB.w = 0;
            }
            else if (1 <= t) {
                rB.x = B[tx * N + nb + 0];
                rB.y = rB.z = rB.w = 0;
            }
            else {
                rB.x = rB.y = rB.z = rB.w = 0;
            }
        }
    }
    else {
        rA.x = rA.y = rA.z = rA.w = 0;
        rB.x = rB.y = rB.z = rB.w = 0;
    }

    for (kb = 0; kb < K; kb += 256) {

        (*psA)[0][tx] = rA.x;
        (*psA)[1][tx] = rA.y;
        (*psA)[2][tx] = rA.z;
        (*psA)[3][tx] = rA.w;

        __syncthreads();

        if (kb + 256 < K) {
            if (kb + 256 + tx < K) {
                if (aligned) {
                    rA = *reinterpret_cast<float4*>(A + (kb + 256 + tx) * M + mb);
                    rBb = *reinterpret_cast<float4*>(B + (kb + 256 + tx) * N + nb);
                }
                else {
                    int t = M - mb;
                    if (4 <= t) {
                        rA.x = A[(kb + 256 + tx) * M + mb + 0];
                        rA.y = A[(kb + 256 + tx) * M + mb + 1];
                        rA.z = A[(kb + 256 + tx) * M + mb + 2];
                        rA.w = A[(kb + 256 + tx) * M + mb + 3];
                    }
                    else if (3 <= t) {
                        rA.x = A[(kb + 256 + tx) * M + mb + 0];
                        rA.y = A[(kb + 256 + tx) * M + mb + 1];
                        rA.z = A[(kb + 256 + tx) * M + mb + 2];
                        rA.w = 0;
                    }
                    else if (2 <= t) {
                        rA.x = A[(kb + 256 + tx) * M + mb + 0];
                        rA.y = A[(kb + 256 + tx) * M + mb + 1];
                        rA.z = rA.w = 0;
                    }
                    else if (1 <= t) {
                        rA.x = A[(kb + 256 + tx) * M + mb + 0];
                        rA.y = rA.z = rA.w = 0;
                    }
                    else {
                        rA.x = rA.y = rA.z = rA.w = 0;
                    }
                    t = N - nb;
                    if (4 <= t) {
                        rBb.x = B[(kb + 256 + tx) * N + nb + 0];
                        rBb.y = B[(kb + 256 + tx) * N + nb + 1];
                        rBb.z = B[(kb + 256 + tx) * N + nb + 2];
                        rBb.w = B[(kb + 256 + tx) * N + nb + 3];
                    }
                    else if (3 <= t) {
                        rBb.x = B[(kb + 256 + tx) * N + nb + 0];
                        rBb.y = B[(kb + 256 + tx) * N + nb + 1];
                        rBb.z = B[(kb + 256 + tx) * N + nb + 2];
                        rBb.w = 0;
                    }
                    else if (2 <= t) {
                        rBb.x = B[(kb + 256 + tx) * N + nb + 0];
                        rBb.y = B[(kb + 256 + tx) * N + nb + 1];
                        rBb.z = rBb.w = 0;
                    }
                    else if (1 <= t) {
                        rBb.x = B[(kb + 256 + tx) * N + nb + 0];
                        rBb.y = rBb.z = rBb.w = 0;
                    }
                    else {
                        rBb.x = rBb.y = rBb.z = rBb.w = 0;
                    }
                }
            }
            else {
                rA.x = rA.y = rA.z = rA.w = 0;
                rBb.x = rBb.y = rBb.z = rBb.w = 0;
            }
        }

        // COMPUTE -------------------------

#pragma unroll
        for (m = 0; m < 4; m++) {
            rC[m][0] += (*psA)[m][tx] * rB.x;
            rC[m][1] += (*psA)[m][tx] * rB.y;
            rC[m][2] += (*psA)[m][tx] * rB.z;
            rC[m][3] += (*psA)[m][tx] * rB.w;
        }

        rB.x = rBb.x;
        rB.y = rBb.y;
        rB.z = rBb.z;
        rB.w = rBb.w;

        psA = (psA == &sA) ? &sAb : &sA;

    }

#pragma unroll
    for (m = 0; m < 4; m++) {
#pragma unroll
        for (n = 0; n < 4; n++) {
            float t = __shfl_sync(FULL_MASK, rC[m][n], txm32 + 16, 32);
            rC[m][n] += t;
            t = __shfl_sync(FULL_MASK, rC[m][n], txm32 + 8, 32);
            rC[m][n] += t;
            t = __shfl_sync(FULL_MASK, rC[m][n], txm32 + 4, 32);
            rC[m][n] += t;
            t = __shfl_sync(FULL_MASK, rC[m][n], txm32 + 2, 32);
            rC[m][n] += t;
            t = __shfl_sync(FULL_MASK, rC[m][n], txm32 + 1, 32);
            rC[m][n] += t;
            if (txm32 == 0) {
                sC[txb32][m][n] = rC[m][n];
            }
        }
    }

    __syncthreads();

    if (tx == 0) {
        if (aligned) {
#pragma unroll
            for (m = 0; m < 4; m++) {
                if (mb + m < M) {
                    float4 t;
                    t.x = rC[m][0] + sC[1][m][0] + sC[2][m][0] + sC[3][m][0] + sC[4][m][0] + sC[5][m][0] + sC[6][m][0] + sC[7][m][0];
                    t.y = rC[m][1] + sC[1][m][1] + sC[2][m][1] + sC[3][m][1] + sC[4][m][1] + sC[5][m][1] + sC[6][m][1] + sC[7][m][1];
                    t.z = rC[m][2] + sC[1][m][2] + sC[2][m][2] + sC[3][m][2] + sC[4][m][2] + sC[5][m][2] + sC[6][m][2] + sC[7][m][2];
                    t.w = rC[m][3] + sC[1][m][3] + sC[2][m][3] + sC[3][m][3] + sC[4][m][3] + sC[5][m][3] + sC[6][m][3] + sC[7][m][3];
                    *reinterpret_cast<float4*>(C + (mb + m) * N + nb) = t;
                }
            }
        }
        else {
            int t = N - nb;
#pragma unroll
            for (m = 0; m < 4; m++) {
                if (mb + m < M) {
                    if (4 <= t) {
                        C[(mb + m) * N + nb + 0] = rC[m][0] + sC[1][m][0] + sC[2][m][0] + sC[3][m][0] + sC[4][m][0] + sC[5][m][0] + sC[6][m][0] + sC[7][m][0];
                        C[(mb + m) * N + nb + 1] = rC[m][1] + sC[1][m][1] + sC[2][m][1] + sC[3][m][1] + sC[4][m][1] + sC[5][m][1] + sC[6][m][1] + sC[7][m][1];
                        C[(mb + m) * N + nb + 2] = rC[m][2] + sC[1][m][2] + sC[2][m][2] + sC[3][m][2] + sC[4][m][2] + sC[5][m][2] + sC[6][m][2] + sC[7][m][2];
                        C[(mb + m) * N + nb + 3] = rC[m][3] + sC[1][m][3] + sC[2][m][3] + sC[3][m][3] + sC[4][m][3] + sC[5][m][3] + sC[6][m][3] + sC[7][m][3];
                    }
                    else if (3 <= t) {
                        C[(mb + m) * N + nb + 0] = rC[m][0] + sC[1][m][0] + sC[2][m][0] + sC[3][m][0] + sC[4][m][0] + sC[5][m][0] + sC[6][m][0] + sC[7][m][0];
                        C[(mb + m) * N + nb + 1] = rC[m][1] + sC[1][m][1] + sC[2][m][1] + sC[3][m][1] + sC[4][m][1] + sC[5][m][1] + sC[6][m][1] + sC[7][m][1];
                        C[(mb + m) * N + nb + 2] = rC[m][2] + sC[1][m][2] + sC[2][m][2] + sC[3][m][2] + sC[4][m][2] + sC[5][m][2] + sC[6][m][2] + sC[7][m][2];
                    }
                    else if (2 <= t) {
                        C[(mb + m) * N + nb + 0] = rC[m][0] + sC[1][m][0] + sC[2][m][0] + sC[3][m][0] + sC[4][m][0] + sC[5][m][0] + sC[6][m][0] + sC[7][m][0];
                        C[(mb + m) * N + nb + 1] = rC[m][1] + sC[1][m][1] + sC[2][m][1] + sC[3][m][1] + sC[4][m][1] + sC[5][m][1] + sC[6][m][1] + sC[7][m][1];
                    }
                    else if (1 <= t) {
                        C[(mb + m) * N + nb + 0] = rC[m][0] + sC[1][m][0] + sC[2][m][0] + sC[3][m][0] + sC[4][m][0] + sC[5][m][0] + sC[6][m][0] + sC[7][m][0];
                    }
                }
            }
        }
    }

}



__global__ void sgemm_strided_128x128x8_NN(float* A, float* B, float* C, int M, int N, int K, int S) {

    __shared__ __align__(16) float sA[8][132];
    __shared__ __align__(16) float sB[8][128];
    __shared__ __align__(16) float sAb[8][132];
    __shared__ __align__(16) float sBb[8][128];

    int tx = threadIdx.x;
    int mb = blockIdx.z * 256 + (blockIdx.x / 2) * 128;
    int nb = blockIdx.y * 256 + (blockIdx.x % 2) * 128;

    if (mb >= M || nb >= N) {
        return;
    }

    float rC[8][8];
    float4 rA;
    float4 rB;
    float rAs[8];
    float rBs[8];
    float rAsb[8];
    float rBsb[8];

    float(*psA)[8][132] = &sA;
    float(*psB)[8][128] = &sB;

    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm2 = tx % 2;
    int txb2 = tx / 2;
    int txm8 = tx % 8;
    int txb8 = tx / 8;
    int txm128 = tx % 128;
    int txb128 = tx / 128;
    int txm32b4 = txm32 / 4;

    int SK = S * K;
    int SN = S * N;

    int wnb = (txb32 / 4) * 64;
    int wmb = (txb32 % 4) * 32;

    int m, n, k, kb, kbb, s;

    float4 f4;

    int sAi1 = wmb + (txm16 / 4) * 4;
    int sAi2 = 16 + sAi1;
    int sBi1 = wnb + (16 * (txm32 / 16)) + (txm32 % 4) * 4;
    int sBi2 = 32 + sBi1;

    for (s = 0; s < S; s++) {

#pragma unroll
        for (m = 0; m < 8; m++) {
#pragma unroll
            for (n = 0; n < 8; n++) {
                rC[m][n] = 0;
            }
        }

        rA = *reinterpret_cast<float4*>(A + (mb + txb2) * SK + s * K + txm2 * 4);
        rB = *reinterpret_cast<float4*>(B + txb32 * SN + s * N + nb + txm32 * 4);

        for (kb = 0; kb < K; kb += 8) {

            (*psA)[txm2 * 4 + 0][txb2] = rA.x;
            (*psA)[txm2 * 4 + 1][txb2] = rA.y;
            (*psA)[txm2 * 4 + 2][txb2] = rA.z;
            (*psA)[txm2 * 4 + 3][txb2] = rA.w;
            (*psB)[txb32][txm32 * 4 + 0] = rB.x;
            (*psB)[txb32][txm32 * 4 + 1] = rB.y;
            (*psB)[txb32][txm32 * 4 + 2] = rB.z;
            (*psB)[txb32][txm32 * 4 + 3] = rB.w;

            __syncthreads();

            if (kb + 8 < K) {
                rA = *reinterpret_cast<float4*>(A + (mb + txb2) * SK + s * K + kb + 8 + txm2 * 4);
                rB = *reinterpret_cast<float4*>(B + (kb + 8 + txb32) * SN + s * N + nb +  txm32 * 4);
            }

            // COMPUTE -------------------

            f4 = *reinterpret_cast<float4*>((*psA)[0] + sAi1);
            rAs[0] = f4.x;
            rAs[1] = f4.y;
            rAs[2] = f4.z;
            rAs[3] = f4.w;
            f4 = *reinterpret_cast<float4*>((*psA)[0] + sAi2);
            rAs[4] = f4.x;
            rAs[5] = f4.y;
            rAs[6] = f4.z;
            rAs[7] = f4.w;
            f4 = *reinterpret_cast<float4*>((*psB)[0] + sBi1);
            rBs[0] = f4.x;
            rBs[1] = f4.y;
            rBs[2] = f4.z;
            rBs[3] = f4.w;
            f4 = *reinterpret_cast<float4*>((*psB)[0] + sBi2);
            rBs[4] = f4.x;
            rBs[5] = f4.y;
            rBs[6] = f4.z;
            rBs[7] = f4.w;

#pragma unroll
            for (k = 0; k < 8; k++) {

                if (k < 7) {
                    f4 = *reinterpret_cast<float4*>((*psA)[k + 1] + sAi1);
                    rAsb[0] = f4.x;
                    rAsb[1] = f4.y;
                    rAsb[2] = f4.z;
                    rAsb[3] = f4.w;
                    f4 = *reinterpret_cast<float4*>((*psA)[k + 1] + sAi2);
                    rAsb[4] = f4.x;
                    rAsb[5] = f4.y;
                    rAsb[6] = f4.z;
                    rAsb[7] = f4.w;
                    f4 = *reinterpret_cast<float4*>((*psB)[k + 1] + sBi1);
                    rBsb[0] = f4.x;
                    rBsb[1] = f4.y;
                    rBsb[2] = f4.z;
                    rBsb[3] = f4.w;
                    f4 = *reinterpret_cast<float4*>((*psB)[k + 1] + sBi2);
                    rBsb[4] = f4.x;
                    rBsb[5] = f4.y;
                    rBsb[6] = f4.z;
                    rBsb[7] = f4.w;
                }

#pragma unroll
                for (m = 0; m < 8; m++) {
#pragma unroll
                    for (n = 0; n < 8; n++) {
                        rC[m][n] += rAs[m] * rBs[n];
                    }
                }

#pragma unroll
                for (m = 0; m < 8; m++) {
                    rAs[m] = rAsb[m];
                    rBs[m] = rBsb[m];
                }

            }

            // ---------------------------

            if (psA == &sA) {
                psA = &sAb;
                psB = &sBb;
            }
            else {
                psA = &sA;
                psB = &sB;
            }

        }

#pragma unroll
        for (m = 0; m < 4; m++) {
            f4.x = rC[m][0];
            f4.y = rC[m][1];
            f4.z = rC[m][2];
            f4.w = rC[m][3];
            *reinterpret_cast<float4*>(C + (mb + sAi1 + m) * SN + s * N + nb + sBi1) = f4;
            f4.x = rC[4 + m][0];
            f4.y = rC[4 + m][1];
            f4.z = rC[4 + m][2];
            f4.w = rC[4 + m][3];
            *reinterpret_cast<float4*>(C + (mb + sAi2 + m) * SN + s * N + nb + sBi1) = f4;
            f4.x = rC[m][4];
            f4.y = rC[m][5];
            f4.z = rC[m][6];
            f4.w = rC[m][7];
            *reinterpret_cast<float4*>(C + (mb + sAi1 + m) * SN + s * N + nb + sBi2) = f4;
            f4.x = rC[4 + m][4];
            f4.y = rC[4 + m][5];
            f4.z = rC[4 + m][6];
            f4.w = rC[4 + m][7];
            *reinterpret_cast<float4*>(C + (mb + sAi2 + m) * SN + s * N + nb + sBi2) = f4;
        }
    }

}

__global__ void sgemm_strided_128x16x8_NN(float* A, float* B, float* C, int M, int N, int K, int S) {

    __shared__ __align__(16) float sA[8][132];
    __shared__ __align__(16) float sB[8][16];
    __shared__ __align__(16) float sAb[8][132];
    __shared__ __align__(16) float sBb[8][16];

    int tx = threadIdx.x;
    int mb = blockIdx.z * 256 + (blockIdx.x / 2) * 128;
    int nb = blockIdx.y * 32 + (blockIdx.x % 2) * 16;

    if (mb >= M || nb >= N) {
        return;
    }

    float rC[4][4];
    float4 rA[2];
    float rB;
    float rAs[4];
    float rBs[4];
    float rAsb[4];
    float rBsb[4];

    float(*psA)[8][132] = &sA;
    float(*psB)[8][16] = &sB;

    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm2 = tx % 2;
    int txb2 = tx / 2;
    int txm8 = tx % 8;
    int txb8 = tx / 8;
    int txm128 = tx % 128;
    int txb128 = tx / 128;
    int txm32b4 = txm32 / 4;

    int SK = S * K;
    int SN = S * N;

    int wnb = (txb32 / 2) * 8;
    int wmb = (txb32 % 2) * 64;

    int m, n, k, kb, kbb, s;

    float4 t;

    int sAi1 = wmb + (txm32 / 2) * 4;
    int sBi1 = wnb + (txm32 % 2) * 4;

    for (s = 0; s < S; s++) {

#pragma unroll
        for (m = 0; m < 4; m++) {
#pragma unroll
            for (n = 0; n < 4; n++) {
                rC[m][n] = 0;
            }
        }

        rA[0] = *reinterpret_cast<float4*>(A + (mb + txb2) * SK + s * K + txm2 * 4);
        rA[1] = *reinterpret_cast<float4*>(A + (mb + 64 + txb2) * SK + s * K + txm2 * 4);
        rB = B[txb16 * SN + s * N + nb + txm16];

        for (kb = 0; kb < K; kb += 8) {

            (*psA)[txm2 * 4 + 0][txb2] = rA[0].x;
            (*psA)[txm2 * 4 + 1][txb2] = rA[0].y;
            (*psA)[txm2 * 4 + 2][txb2] = rA[0].z;
            (*psA)[txm2 * 4 + 3][txb2] = rA[0].w;
            (*psA)[txm2 * 4 + 0][64 + txb2] = rA[1].x;
            (*psA)[txm2 * 4 + 1][64 + txb2] = rA[1].y;
            (*psA)[txm2 * 4 + 2][64 + txb2] = rA[1].z;
            (*psA)[txm2 * 4 + 3][64 + txb2] = rA[1].w;

            (*psB)[txb16][txm16] = rB;

            __syncthreads();

            if (kb + 8 < K) {
                rA[0] = *reinterpret_cast<float4*>(A + (mb + txb2) * SK + s * K + kb + 8 + txm2 * 4);
                rA[1] = *reinterpret_cast<float4*>(A + (mb + 64 + txb2) * SK + s * K + kb + 8 + txm2 * 4);
                rB = B[(kb + 8 + txb16) * SN + s * N + nb + txm16];
            }

            // COMPUTE -------------------

            float4 t = *reinterpret_cast<float4*>((*psA)[0] + sAi1);
            rAs[0] = t.x;
            rAs[1] = t.y;
            rAs[2] = t.z;
            rAs[3] = t.w;
            t = *reinterpret_cast<float4*>((*psB)[0] + sBi1);
            rBs[0] = t.x;
            rBs[1] = t.y;
            rBs[2] = t.z;
            rBs[3] = t.w;

#pragma unroll
            for (k = 0; k < 8; k++) {

                if (k < 7) {
                    t = *reinterpret_cast<float4*>((*psA)[k + 1] + sAi1);
                    rAsb[0] = t.x;
                    rAsb[1] = t.y;
                    rAsb[2] = t.z;
                    rAsb[3] = t.w;
                    t = *reinterpret_cast<float4*>((*psB)[k + 1] + sBi1);
                    rBsb[0] = t.x;
                    rBsb[1] = t.y;
                    rBsb[2] = t.z;
                    rBsb[3] = t.w;
                }

#pragma unroll
                for (m = 0; m < 4; m++) {
#pragma unroll
                    for (n = 0; n < 4; n++) {
                        rC[m][n] += rAs[m] * rBs[n];
                    }
                }

#pragma unroll
                for (m = 0; m < 4; m++) {
                    rAs[m] = rAsb[m];
                    rBs[m] = rBsb[m];
                }

            }

            // ---------------------------

            if (psA == &sA) {
                psA = &sAb;
                psB = &sBb;
            }
            else {
                psA = &sA;
                psB = &sB;
            }

        }

#pragma unroll
        for (m = 0; m < 4; m++) {
            t.x = rC[m][0];
            t.y = rC[m][1];
            t.z = rC[m][2];
            t.w = rC[m][3];
            *reinterpret_cast<float4*>(C + (mb + sAi1 + m) * SN + s * N + nb + sBi1) = t;
        }
    }

}

__global__ void sgemm_strided_64x64x16_NN(float* A, float* B, float* C, int M, int N, int K, int S) {

    __shared__ __align__(16) float sA[16][64];
    __shared__ __align__(16) float sB[16][64];
    __shared__ __align__(16) float sAb[16][64];
    __shared__ __align__(16) float sBb[16][64];

    int tx = threadIdx.x;
    int mb = blockIdx.z * 128 + (blockIdx.x / 2) * 64;
    int nb = blockIdx.y * 128 + (blockIdx.x % 2) * 64;

    if (mb >= M || nb >= N) {
        return;
    }

    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm64 = tx % 64;
    int txb64 = tx / 64;
    int txm8 = tx % 8;
    int txm32b8 = txm32 / 8;

    int SK = S * K;
    int SN = S * N;

    int wmb = (txb32 / 2) * 32;
    int wnb = (txb32 % 2) * 32;

    int kb, k, m, n, s;

    float4 rA[2];
    float4 rB[2];
    float rC[8][4];
    float rAs[8];
    float rBs[4];
    float rAsb[8];
    float rBsb[4];

    float(*psA)[16][64] = &sA;
    float(*psB)[16][64] = &sB;

    for (s = 0; s < S; s++) {
#pragma unroll
        for (m = 0; m < 8; m++) {
#pragma unroll
            for (n = 0; n < 4; n++) {
                rC[m][n] = 0;
            }
        }

        rA[0] = *reinterpret_cast<float4*>(A + (mb + txm32) * SK + s * K + txb32 * 4);
        rA[1] = *reinterpret_cast<float4*>(A + (mb + 32 + txm32) * SK + s * K + txb32 * 4);
        rB[0] = *reinterpret_cast<float4*>(B + txb16 * SN + s * N + nb + txm16 * 4);
        rB[1] = *reinterpret_cast<float4*>(B + (8 + txb16) * SN + s * N + nb + txm16 * 4);

        for (kb = 0; kb < K; kb += 16) {

            (*psA)[txb32 * 4 + 0][txm32] = rA[0].x;
            (*psA)[txb32 * 4 + 1][txm32] = rA[0].y;
            (*psA)[txb32 * 4 + 2][txm32] = rA[0].z;
            (*psA)[txb32 * 4 + 3][txm32] = rA[0].w;
            (*psA)[txb32 * 4 + 0][32 + txm32] = rA[1].x;
            (*psA)[txb32 * 4 + 1][32 + txm32] = rA[1].y;
            (*psA)[txb32 * 4 + 2][32 + txm32] = rA[1].z;
            (*psA)[txb32 * 4 + 3][32 + txm32] = rA[1].w;
            (*psB)[txb16][txm16 * 4 + 0] = rB[0].x;
            (*psB)[txb16][txm16 * 4 + 1] = rB[0].y;
            (*psB)[txb16][txm16 * 4 + 2] = rB[0].z;
            (*psB)[txb16][txm16 * 4 + 3] = rB[0].w;
            (*psB)[8 + txb16][txm16 * 4 + 0] = rB[1].x;
            (*psB)[8 + txb16][txm16 * 4 + 1] = rB[1].y;
            (*psB)[8 + txb16][txm16 * 4 + 2] = rB[1].z;
            (*psB)[8 + txb16][txm16 * 4 + 3] = rB[1].w;

            __syncthreads();

            if (kb + 16 < K) {
                rA[0] = *reinterpret_cast<float4*>(A + (mb + txm32) * SK + s * K + kb + 16 + txb32 * 4);
                rA[1] = *reinterpret_cast<float4*>(A + (mb + 32 + txm32) * SK + s * K + kb + 16 + txb32 * 4);
                rB[0] = *reinterpret_cast<float4*>(B + (kb + 16 + txb16) * SN + s * N + nb + txm16 * 4);
                rB[1] = *reinterpret_cast<float4*>(B + (kb + 24 + txb16) * SN + s * N + nb + txm16 * 4);
            }

            for (m = 0; m < 8; m += 4) {
                float4 t = *reinterpret_cast<float4*>((*psA)[0] + wmb + m * 4 + txm32b8 * 4);
                rAs[m + 0] = t.x;
                rAs[m + 1] = t.y;
                rAs[m + 2] = t.z;
                rAs[m + 3] = t.w;
            }
            float4 t = *reinterpret_cast<float4*>((*psB)[0] + wnb + txm8 * 4);
            rBs[0] = t.x;
            rBs[1] = t.y;
            rBs[2] = t.z;
            rBs[3] = t.w;

#pragma unroll
            for (k = 0; k < 16; k++) {
                if (k < 15) {
#pragma unroll
                    for (m = 0; m < 8; m += 4) {
                        float4 t = *reinterpret_cast<float4*>((*psA)[k + 1] + wmb + m * 4 + txm32b8 * 4);
                        rAsb[m + 0] = t.x;
                        rAsb[m + 1] = t.y;
                        rAsb[m + 2] = t.z;
                        rAsb[m + 3] = t.w;
                    }
                    float4 t = *reinterpret_cast<float4*>((*psB)[k + 1] + wnb + txm8 * 4);
                    rBsb[0] = t.x;
                    rBsb[1] = t.y;
                    rBsb[2] = t.z;
                    rBsb[3] = t.w;
                }

#pragma unroll
                for (m = 0; m < 8; m++) {
#pragma unroll
                    for (n = 0; n < 4; n++) {
                        rC[m][n] += rAs[m] * rBs[n];
                    }
                }

#pragma unroll
                for (m = 0; m < 8; m++) {
                    rAs[m] = rAsb[m];
                }
#pragma unroll
                for (n = 0; n < 4; n++) {
                    rBs[n] = rBsb[n];
                }

            }

            psA = (psA == &sA) ? &sAb : &sA;
            psB = (psB == &sB) ? &sBb : &sB;

        }

#pragma unroll
        for (m = 0; m < 8; m += 4) {
#pragma unroll
            for (n = 0; n < 4; n++) {
                float4 t;
                t.x = rC[m + n][0];
                t.y = rC[m + n][1];
                t.z = rC[m + n][2];
                t.w = rC[m + n][3];
                *reinterpret_cast<float4*>(C + (mb + wmb + m * 4 + txm32b8 * 4 + n) * SN + s * N + nb + wnb + txm8 * 4) = t;
            }
        }
    }


}

__global__ void sgemm_strided_32x32x16_NN(float* A, float* B, float* C, int M, int N, int K, int S) {

    __shared__ __align__(16) float sA[16][32];
    __shared__ __align__(16) float sB[16][32];
    __shared__ __align__(16) float sAb[16][32];
    __shared__ __align__(16) float sBb[16][32];

    int tx = threadIdx.x;
    int mb = blockIdx.z * 64 + (blockIdx.x / 2) * 32;
    int nb = blockIdx.y * 64 + (blockIdx.x % 2) * 32;

    if (mb >= M || nb >= N) {
        return;
    }

    int txm8 = tx % 8;
    int txb8 = tx / 8;
    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm32m4 = txm32 % 4;
    int txm32b4 = txm32 / 4;
    int txm16b2 = txm16 / 2;
    int txm32m2 = txm32 % 2;
    int txm32b16 = txm32 / 16;

    int SK = S * K;
    int SN = S * N;

    int wmb = 0;
    int wnb = txb32 * 16;

    float rC[4][4];
    float4 rA[2];
    float4 rB[2];
    float rAs[4];
    float rBs[4];
    float rAsb[4];
    float rBsb[4];

    float(*psA)[16][32] = &sA;
    float(*psB)[16][32] = &sB;

    int m, n, k, kb, s;

    for (s = 0; s < S; s++) {
#pragma unroll
        for (m = 0; m < 4; m++) {
#pragma unroll
            for (n = 0; n < 4; n++) {
                rC[m][n] = 0;
            }
        }

#pragma unroll
        for (m = 0; m < 2; m++) {
            rA[m] = *reinterpret_cast<float4*>(A + (mb + m * 16 + txm16) * SK + s * K + txb16 * 4);
            rB[m] = *reinterpret_cast<float4*>(B + (m * 8 + txb8) * SN + s * N + nb + txm8 * 4);
        }

        for (kb = 0; kb < K; kb += 16) {

#pragma unroll
            for (m = 0; m < 2; m++) {
                (*psA)[txb16 * 4 + 0][m * 16 + txm16] = rA[m].x;
                (*psA)[txb16 * 4 + 1][m * 16 + txm16] = rA[m].y;
                (*psA)[txb16 * 4 + 2][m * 16 + txm16] = rA[m].z;
                (*psA)[txb16 * 4 + 3][m * 16 + txm16] = rA[m].w;

                (*psB)[m * 8 + txb8][txm8 * 4 + 0] = rB[m].x;
                (*psB)[m * 8 + txb8][txm8 * 4 + 1] = rB[m].y;
                (*psB)[m * 8 + txb8][txm8 * 4 + 2] = rB[m].z;
                (*psB)[m * 8 + txb8][txm8 * 4 + 3] = rB[m].w;
            }

            __syncthreads();

            if (kb + 16 < K) {
#pragma unroll
                for (m = 0; m < 2; m++) {
                    rA[m] = *reinterpret_cast<float4*>(A + (mb + m * 16 + txm16) * SK + s * K + kb + 16 + txb16 * 4);
                    rB[m] = *reinterpret_cast<float4*>(B + (m * 8 + kb + 16 + txb8) * SN + s * N + nb + txm8 * 4);
                }
            }

            // COMPUTE -------------------------

            float4 tA = *reinterpret_cast<float4*>((*psA)[0] + wmb + txm16b2 * 4);
            rAs[0] = tA.x;
            rAs[1] = tA.y;
            rAs[2] = tA.z;
            rAs[3] = tA.w;
            float4 tB = *reinterpret_cast<float4*>((*psB)[0] + wnb + txm32b16 * 8 + txm32m2 * 4);
            rBs[0] = tB.x;
            rBs[1] = tB.y;
            rBs[2] = tB.z;
            rBs[3] = tB.w;

#pragma unroll
            for (k = 0; k < 16; k++) {

                if (k < 15) {
                    float4 tA = *reinterpret_cast<float4*>((*psA)[k + 1] + wmb + txm16b2 * 4);
                    rAsb[0] = tA.x;
                    rAsb[1] = tA.y;
                    rAsb[2] = tA.z;
                    rAsb[3] = tA.w;
                    float4 tB = *reinterpret_cast<float4*>((*psB)[k + 1] + wnb + txm32b16 * 8 + txm32m2 * 4);
                    rBsb[0] = tB.x;
                    rBsb[1] = tB.y;
                    rBsb[2] = tB.z;
                    rBsb[3] = tB.w;
                }

#pragma unroll
                for (m = 0; m < 4; m++) {
#pragma unroll
                    for (n = 0; n < 4; n++) {
                        rC[m][n] += rAs[m] * rBs[n];
                    }
                }

#pragma unroll
                for (m = 0; m < 4; m++) {
                    rAs[m] = rAsb[m];
                    rBs[m] = rBsb[m];
                }
            }

            // ---------------------------------

            psA = (psA == &sA) ? &sAb : &sA;
            psB = (psB == &sB) ? &sBb : &sB;

        }

#pragma unroll
        for (m = 0; m < 4; m++) {
#pragma unroll
            for (n = 0; n < 4; n += 4) {
                float4 t;
                t.x = rC[m][n + 0];
                t.y = rC[m][n + 1];
                t.z = rC[m][n + 2];
                t.w = rC[m][n + 3];
                *reinterpret_cast<float4*>(C + (mb + wmb + txm16b2 * 4 + m) * SN + s * N + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + n) = t;
            }
        }
    }

}

__global__ void sgemm_strided_16x16x16_NN(float* A, float* B, float* C, int M, int N, int K, int S) {

    __shared__ __align__(16) float sA[16][16];
    __shared__ __align__(16) float sB[16][16];
    __shared__ __align__(16) float sAb[16][16];
    __shared__ __align__(16) float sBb[16][16];

    int tx = threadIdx.x;
    int mb = blockIdx.z * 32 + (blockIdx.x / 2) * 16;
    int nb = blockIdx.y * 32 + (blockIdx.x % 2) * 16;

    if (mb >= M || nb >= N) {
        return;
    }

    int txm4 = tx % 4;
    int txb4 = tx / 4;
    int txm8 = tx % 8;
    int txb8 = tx / 8;
    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm32m4 = txm32 % 4;
    int txm32b4 = txm32 / 4;
    int txm16b2 = txm16 / 2;
    int txm32m2 = txm32 % 2;
    int txm32b16 = txm32 / 16;

    int SK = S * K;
    int SN = S * N;

    int wmb = 0;
    int wnb = txb32 * 8;

    float rC[2][2];
    float4 rA;
    float4 rB;
    float rAs[2];
    float rBs[2];
    float rAsb[2];
    float rBsb[2];

    float(*psA)[16][16] = &sA;
    float(*psB)[16][16] = &sB;

    int m, n, k, kb, s;

    for (s = 0; s < S; s++) {
#pragma unroll
        for (m = 0; m < 2; m++) {
#pragma unroll
            for (n = 0; n < 2; n++) {
                rC[m][n] = 0;
            }
        }

        rA = *reinterpret_cast<float4*>(A + (mb + txb4) * SK + s * K + txm4 * 4);
        rB = *reinterpret_cast<float4*>(B + txb4 * SN + s * N + nb + txm4 * 4);

        for (kb = 0; kb < K; kb += 16) {

            (*psA)[txm4 * 4 + 0][txb4] = rA.x;
            (*psA)[txm4 * 4 + 1][txb4] = rA.y;
            (*psA)[txm4 * 4 + 2][txb4] = rA.z;
            (*psA)[txm4 * 4 + 3][txb4] = rA.w;

            (*psB)[txb4][txm4 * 4 + 0] = rB.x;
            (*psB)[txb4][txm4 * 4 + 1] = rB.y;
            (*psB)[txb4][txm4 * 4 + 2] = rB.z;
            (*psB)[txb4][txm4 * 4 + 3] = rB.w;

            __syncthreads();

            if (kb + 16 < K) {
                rA = *reinterpret_cast<float4*>(A + (mb + txb4) * SK + s * K + kb + 16 + txm4 * 4);
                rB = *reinterpret_cast<float4*>(B + (kb + 16 + txb4) * SN + s * N + nb + txm4 * 4);
            }

            // COMPUTE -------------------------

            float2 tA = *reinterpret_cast<float2*>((*psA)[0] + wmb + txm32b4 * 2);
            rAs[0] = tA.x;
            rAs[1] = tA.y;
            float2 tB = *reinterpret_cast<float2*>((*psB)[0] + wnb + txm32m4 * 2);
            rBs[0] = tB.x;
            rBs[1] = tB.y;

#pragma unroll
            for (k = 0; k < 16; k++) {

                if (k < 15) {
                    float2 tA = *reinterpret_cast<float2*>((*psA)[k + 1] + wmb + txm32b4 * 2);
                    rAsb[0] = tA.x;
                    rAsb[1] = tA.y;
                    float2 tB = *reinterpret_cast<float2*>((*psB)[k + 1] + wnb + txm32m4 * 2);
                    rBsb[0] = tB.x;
                    rBsb[1] = tB.y;
                }

#pragma unroll
                for (m = 0; m < 2; m++) {
#pragma unroll
                    for (n = 0; n < 2; n++) {
                        rC[m][n] += rAs[m] * rBs[n];
                    }
                }

#pragma unroll
                for (m = 0; m < 2; m++) {
                    rAs[m] = rAsb[m];
                    rBs[m] = rBsb[m];
                }
            }

            // ---------------------------------

            psA = (psA == &sA) ? &sAb : &sA;
            psB = (psB == &sB) ? &sBb : &sB;

        }

#pragma unroll
        for (m = 0; m < 2; m++) {
#pragma unroll
            for (n = 0; n < 2; n += 2) {
                float2 t;
                t.x = rC[m][n + 0];
                t.y = rC[m][n + 1];
                *reinterpret_cast<float2*>(C + (mb + wmb + txm32b4 * 2 + m) * SN + s * N + nb + wnb + txm32m4 * 2 + n) = t;
            }
        }
    }

}



__global__ void sgemm_strided_128x128x8_NT(float* A, float* B, float* C, int M, int N, int K, int S) {

    __shared__ __align__(16) float sA[8][132];
    __shared__ __align__(16) float sB[8][132];
    __shared__ __align__(16) float sAb[8][132];
    __shared__ __align__(16) float sBb[8][132];

    int tx = threadIdx.x;
    int mb = blockIdx.z * 256 + (blockIdx.x / 2) * 128;
    int nb = blockIdx.y * 256 + (blockIdx.x % 2) * 128;

    if (mb >= M || nb >= N) {
        return;
    }

    float rC[8][8];
    float4 rA;
    float4 rB;
    float rAs[8];
    float rBs[8];
    float rAsb[8];
    float rBsb[8];

    float(*psA)[8][132] = &sA;
    float(*psB)[8][132] = &sB;

    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm2 = tx % 2;
    int txb2 = tx / 2;
    int txm8 = tx % 8;
    int txb8 = tx / 8;
    int txm128 = tx % 128;
    int txb128 = tx / 128;
    int txm32b4 = txm32 / 4;

    int SK = S * K;
    int SN = S * N;

    int wnb = (txb32 / 4) * 64;
    int wmb = (txb32 % 4) * 32;

    int m, n, k, kb, kbb, s;

    float4 f4;

    int sAi1 = wmb + (txm16 / 4) * 4;
    int sAi2 = 16 + sAi1;
    int sBi1 = wnb + (16 * (txm32 / 16)) + (txm32 % 4) * 4;
    int sBi2 = 32 + sBi1;

    for (s = 0; s < S; s++) {

#pragma unroll
        for (m = 0; m < 8; m++) {
#pragma unroll
            for (n = 0; n < 8; n++) {
                rC[m][n] = 0;
            }
        }

        rA = *reinterpret_cast<float4*>(A + (mb + txb2) * SK + s * K + txm2 * 4);
        rB = *reinterpret_cast<float4*>(B + (nb + txb2) * SK + s * K + txm2 * 4);

        for (kb = 0; kb < K; kb += 8) {

            (*psA)[txm2 * 4 + 0][txb2] = rA.x;
            (*psA)[txm2 * 4 + 1][txb2] = rA.y;
            (*psA)[txm2 * 4 + 2][txb2] = rA.z;
            (*psA)[txm2 * 4 + 3][txb2] = rA.w;
            (*psB)[txm2 * 4 + 0][txb2] = rB.x;
            (*psB)[txm2 * 4 + 1][txb2] = rB.y;
            (*psB)[txm2 * 4 + 2][txb2] = rB.z;
            (*psB)[txm2 * 4 + 3][txb2] = rB.w;

            __syncthreads();

            if (kb + 8 < K) {
                rA = *reinterpret_cast<float4*>(A + (mb + txb2) * SK + s * K + kb + 8 + txm2 * 4);
                rB = *reinterpret_cast<float4*>(B + (nb + txb2) * SK + s * K + kb + 8 + txm2 * 4);
            }

            // COMPUTE -------------------

            f4 = *reinterpret_cast<float4*>((*psA)[0] + sAi1);
            rAs[0] = f4.x;
            rAs[1] = f4.y;
            rAs[2] = f4.z;
            rAs[3] = f4.w;
            f4 = *reinterpret_cast<float4*>((*psA)[0] + sAi2);
            rAs[4] = f4.x;
            rAs[5] = f4.y;
            rAs[6] = f4.z;
            rAs[7] = f4.w;
            f4 = *reinterpret_cast<float4*>((*psB)[0] + sBi1);
            rBs[0] = f4.x;
            rBs[1] = f4.y;
            rBs[2] = f4.z;
            rBs[3] = f4.w;
            f4 = *reinterpret_cast<float4*>((*psB)[0] + sBi2);
            rBs[4] = f4.x;
            rBs[5] = f4.y;
            rBs[6] = f4.z;
            rBs[7] = f4.w;

#pragma unroll
            for (k = 0; k < 8; k++) {

                if (k < 7) {
                    f4 = *reinterpret_cast<float4*>((*psA)[k + 1] + sAi1);
                    rAsb[0] = f4.x;
                    rAsb[1] = f4.y;
                    rAsb[2] = f4.z;
                    rAsb[3] = f4.w;
                    f4 = *reinterpret_cast<float4*>((*psA)[k + 1] + sAi2);
                    rAsb[4] = f4.x;
                    rAsb[5] = f4.y;
                    rAsb[6] = f4.z;
                    rAsb[7] = f4.w;
                    f4 = *reinterpret_cast<float4*>((*psB)[k + 1] + sBi1);
                    rBsb[0] = f4.x;
                    rBsb[1] = f4.y;
                    rBsb[2] = f4.z;
                    rBsb[3] = f4.w;
                    f4 = *reinterpret_cast<float4*>((*psB)[k + 1] + sBi2);
                    rBsb[4] = f4.x;
                    rBsb[5] = f4.y;
                    rBsb[6] = f4.z;
                    rBsb[7] = f4.w;
                }

#pragma unroll
                for (m = 0; m < 8; m++) {
#pragma unroll
                    for (n = 0; n < 8; n++) {
                        rC[m][n] += rAs[m] * rBs[n];
                    }
                }

#pragma unroll
                for (m = 0; m < 8; m++) {
                    rAs[m] = rAsb[m];
                    rBs[m] = rBsb[m];
                }

            }

            // ---------------------------

            if (psA == &sA) {
                psA = &sAb;
                psB = &sBb;
            }
            else {
                psA = &sA;
                psB = &sB;
            }

        }

#pragma unroll
        for (m = 0; m < 4; m++) {
            f4.x = rC[m][0];
            f4.y = rC[m][1];
            f4.z = rC[m][2];
            f4.w = rC[m][3];
            *reinterpret_cast<float4*>(C + (mb + sAi1 + m) * SN + s * N + nb + sBi1) = f4;
            f4.x = rC[4 + m][0];
            f4.y = rC[4 + m][1];
            f4.z = rC[4 + m][2];
            f4.w = rC[4 + m][3];
            *reinterpret_cast<float4*>(C + (mb + sAi2 + m) * SN + s * N + nb + sBi1) = f4;
            f4.x = rC[m][4];
            f4.y = rC[m][5];
            f4.z = rC[m][6];
            f4.w = rC[m][7];
            *reinterpret_cast<float4*>(C + (mb + sAi1 + m) * SN + s * N + nb + sBi2) = f4;
            f4.x = rC[4 + m][4];
            f4.y = rC[4 + m][5];
            f4.z = rC[4 + m][6];
            f4.w = rC[4 + m][7];
            *reinterpret_cast<float4*>(C + (mb + sAi2 + m) * SN + s * N + nb + sBi2) = f4;
        }
    }

}

__global__ void sgemm_strided_64x64x16_NT(float* A, float* B, float* C, int M, int N, int K, int S) {

    __shared__ __align__(16) float sA[16][64];
    __shared__ __align__(16) float sB[16][64];
    __shared__ __align__(16) float sAb[16][64];
    __shared__ __align__(16) float sBb[16][64];

    int tx = threadIdx.x;
    int mb = blockIdx.z * 128 + (blockIdx.x / 2) * 64;
    int nb = blockIdx.y * 128 + (blockIdx.x % 2) * 64;

    if (mb >= M || nb >= N) {
        return;
    }

    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm64 = tx % 64;
    int txb64 = tx / 64;
    int txm8 = tx % 8;
    int txm32b8 = txm32 / 8;

    int SK = S * K;
    int SN = S * N;

    int wmb = (txb32 / 2) * 32;
    int wnb = (txb32 % 2) * 32;

    int kb, k, m, n, s;

    float4 rA[2];
    float4 rB[2];
    float rC[8][4];
    float rAs[8];
    float rBs[4];
    float rAsb[8];
    float rBsb[4];

    float(*psA)[16][64] = &sA;
    float(*psB)[16][64] = &sB;

    for (s = 0; s < S; s++) {
#pragma unroll
        for (m = 0; m < 8; m++) {
#pragma unroll
            for (n = 0; n < 4; n++) {
                rC[m][n] = 0;
            }
        }

        rA[0] = *reinterpret_cast<float4*>(A + (mb + txm32) * SK + s * K + txb32 * 4);
        rA[1] = *reinterpret_cast<float4*>(A + (mb + 32 + txm32) * SK + s * K + txb32 * 4);
        rB[0] = *reinterpret_cast<float4*>(B + (nb + txm32) * SK + s * K + txb32 * 4);
        rB[1] = *reinterpret_cast<float4*>(B + (nb + 32 + txm32) * SK + s * K + txb32 * 4);

        for (kb = 0; kb < K; kb += 16) {

            (*psA)[txb32 * 4 + 0][txm32] = rA[0].x;
            (*psA)[txb32 * 4 + 1][txm32] = rA[0].y;
            (*psA)[txb32 * 4 + 2][txm32] = rA[0].z;
            (*psA)[txb32 * 4 + 3][txm32] = rA[0].w;
            (*psA)[txb32 * 4 + 0][32 + txm32] = rA[1].x;
            (*psA)[txb32 * 4 + 1][32 + txm32] = rA[1].y;
            (*psA)[txb32 * 4 + 2][32 + txm32] = rA[1].z;
            (*psA)[txb32 * 4 + 3][32 + txm32] = rA[1].w;
            (*psB)[txb32 * 4 + 0][txm32] = rB[0].x;
            (*psB)[txb32 * 4 + 1][txm32] = rB[0].y;
            (*psB)[txb32 * 4 + 2][txm32] = rB[0].z;
            (*psB)[txb32 * 4 + 3][txm32] = rB[0].w;
            (*psB)[txb32 * 4 + 0][32 + txm32] = rB[1].x;
            (*psB)[txb32 * 4 + 1][32 + txm32] = rB[1].y;
            (*psB)[txb32 * 4 + 2][32 + txm32] = rB[1].z;
            (*psB)[txb32 * 4 + 3][32 + txm32] = rB[1].w;

            __syncthreads();

            if (kb + 16 < K) {
                rA[0] = *reinterpret_cast<float4*>(A + (mb + txm32) * SK + s * K + kb + 16 + txb32 * 4);
                rA[1] = *reinterpret_cast<float4*>(A + (mb + 32 + txm32) * SK + s * K + kb + 16 + txb32 * 4);
                rB[0] = *reinterpret_cast<float4*>(B + (nb + txm32) * SK + s * K + kb + 16 + txb32 * 4);
                rB[1] = *reinterpret_cast<float4*>(B + (nb + 32 + txm32) * SK + s * K + kb + 16 + txb32 * 4);
            }

            for (m = 0; m < 8; m += 4) {
                float4 t = *reinterpret_cast<float4*>((*psA)[0] + wmb + m * 4 + txm32b8 * 4);
                rAs[m + 0] = t.x;
                rAs[m + 1] = t.y;
                rAs[m + 2] = t.z;
                rAs[m + 3] = t.w;
            }
            float4 t = *reinterpret_cast<float4*>((*psB)[0] + wnb + txm8 * 4);
            rBs[0] = t.x;
            rBs[1] = t.y;
            rBs[2] = t.z;
            rBs[3] = t.w;

#pragma unroll
            for (k = 0; k < 16; k++) {
                if (k < 15) {
#pragma unroll
                    for (m = 0; m < 8; m += 4) {
                        float4 t = *reinterpret_cast<float4*>((*psA)[k + 1] + wmb + m * 4 + txm32b8 * 4);
                        rAsb[m + 0] = t.x;
                        rAsb[m + 1] = t.y;
                        rAsb[m + 2] = t.z;
                        rAsb[m + 3] = t.w;
                    }
                    float4 t = *reinterpret_cast<float4*>((*psB)[k + 1] + wnb + txm8 * 4);
                    rBsb[0] = t.x;
                    rBsb[1] = t.y;
                    rBsb[2] = t.z;
                    rBsb[3] = t.w;
                }

#pragma unroll
                for (m = 0; m < 8; m++) {
#pragma unroll
                    for (n = 0; n < 4; n++) {
                        rC[m][n] += rAs[m] * rBs[n];
                    }
                }

#pragma unroll
                for (m = 0; m < 8; m++) {
                    rAs[m] = rAsb[m];
                }
#pragma unroll
                for (n = 0; n < 4; n++) {
                    rBs[n] = rBsb[n];
                }

            }

            psA = (psA == &sA) ? &sAb : &sA;
            psB = (psB == &sB) ? &sBb : &sB;

        }

#pragma unroll
        for (m = 0; m < 8; m += 4) {
#pragma unroll
            for (n = 0; n < 4; n++) {
                float4 t;
                t.x = rC[m + n][0];
                t.y = rC[m + n][1];
                t.z = rC[m + n][2];
                t.w = rC[m + n][3];
                *reinterpret_cast<float4*>(C + (mb + wmb + m * 4 + txm32b8 * 4 + n) * SN + s * N + nb + wnb + txm8 * 4) = t;
            }
        }
    }


}

__global__ void sgemm_strided_32x32x16_NT(float* A, float* B, float* C, int M, int N, int K, int S) {

    __shared__ __align__(16) float sA[16][32];
    __shared__ __align__(16) float sB[16][32];
    __shared__ __align__(16) float sAb[16][32];
    __shared__ __align__(16) float sBb[16][32];

    int tx = threadIdx.x;
    int mb = blockIdx.z * 64 + (blockIdx.x / 2) * 32;
    int nb = blockIdx.y * 64 + (blockIdx.x % 2) * 32;

    if (mb >= M || nb >= N) {
        return;
    }

    int txm8 = tx % 8;
    int txb8 = tx / 8;
    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm32m4 = txm32 % 4;
    int txm32b4 = txm32 / 4;
    int txm16b2 = txm16 / 2;
    int txm32m2 = txm32 % 2;
    int txm32b16 = txm32 / 16;

    int SK = S * K;
    int SN = S * N;

    int wmb = 0;
    int wnb = txb32 * 16;

    float rC[4][4];
    float4 rA[2];
    float4 rB[2];
    float rAs[4];
    float rBs[4];
    float rAsb[4];
    float rBsb[4];

    float(*psA)[16][32] = &sA;
    float(*psB)[16][32] = &sB;

    int m, n, k, kb, s;

    for (s = 0; s < S; s++) {
#pragma unroll
        for (m = 0; m < 4; m++) {
#pragma unroll
            for (n = 0; n < 4; n++) {
                rC[m][n] = 0;
            }
        }

#pragma unroll
        for (m = 0; m < 2; m++) {
            rA[m] = *reinterpret_cast<float4*>(A + (mb + m * 16 + txm16) * SK + s * K + txb16 * 4);
            rB[m] = *reinterpret_cast<float4*>(B + (nb + m * 16 + txm16) * SK + s * K + txb16 * 4);
        }

        for (kb = 0; kb < K; kb += 16) {

#pragma unroll
            for (m = 0; m < 2; m++) {
                (*psA)[txb16 * 4 + 0][m * 16 + txm16] = rA[m].x;
                (*psA)[txb16 * 4 + 1][m * 16 + txm16] = rA[m].y;
                (*psA)[txb16 * 4 + 2][m * 16 + txm16] = rA[m].z;
                (*psA)[txb16 * 4 + 3][m * 16 + txm16] = rA[m].w;

                (*psB)[txb16 * 4 + 0][m * 16 + txm16] = rB[m].x;
                (*psB)[txb16 * 4 + 1][m * 16 + txm16] = rB[m].y;
                (*psB)[txb16 * 4 + 2][m * 16 + txm16] = rB[m].z;
                (*psB)[txb16 * 4 + 3][m * 16 + txm16] = rB[m].w;
            }

            __syncthreads();

            if (kb + 16 < K) {
#pragma unroll
                for (m = 0; m < 2; m++) {
                    rA[m] = *reinterpret_cast<float4*>(A + (mb + m * 16 + txm16) * SK + s * K + kb + 16 + txb16 * 4);
                    rB[m] = *reinterpret_cast<float4*>(B + (nb + m * 16 + txm16) * SK + s * K + kb + 16 + txb16 * 4);
                }
            }

            // COMPUTE -------------------------

            float4 tA = *reinterpret_cast<float4*>((*psA)[0] + wmb + txm16b2 * 4);
            rAs[0] = tA.x;
            rAs[1] = tA.y;
            rAs[2] = tA.z;
            rAs[3] = tA.w;
            float4 tB = *reinterpret_cast<float4*>((*psB)[0] + wnb + txm32b16 * 8 + txm32m2 * 4);
            rBs[0] = tB.x;
            rBs[1] = tB.y;
            rBs[2] = tB.z;
            rBs[3] = tB.w;

#pragma unroll
            for (k = 0; k < 16; k++) {

                if (k < 15) {
                    float4 tA = *reinterpret_cast<float4*>((*psA)[k + 1] + wmb + txm16b2 * 4);
                    rAsb[0] = tA.x;
                    rAsb[1] = tA.y;
                    rAsb[2] = tA.z;
                    rAsb[3] = tA.w;
                    float4 tB = *reinterpret_cast<float4*>((*psB)[k + 1] + wnb + txm32b16 * 8 + txm32m2 * 4);
                    rBsb[0] = tB.x;
                    rBsb[1] = tB.y;
                    rBsb[2] = tB.z;
                    rBsb[3] = tB.w;
                }

#pragma unroll
                for (m = 0; m < 4; m++) {
#pragma unroll
                    for (n = 0; n < 4; n++) {
                        rC[m][n] += rAs[m] * rBs[n];
                    }
                }

#pragma unroll
                for (m = 0; m < 4; m++) {
                    rAs[m] = rAsb[m];
                    rBs[m] = rBsb[m];
                }
            }

            // ---------------------------------

            psA = (psA == &sA) ? &sAb : &sA;
            psB = (psB == &sB) ? &sBb : &sB;

        }

#pragma unroll
        for (m = 0; m < 4; m++) {
#pragma unroll
            for (n = 0; n < 4; n += 4) {
                float4 t;
                t.x = rC[m][n + 0];
                t.y = rC[m][n + 1];
                t.z = rC[m][n + 2];
                t.w = rC[m][n + 3];
                *reinterpret_cast<float4*>(C + (mb + wmb + txm16b2 * 4 + m) * SN + s * N + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + n) = t;
            }
        }
    }

}


__global__ void sgemm_strided_128x128x8_TN(float* A, float* B, float* C, int M, int N, int K, int S) {

    __shared__ __align__(16) float sA[8][128];
    __shared__ __align__(16) float sB[8][128];
    __shared__ __align__(16) float sAb[8][128];
    __shared__ __align__(16) float sBb[8][128];

    int tx = threadIdx.x;
    int mb = blockIdx.z * 256 + (blockIdx.x / 2) * 128;
    int nb = blockIdx.y * 256 + (blockIdx.x % 2) * 128;

    if (mb >= M || nb >= N) {
        return;
    }

    float rC[8][8];
    float4 rA;
    float4 rB;
    float rAs[8];
    float rBs[8];
    float rAsb[8];
    float rBsb[8];

    float(*psA)[8][128] = &sA;
    float(*psB)[8][128] = &sB;

    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm2 = tx % 2;
    int txb2 = tx / 2;
    int txm8 = tx % 8;
    int txb8 = tx / 8;
    int txm128 = tx % 128;
    int txb128 = tx / 128;
    int txm32b4 = txm32 / 4;

    int SM = S * M;
    int SN = S * N;

    int wnb = (txb32 / 4) * 64;
    int wmb = (txb32 % 4) * 32;

    int m, n, k, kb, kbb, s;

    float4 f4;
    float8 f8;

    int sAi1 = wmb + (txm32 % 2) * 8 + (txm32 / 16) * 16;
    int sBi1 = wnb + (txm32 / 2) * 4;
    int sBi2 = wnb + (32 + (txm32 / 2) * 4) % 64;

    for (s = 0; s < S; s++) {

#pragma unroll
        for (m = 0; m < 8; m++) {
#pragma unroll
            for (n = 0; n < 8; n++) {
                rC[m][n] = 0;
            }
        }

        rA = *reinterpret_cast<float4*>(A + txb32 * SM + s * M + mb + txm32 * 4);
        rB = *reinterpret_cast<float4*>(B + txb32 * SN + s * N + nb + txm32 * 4);

        for (kb = 0; kb < K; kb += 8) {

            (*psA)[txb32][txm32 * 4 + 0] = rA.x;
            (*psA)[txb32][txm32 * 4 + 1] = rA.y;
            (*psA)[txb32][txm32 * 4 + 2] = rA.z;
            (*psA)[txb32][txm32 * 4 + 3] = rA.w;
            (*psB)[txb32][txm32 * 4 + 0] = rB.x;
            (*psB)[txb32][txm32 * 4 + 1] = rB.y;
            (*psB)[txb32][txm32 * 4 + 2] = rB.z;
            (*psB)[txb32][txm32 * 4 + 3] = rB.w;

            __syncthreads();

            if (kb + 8 < K) {
                rA = *reinterpret_cast<float4*>(A + (kb + 8 + txb32) * SM + s * M + mb + txm32 * 4);
                rB = *reinterpret_cast<float4*>(B + (kb + 8 + txb32) * SN + s * N + nb + txm32 * 4);
            }

            // COMPUTE -------------------

            f8 = *reinterpret_cast<float8*>((*psA)[0] + sAi1);
            rAs[0] = f8.a;
            rAs[1] = f8.b;
            rAs[2] = f8.c;
            rAs[3] = f8.d;
            rAs[4] = f8.e;
            rAs[5] = f8.f;
            rAs[6] = f8.g;
            rAs[7] = f8.h;
            f4 = *reinterpret_cast<float4*>((*psB)[0] + sBi1);
            rBs[0] = f4.x;
            rBs[1] = f4.y;
            rBs[2] = f4.z;
            rBs[3] = f4.w;
            f4 = *reinterpret_cast<float4*>((*psB)[0] + sBi2);
            rBs[4] = f4.x;
            rBs[5] = f4.y;
            rBs[6] = f4.z;
            rBs[7] = f4.w;

#pragma unroll
            for (k = 0; k < 8; k++) {

                if (k < 7) {
                    f8 = *reinterpret_cast<float8*>((*psA)[k + 1] + sAi1);
                    rAsb[0] = f8.a;
                    rAsb[1] = f8.b;
                    rAsb[2] = f8.c;
                    rAsb[3] = f8.d;
                    rAsb[4] = f8.e;
                    rAsb[5] = f8.f;
                    rAsb[6] = f8.g;
                    rAsb[7] = f8.h;
                    f4 = *reinterpret_cast<float4*>((*psB)[k + 1] + sBi1);
                    rBsb[0] = f4.x;
                    rBsb[1] = f4.y;
                    rBsb[2] = f4.z;
                    rBsb[3] = f4.w;
                    f4 = *reinterpret_cast<float4*>((*psB)[k + 1] + sBi2);
                    rBsb[4] = f4.x;
                    rBsb[5] = f4.y;
                    rBsb[6] = f4.z;
                    rBsb[7] = f4.w;
                }

#pragma unroll
                for (m = 0; m < 8; m++) {
#pragma unroll
                    for (n = 0; n < 8; n++) {
                        rC[m][n] += rAs[m] * rBs[n];
                    }
                }

#pragma unroll
                for (m = 0; m < 8; m++) {
                    rAs[m] = rAsb[m];
                    rBs[m] = rBsb[m];
                }

            }

            // ---------------------------

            if (psA == &sA) {
                psA = &sAb;
                psB = &sBb;
            }
            else {
                psA = &sA;
                psB = &sB;
            }

        }

#pragma unroll
        for (m = 0; m < 8; m++) {
            float4 t, z;
            t.x = rC[m][0];
            t.y = rC[m][1];
            t.z = rC[m][2];
            t.w = rC[m][3];
            *reinterpret_cast<float4*>(C + (mb + sAi1 + m) * SN + s * N + nb + sBi1) = t;
            z.x = rC[m][4];
            z.y = rC[m][5];
            z.z = rC[m][6];
            z.w = rC[m][7];
            *reinterpret_cast<float4*>(C + (mb + sAi1 + m) * SN + s * N + nb + sBi2) = z;
        }
    }

}

__global__ void sgemm_strided_64x64x16_TN(float* A, float* B, float* C, int M, int N, int K, int S) {

    __shared__ __align__(16) float sA[16][64];
    __shared__ __align__(16) float sB[16][64];
    __shared__ __align__(16) float sAb[16][64];
    __shared__ __align__(16) float sBb[16][64];

    int tx = threadIdx.x;
    int mb = blockIdx.z * 128 + (blockIdx.x / 2) * 64;
    int nb = blockIdx.y * 128 + (blockIdx.x % 2) * 64;

    if (mb >= M || nb >= N) {
        return;
    }

    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm64 = tx % 64;
    int txb64 = tx / 64;
    int txm8 = tx % 8;
    int txm32b8 = txm32 / 8;

    int SM = S * M;
    int SN = S * N;

    int wmb = (txb32 / 2) * 32;
    int wnb = (txb32 % 2) * 32;

    int kb, k, m, n, s;

    float4 rA[2];
    float4 rB[2];
    float rC[8][4];
    float rAs[8];
    float rBs[4];
    float rAsb[8];
    float rBsb[4];

    float(*psA)[16][64] = &sA;
    float(*psB)[16][64] = &sB;

    for (s = 0; s < S; s++) {
#pragma unroll
        for (m = 0; m < 8; m++) {
#pragma unroll
            for (n = 0; n < 4; n++) {
                rC[m][n] = 0;
            }
        }

        rA[0] = *reinterpret_cast<float4*>(A + txb16 * SM + s * M + mb + txm16 * 4);
        rA[1] = *reinterpret_cast<float4*>(A + (8 + txb16) * SM + s * M + mb + txm16 * 4);
        rB[0] = *reinterpret_cast<float4*>(B + txb16 * SN + s * N + nb + txm16 * 4);
        rB[1] = *reinterpret_cast<float4*>(B + (8 + txb16) * SN + s * N + nb + txm16 * 4);

        for (kb = 0; kb < K; kb += 16) {

            (*psA)[txb16][txm16 * 4 + 0] = rA[0].x;
            (*psA)[txb16][txm16 * 4 + 1] = rA[0].y;
            (*psA)[txb16][txm16 * 4 + 2] = rA[0].z;
            (*psA)[txb16][txm16 * 4 + 3] = rA[0].w;
            (*psA)[8 + txb16][txm16 * 4 + 0] = rA[1].x;
            (*psA)[8 + txb16][txm16 * 4 + 1] = rA[1].y;
            (*psA)[8 + txb16][txm16 * 4 + 2] = rA[1].z;
            (*psA)[8 + txb16][txm16 * 4 + 3] = rA[1].w;
            (*psB)[txb16][txm16 * 4 + 0] = rB[0].x;
            (*psB)[txb16][txm16 * 4 + 1] = rB[0].y;
            (*psB)[txb16][txm16 * 4 + 2] = rB[0].z;
            (*psB)[txb16][txm16 * 4 + 3] = rB[0].w;
            (*psB)[8 + txb16][txm16 * 4 + 0] = rB[1].x;
            (*psB)[8 + txb16][txm16 * 4 + 1] = rB[1].y;
            (*psB)[8 + txb16][txm16 * 4 + 2] = rB[1].z;
            (*psB)[8 + txb16][txm16 * 4 + 3] = rB[1].w;

            __syncthreads();

            if (kb + 16 < K) {
                rA[0] = *reinterpret_cast<float4*>(A + (kb + 16 + txb16) * SM + s * M + mb + txm16 * 4);
                rA[1] = *reinterpret_cast<float4*>(A + (kb + 24 + txb16) * SM + s * M + mb + txm16 * 4);
                rB[0] = *reinterpret_cast<float4*>(B + (kb + 16 + txb16) * SN + s * N + nb + txm16 * 4);
                rB[1] = *reinterpret_cast<float4*>(B + (kb + 24 + txb16) * SN + s * N + nb + txm16 * 4);
            }

            for (m = 0; m < 8; m += 4) {
                float4 t = *reinterpret_cast<float4*>((*psA)[0] + wmb + m * 4 + txm32b8 * 4);
                rAs[m + 0] = t.x;
                rAs[m + 1] = t.y;
                rAs[m + 2] = t.z;
                rAs[m + 3] = t.w;
            }
            float4 t = *reinterpret_cast<float4*>((*psB)[0] + wnb + txm8 * 4);
            rBs[0] = t.x;
            rBs[1] = t.y;
            rBs[2] = t.z;
            rBs[3] = t.w;

#pragma unroll
            for (k = 0; k < 16; k++) {
                if (k < 15) {
#pragma unroll
                    for (m = 0; m < 8; m += 4) {
                        float4 t = *reinterpret_cast<float4*>((*psA)[k + 1] + wmb + m * 4 + txm32b8 * 4);
                        rAsb[m + 0] = t.x;
                        rAsb[m + 1] = t.y;
                        rAsb[m + 2] = t.z;
                        rAsb[m + 3] = t.w;
                    }
                    float4 t = *reinterpret_cast<float4*>((*psB)[k + 1] + wnb + txm8 * 4);
                    rBsb[0] = t.x;
                    rBsb[1] = t.y;
                    rBsb[2] = t.z;
                    rBsb[3] = t.w;
                }

#pragma unroll
                for (m = 0; m < 8; m++) {
#pragma unroll
                    for (n = 0; n < 4; n++) {
                        rC[m][n] += rAs[m] * rBs[n];
                    }
                }

#pragma unroll
                for (m = 0; m < 8; m++) {
                    rAs[m] = rAsb[m];
                }
#pragma unroll
                for (n = 0; n < 4; n++) {
                    rBs[n] = rBsb[n];
                }

            }

            psA = (psA == &sA) ? &sAb : &sA;
            psB = (psB == &sB) ? &sBb : &sB;

        }

#pragma unroll
        for (m = 0; m < 8; m += 4) {
#pragma unroll
            for (n = 0; n < 4; n++) {
                float4 t;
                t.x = rC[m + n][0];
                t.y = rC[m + n][1];
                t.z = rC[m + n][2];
                t.w = rC[m + n][3];
                *reinterpret_cast<float4*>(C + (mb + wmb + m * 4 + txm32b8 * 4 + n) * SN + s * N + nb + wnb + txm8 * 4) = t;
            }
        }
    }


}

__global__ void sgemm_strided_32x32x16_TN(float* A, float* B, float* C, int M, int N, int K, int S) {

    __shared__ __align__(16) float sA[16][32];
    __shared__ __align__(16) float sB[16][32];
    __shared__ __align__(16) float sAb[16][32];
    __shared__ __align__(16) float sBb[16][32];

    int tx = threadIdx.x;
    int mb = blockIdx.z * 64 + (blockIdx.x / 2) * 32;
    int nb = blockIdx.y * 64 + (blockIdx.x % 2) * 32;

    if (mb >= M || nb >= N) {
        return;
    }

    int txm8 = tx % 8;
    int txb8 = tx / 8;
    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm32m4 = txm32 % 4;
    int txm32b4 = txm32 / 4;
    int txm16b2 = txm16 / 2;
    int txm32m2 = txm32 % 2;
    int txm32b16 = txm32 / 16;

    int SM = S * M;
    int SN = S * N;

    int wmb = 0;
    int wnb = txb32 * 16;

    float rC[4][4];
    float4 rA[2];
    float4 rB[2];
    float rAs[4];
    float rBs[4];
    float rAsb[4];
    float rBsb[4];

    float(*psA)[16][32] = &sA;
    float(*psB)[16][32] = &sB;

    int m, n, k, kb, s;

    for (s = 0; s < S; s++) {
#pragma unroll
        for (m = 0; m < 4; m++) {
#pragma unroll
            for (n = 0; n < 4; n++) {
                rC[m][n] = 0;
            }
        }

#pragma unroll
        for (m = 0; m < 2; m++) {
            rA[m] = *reinterpret_cast<float4*>(A + (m * 8 + txb8) * SM + s * M + mb + txm8 * 4);
            rB[m] = *reinterpret_cast<float4*>(B + (m * 8 + txb8) * SN + s * N + nb + txm8 * 4);
        }

        for (kb = 0; kb < K; kb += 16) {

#pragma unroll
            for (m = 0; m < 2; m++) {
                (*psA)[m * 8 + txb8][txm8 * 4 + 0] = rA[m].x;
                (*psA)[m * 8 + txb8][txm8 * 4 + 1] = rA[m].y;
                (*psA)[m * 8 + txb8][txm8 * 4 + 2] = rA[m].z;
                (*psA)[m * 8 + txb8][txm8 * 4 + 3] = rA[m].w;

                (*psB)[m * 8 + txb8][txm8 * 4 + 0] = rB[m].x;
                (*psB)[m * 8 + txb8][txm8 * 4 + 1] = rB[m].y;
                (*psB)[m * 8 + txb8][txm8 * 4 + 2] = rB[m].z;
                (*psB)[m * 8 + txb8][txm8 * 4 + 3] = rB[m].w;
            }

            __syncthreads();

            if (kb + 16 < K) {
#pragma unroll
                for (m = 0; m < 2; m++) {
                    rA[m] = *reinterpret_cast<float4*>(A + (kb + 16 + m * 8 + txb8) * SM + s * M + mb + txm8 * 4);
                    rB[m] = *reinterpret_cast<float4*>(B + (kb + 16 + m * 8 + txb8) * SN + s * N + nb + txm8 * 4);
                }
            }

            // COMPUTE -------------------------

            float4 tA = *reinterpret_cast<float4*>((*psA)[0] + wmb + txm16b2 * 4);
            rAs[0] = tA.x;
            rAs[1] = tA.y;
            rAs[2] = tA.z;
            rAs[3] = tA.w;
            float4 tB = *reinterpret_cast<float4*>((*psB)[0] + wnb + txm32b16 * 8 + txm32m2 * 4);
            rBs[0] = tB.x;
            rBs[1] = tB.y;
            rBs[2] = tB.z;
            rBs[3] = tB.w;

#pragma unroll
            for (k = 0; k < 16; k++) {

                if (k < 15) {
                    float4 tA = *reinterpret_cast<float4*>((*psA)[k + 1] + wmb + txm16b2 * 4);
                    rAsb[0] = tA.x;
                    rAsb[1] = tA.y;
                    rAsb[2] = tA.z;
                    rAsb[3] = tA.w;
                    float4 tB = *reinterpret_cast<float4*>((*psB)[k + 1] + wnb + txm32b16 * 8 + txm32m2 * 4);
                    rBsb[0] = tB.x;
                    rBsb[1] = tB.y;
                    rBsb[2] = tB.z;
                    rBsb[3] = tB.w;
                }

#pragma unroll
                for (m = 0; m < 4; m++) {
#pragma unroll
                    for (n = 0; n < 4; n++) {
                        rC[m][n] += rAs[m] * rBs[n];
                    }
                }

#pragma unroll
                for (m = 0; m < 4; m++) {
                    rAs[m] = rAsb[m];
                    rBs[m] = rBsb[m];
                }
            }

            // ---------------------------------

            psA = (psA == &sA) ? &sAb : &sA;
            psB = (psB == &sB) ? &sBb : &sB;

        }

#pragma unroll
        for (m = 0; m < 4; m++) {
#pragma unroll
            for (n = 0; n < 4; n += 4) {
                float4 t;
                t.x = rC[m][n + 0];
                t.y = rC[m][n + 1];
                t.z = rC[m][n + 2];
                t.w = rC[m][n + 3];
                *reinterpret_cast<float4*>(C + (mb + wmb + txm16b2 * 4 + m) * SN + s * N + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + n) = t;
            }
        }
    }

}

__global__ void sgemm_strided_8x8x64_TN(float* A, float* B, float* C, int M, int N, int K, int S) {

    __shared__ float rCs[8][8];

    int tx = threadIdx.x;
    int mb = blockIdx.z * 8;
    int nb = blockIdx.y * 8;

    if (mb >= M || nb >= N) {
        return;
    }

    int txm8 = tx % 8;
    int txb8 = tx / 8;
    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm32m4 = txm32 % 4;
    int txm32b4 = txm32 / 4;
    int txm16b2 = txm16 / 2;
    int txm32m2 = txm32 % 2;
    int txm32b16 = txm32 / 16;

    int SM = S * M;
    int SN = S * N;

    float rC[8][8];
    float rA[8];
    float rB[8];
    float rAb[8];
    float rBb[8];

    float4 f4;

    int m, n, k, kb, s;

    for (s = 0; s < S; s++) {
#pragma unroll
        for (m = 0; m < 8; m++) {
#pragma unroll
            for (n = 0; n < 8; n++) {
                rC[m][n] = 0;
            }
        }

        f4 = *reinterpret_cast<float4*>(A + tx * SM + s * M + mb);
        rA[0] = f4.x;
        rA[1] = f4.y;
        rA[2] = f4.z;
        rA[3] = f4.w;
        f4 = *reinterpret_cast<float4*>(A + tx * SM + s * M + mb + 4);
        rA[4] = f4.x;
        rA[5] = f4.y;
        rA[6] = f4.z;
        rA[7] = f4.w;
        f4 = *reinterpret_cast<float4*>(B + tx * SN + s * N + nb);
        rB[0] = f4.x;
        rB[1] = f4.y;
        rB[2] = f4.z;
        rB[3] = f4.w;
        f4 = *reinterpret_cast<float4*>(B + tx * SN + s * N + nb + 4);
        rB[4] = f4.x;
        rB[5] = f4.y;
        rB[6] = f4.z;
        rB[7] = f4.w;

        for (kb = 0; kb < K; kb += 64) {

            if (kb + 64 + tx < K) {

                f4 = *reinterpret_cast<float4*>(A + (kb + 64 + tx) * SM + s * M + mb);
                rAb[0] = f4.x;
                rAb[1] = f4.y;
                rAb[2] = f4.z;
                rAb[3] = f4.w;
                f4 = *reinterpret_cast<float4*>(A + (kb + 64 + tx) * SM + s * M + mb + 4);
                rAb[4] = f4.x;
                rAb[5] = f4.y;
                rAb[6] = f4.z;
                rAb[7] = f4.w;
                f4 = *reinterpret_cast<float4*>(B + (kb + 64 + tx) * SN + s * N + nb);
                rBb[0] = f4.x;
                rBb[1] = f4.y;
                rBb[2] = f4.z;
                rBb[3] = f4.w;
                f4 = *reinterpret_cast<float4*>(B + (kb + 64 + tx) * SN + s * N + nb + 4);
                rBb[4] = f4.x;
                rBb[5] = f4.y;
                rBb[6] = f4.z;
                rBb[7] = f4.w;

            }
            else {
#pragma unroll
                for (k = 0; k < 8; k++) {
                    rAb[k] = 0;
                    rBb[k] = 0;
                }
            }

            // COMPUTE -------------------------

#pragma unroll
            for (m = 0; m < 8; m++) {
#pragma unroll
                for (n = 0; n < 8; n++) {
                    rC[m][n] += rA[m] * rB[n];
                }
            }

#pragma unroll
            for (m = 0; m < 8; m++) {
                rA[m] = rAb[m];
                rB[m] = rBb[m];
            }

        }

#pragma unroll
        for (m = 0; m < 8; m++) {
#pragma unroll
            for (n = 0; n < 8; n++) {
                float t = __shfl_sync(FULL_MASK, rC[m][n], txm32 + 16, 32);
                if (txm32 < 16) {
                    rC[m][n] += t;
                }
                t = __shfl_sync(FULL_MASK, rC[m][n], txm32 + 8, 32);
                if (txm32 < 8) {
                    rC[m][n] += t;
                }
                t = __shfl_sync(FULL_MASK, rC[m][n], txm32 + 4, 32);
                if (txm32 < 4) {
                    rC[m][n] += t;
                }
                t = __shfl_sync(FULL_MASK, rC[m][n], txm32 + 2, 32);
                if (txm32 < 2) {
                    rC[m][n] += t;
                }
                t = __shfl_sync(FULL_MASK, rC[m][n], txm32 + 1, 32);
                if (txm32 < 1) {
                    rC[m][n] += t;
                }
                if (tx == 32) {
                    rCs[m][n] = rC[m][n];
                }
            }
        }

        __syncthreads();

        if (tx == 0) {
#pragma unroll
            for (m = 0; m < 8; m++) {
#pragma unroll
                for (n = 0; n < 8; n += 4) {
                    float4 t;
                    t.x = rC[m][n + 0] + rCs[m][n + 0];
                    t.y = rC[m][n + 1] + rCs[m][n + 1];
                    t.z = rC[m][n + 2] + rCs[m][n + 2];
                    t.w = rC[m][n + 3] + rCs[m][n + 3];
                    *reinterpret_cast<float4*>(C + (mb + m) * SN + s * N + nb + n) = t;
                }
            }
        }
    }

}


__global__ void sgemm_strided_batched_K_32x32x16_TN(float* A, float* B, float* C, int M, int N, int K, int S, int Kc) {

    __shared__ float sA[16][32];
    __shared__ float sB[16][32];
    __shared__ float sAb[16][32];
    __shared__ float sBb[16][32];

    float rC[4][4];

    int tx = threadIdx.x;

    int kbb = blockIdx.x * Kc;
    int mb = blockIdx.y * 32;
    int nb = blockIdx.z * 32;

    float4 rA[2], rB[2];
    float4 rAb[2], rBb[2];

    float(*psA)[16][32] = &sA;
    float(*psB)[16][32] = &sB;

    float rAs[4];
    float rAsb[4];
    float rBs[4];
    float rBsb[4];

    int SM = S * M;
    int SN = S * N;
    int SMN = SM * N;

    int txm8 = tx % 8;
    int txb8 = tx / 8;
    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm16 = tx % 16;
    int txm16b2 = txm16 / 2;
    int txm32m2 = txm32 % 2;
    int txm32b2 = txm32 / 2;
    int txm32b16 = txm32 / 16;

    int wmb = 0;
    int wnb = txb32 * 16;

    int s, m, n, k, kb;

    for (s = 0; s < S; s++) {

#pragma unroll
        for (m = 0; m < 4; m++) {
#pragma unroll
            for (n = 0; n < 4; n++) {
                rC[m][n] = 0;
            }
        }

        if (kbb + 0 + txb8 < K) {
            rA[0] = *reinterpret_cast<float4*>(A + (kbb + 0 + txb8) * SM + s * M + (mb + txm8 * 4));
            rB[0] = *reinterpret_cast<float4*>(B + (kbb + 0 + txb8) * SN + s * N + (nb + txm8 * 4));
        }
        else {
            rA[0].x = 0;
            rA[0].y = 0;
            rA[0].z = 0; 
            rA[0].w = 0;
            rB[0].x = 0; 
            rB[0].y = 0; 
            rB[0].z = 0; 
            rB[0].w = 0;
        }
        if (kbb + 8 + txb8 < K) {
            rA[1] = *reinterpret_cast<float4*>(A + (kbb + 8 + txb8) * SM + s * M + (mb + txm8 * 4));
            rB[1] = *reinterpret_cast<float4*>(B + (kbb + 8 + txb8) * SN + s * N + (nb + txm8 * 4));
        }
        else {
            rA[1].x = 0;
            rA[1].y = 0;
            rA[1].z = 0;
            rA[1].w = 0;
            rB[1].x = 0;
            rB[1].y = 0;
            rB[1].z = 0;
            rB[1].w = 0;
        }

        for (kb = 0; kb < Kc; kb += 16) {

            (*psA)[0 + txb8][txm8 * 4 + 0] = rA[0].x;
            (*psA)[0 + txb8][txm8 * 4 + 1] = rA[0].y;
            (*psA)[0 + txb8][txm8 * 4 + 2] = rA[0].z;
            (*psA)[0 + txb8][txm8 * 4 + 3] = rA[0].w;
            (*psA)[8 + txb8][txm8 * 4 + 0] = rA[1].x;
            (*psA)[8 + txb8][txm8 * 4 + 1] = rA[1].y;
            (*psA)[8 + txb8][txm8 * 4 + 2] = rA[1].z;
            (*psA)[8 + txb8][txm8 * 4 + 3] = rA[1].w;

            (*psB)[0 + txb8][txm8 * 4 + 0] = rB[0].x;
            (*psB)[0 + txb8][txm8 * 4 + 1] = rB[0].y;
            (*psB)[0 + txb8][txm8 * 4 + 2] = rB[0].z;
            (*psB)[0 + txb8][txm8 * 4 + 3] = rB[0].w;
            (*psB)[8 + txb8][txm8 * 4 + 0] = rB[1].x;
            (*psB)[8 + txb8][txm8 * 4 + 1] = rB[1].y;
            (*psB)[8 + txb8][txm8 * 4 + 2] = rB[1].z;
            (*psB)[8 + txb8][txm8 * 4 + 3] = rB[1].w;

            __syncthreads();

            if (kbb + kb + 16 + txb8 < K) {
                rA[0] = *reinterpret_cast<float4*>(A + (kbb + kb + 16 + txb8) * SM + s * M + (mb + txm8 * 4));
                rB[0] = *reinterpret_cast<float4*>(B + (kbb + kb + 16 + txb8) * SN + s * N + (nb + txm8 * 4));
            }
            else {
                rA[0].x = 0;
                rA[0].y = 0;
                rA[0].z = 0;
                rA[0].w = 0;
                rB[0].x = 0;
                rB[0].y = 0;
                rB[0].z = 0;
                rB[0].w = 0;
            }
            if (kbb + kb + 24 + txb8 < K) {
                rA[1] = *reinterpret_cast<float4*>(A + (kbb + kb + 24 + txb8) * SM + s * M + (mb + txm8 * 4));
                rB[1] = *reinterpret_cast<float4*>(B + (kbb + kb + 24 + txb8) * SN + s * N + (nb + txm8 * 4));
            }
            else {
                rA[1].x = 0;
                rA[1].y = 0;
                rA[1].z = 0;
                rA[1].w = 0;
                rB[1].x = 0;
                rB[1].y = 0;
                rB[1].z = 0;
                rB[1].w = 0;
            }

            float4 tA = *reinterpret_cast<float4*>((*psA)[0] + wmb + txm16b2 * 4);
            rAs[0] = tA.x;
            rAs[1] = tA.y;
            rAs[2] = tA.z;
            rAs[3] = tA.w;
            float4 tB = *reinterpret_cast<float4*>((*psB)[0] + wnb + txm32b16 * 8 + txm32m2 * 4);
            rBs[0] = tB.x;
            rBs[1] = tB.y;
            rBs[2] = tB.z;
            rBs[3] = tB.w;

#pragma unroll
            for (k = 0; k < 16; k++) {

                if (k < 15) {
                    float4 tA = *reinterpret_cast<float4*>((*psA)[k + 1] + wmb + txm16b2 * 4);
                    rAsb[0] = tA.x;
                    rAsb[1] = tA.y;
                    rAsb[2] = tA.z;
                    rAsb[3] = tA.w;
                    float4 tB = *reinterpret_cast<float4*>((*psB)[k + 1] + wnb + txm32b16 * 8 + txm32m2 * 4);
                    rBsb[0] = tB.x;
                    rBsb[1] = tB.y;
                    rBsb[2] = tB.z;
                    rBsb[3] = tB.w;
                }

#pragma unroll
                for (m = 0; m < 4; m++) {
#pragma unroll
                    for (n = 0; n < 4; n++) {
                        rC[m][n] += rAs[m] * rBs[n];
                    }
                }

#pragma unroll
                for (m = 0; m < 4; m++) {
                    rAs[m] = rAsb[m];
                    rBs[m] = rBsb[m];
                }

            }

            psA = (psA == &sA) ? &sAb : &sA;
            psB = (psB == &sB) ? &sBb : &sB;

        }

#pragma unroll
        for (m = 0; m < 4; m++) {
#pragma unroll
            for (n = 0; n < 4; n++) {
                C[blockIdx.x * SMN + (mb + wmb + txm16b2 * 4 + m) * SN + s * N + (nb + wnb + txm32b16 * 8 + txm32m2 * 4 + n)] = rC[m][n];
            }
        }

    }

}

__global__ void sgemm_strided_batched_K_32x16x16_TN(float* A, float* B, float* C, int M, int N, int K, int S, int Kc) {

    __shared__ float sA[16][32];
    __shared__ float sB[16][16];
    __shared__ float sAb[16][32];
    __shared__ float sBb[16][16];

    float rC[4][2];

    int tx = threadIdx.x;

    int kbb = blockIdx.x * Kc;
    int mb = blockIdx.y * 32;
    int nb = blockIdx.z * 16;

    float4 rA[2], rB;

    float(*psA)[16][32] = &sA;
    float(*psB)[16][16] = &sB;

    float rAs[4];
    float rAsb[4];
    float rBs[2];
    float rBsb[2];

    int SM = S * M;
    int SN = S * N;
    int SMN = SM * N;

    int txm4 = tx % 4;
    int txb4 = tx / 4;
    int txm8 = tx % 8;
    int txb8 = tx / 8;
    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm16 = tx % 16;
    int txm16b2 = txm16 / 2;
    int txm32m4 = txm32 % 4;
    int txm32b4 = txm32 / 4;

    int wmb = 0;
    int wnb = txb32 * 8;

    int s, m, n, k, kb;

    for (s = 0; s < S; s++) {

#pragma unroll
        for (m = 0; m < 4; m++) {
#pragma unroll
            for (n = 0; n < 2; n++) {
                rC[m][n] = 0;
            }
        }

        if (kbb + 0 + txb8 < K) {
            rA[0] = *reinterpret_cast<float4*>(A + (kbb + 0 + txb8) * SM + s * M + (mb + txm8 * 4));
        }
        else {
            rA[0].x = rA[0].y = rA[0].z = rA[0].w = 0;
        }
        if (kbb + 8 + txb8 < K) {
            rA[1] = *reinterpret_cast<float4*>(A + (kbb + 8 + txb8) * SM + s * M + (mb + txm8 * 4));
        }
        else {
            rA[1].x = rA[1].y = rA[1].z = rA[1].w = 0;
        }
        if (kbb + txb4 < K) {
            rB = *reinterpret_cast<float4*>(B + (kbb + txb4) * SN + s * N + (nb + txm4 * 4));
        }
        else {
            rB.x = rB.y = rB.z = rB.w = 0;
        }

        for (kb = 0; kb < Kc; kb += 16) {

            (*psA)[0 + txb8][txm8 * 4 + 0] = rA[0].x;
            (*psA)[0 + txb8][txm8 * 4 + 1] = rA[0].y;
            (*psA)[0 + txb8][txm8 * 4 + 2] = rA[0].z;
            (*psA)[0 + txb8][txm8 * 4 + 3] = rA[0].w;
            (*psA)[8 + txb8][txm8 * 4 + 0] = rA[1].x;
            (*psA)[8 + txb8][txm8 * 4 + 1] = rA[1].y;
            (*psA)[8 + txb8][txm8 * 4 + 2] = rA[1].z;
            (*psA)[8 + txb8][txm8 * 4 + 3] = rA[1].w;

            (*psB)[txb4][txm4 * 4 + 0] = rB.x;
            (*psB)[txb4][txm4 * 4 + 1] = rB.y;
            (*psB)[txb4][txm4 * 4 + 2] = rB.z;
            (*psB)[txb4][txm4 * 4 + 3] = rB.w;

            __syncthreads();

            if (kbb + kb + 16 + txb8 < K) {
                rA[0] = *reinterpret_cast<float4*>(A + (kbb + kb + 16 + txb8) * SM + s * M + (mb + txm8 * 4));
            }
            else {
                rA[0].x = rA[0].y = rA[0].z = rA[0].w = 0;
            }
            if (kbb + kb + 24 + txb8 < K) {
                rA[1] = *reinterpret_cast<float4*>(A + (kbb + kb + 24 + txb8) * SM + s * M + (mb + txm8 * 4));
            }
            else {
                rA[1].x = rA[1].y = rA[1].z = rA[1].w = 0;
            }
            if (kbb + kb + 16 + txb4 < K) {
                rB = *reinterpret_cast<float4*>(B + (kbb + kb + 16 + txb4) * SN + s * N + (nb + txm4 * 4));
            }
            else {
                rB.x = rB.y = rB.z = rB.w = 0;
            }

            float4 tA = *reinterpret_cast<float4*>((*psA)[0] + wmb + txm32b4 * 4);
            rAs[0] = tA.x;
            rAs[1] = tA.y;
            rAs[2] = tA.z;
            rAs[3] = tA.w;
            float2 tB = *reinterpret_cast<float2*>((*psB)[0] + wnb + txm32m4 * 2);
            rBs[0] = tB.x;
            rBs[1] = tB.y;

#pragma unroll
            for (k = 0; k < 16; k++) {

                if (k < 15) {
                    float4 tA = *reinterpret_cast<float4*>((*psA)[k + 1] + wmb + txm32b4 * 4);
                    rAsb[0] = tA.x;
                    rAsb[1] = tA.y;
                    rAsb[2] = tA.z;
                    rAsb[3] = tA.w;
                    float2 tB = *reinterpret_cast<float2*>((*psB)[k + 1] + wnb + txm32m4 * 2);
                    rBsb[0] = tB.x;
                    rBsb[1] = tB.y;
                }

#pragma unroll
                for (m = 0; m < 4; m++) {
#pragma unroll
                    for (n = 0; n < 2; n++) {
                        rC[m][n] += rAs[m] * rBs[n];
                    }
                }

#pragma unroll
                for (m = 0; m < 4; m++) {
                    rAs[m] = rAsb[m];
                }
#pragma unroll
                for (n = 0; n < 2; n++) {
                    rBs[n] = rBsb[n];
                }

            }

            psA = (psA == &sA) ? &sAb : &sA;
            psB = (psB == &sB) ? &sBb : &sB;

        }

#pragma unroll
        for (m = 0; m < 4; m++) {
#pragma unroll
            for (n = 0; n < 2; n++) {
                C[blockIdx.x * SMN + (mb + wmb + txm32b4 * 4 + m) * SN + s * N + (nb + wnb + txm32m4 * 2 + n)] = rC[m][n];
            }
        }

    }

}


__global__ void sgemm_strided_batched_A_128x128x8_NT(float* A, float* B, float* C, int M, int N, int K, int S) {

    __shared__ __align__(16) float sA[8][132];
    __shared__ __align__(16) float sB[8][132];
    __shared__ __align__(16) float sAb[8][132];
    __shared__ __align__(16) float sBb[8][132];

    int tx = threadIdx.x;
    int mb = blockIdx.z * 128;
    int nb = blockIdx.y * 128;

    if (mb >= M || nb >= N) {
        return;
    }

    float rC[8][8];
    float4 rA;
    float4 rB;
    float rAs[8];
    float rBs[8];
    float rAsb[8];
    float rBsb[8];

    float(*psA)[8][132] = &sA;
    float(*psB)[8][132] = &sB;

    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm2 = tx % 2;
    int txb2 = tx / 2;
    int txm8 = tx % 8;
    int txb8 = tx / 8;
    int txm128 = tx % 128;
    int txb128 = tx / 128;
    int txm32b4 = txm32 / 4;

    int SK = S * K;
    int SN = S * N;
    int MSK = M * SK;
    int MSN = M * SN;

    int wnb = (txb32 / 4) * 64;
    int wmb = (txb32 % 4) * 32;

    int m, n, k, kb, kbb, s, b;

    b = blockIdx.x;

    float4 f4;

    int sAi1 = wmb + (txm16 / 4) * 4;
    int sAi2 = 16 + sAi1;
    int sBi1 = wnb + (16 * (txm32 / 16)) + (txm32 % 4) * 4;
    int sBi2 = 32 + sBi1;


    for (s = 0; s < S; s++) {

#pragma unroll
        for (m = 0; m < 8; m++) {
#pragma unroll
            for (n = 0; n < 8; n++) {
                rC[m][n] = 0;
            }
        }

        rA = *reinterpret_cast<float4*>(A + b * MSK + (mb + txb2) * SK + s * K + txm2 * 4);
        rB = *reinterpret_cast<float4*>(B + (nb + txb2) * SK + s * K + txm2 * 4);

        for (kb = 0; kb < K; kb += 8) {

            (*psA)[txm2 * 4 + 0][txb2] = rA.x;
            (*psA)[txm2 * 4 + 1][txb2] = rA.y;
            (*psA)[txm2 * 4 + 2][txb2] = rA.z;
            (*psA)[txm2 * 4 + 3][txb2] = rA.w;
            (*psB)[txm2 * 4 + 0][txb2] = rB.x;
            (*psB)[txm2 * 4 + 1][txb2] = rB.y;
            (*psB)[txm2 * 4 + 2][txb2] = rB.z;
            (*psB)[txm2 * 4 + 3][txb2] = rB.w;

            __syncthreads();

            if (kb + 8 < K) {
                rA = *reinterpret_cast<float4*>(A + b * MSK + (mb + txb2) * SK + s * K + kb + 8 + txm2 * 4);
                rB = *reinterpret_cast<float4*>(B + (nb + txb2) * SK + s * K + kb + 8 + txm2 * 4);
            }

            // COMPUTE -------------------

            f4 = *reinterpret_cast<float4*>((*psA)[0] + sAi1);
            rAs[0] = f4.x;
            rAs[1] = f4.y;
            rAs[2] = f4.z;
            rAs[3] = f4.w;
            f4 = *reinterpret_cast<float4*>((*psA)[0] + sAi2);
            rAs[4] = f4.x;
            rAs[5] = f4.y;
            rAs[6] = f4.z;
            rAs[7] = f4.w;
            f4 = *reinterpret_cast<float4*>((*psB)[0] + sBi1);
            rBs[0] = f4.x;
            rBs[1] = f4.y;
            rBs[2] = f4.z;
            rBs[3] = f4.w;
            f4 = *reinterpret_cast<float4*>((*psB)[0] + sBi2);
            rBs[4] = f4.x;
            rBs[5] = f4.y;
            rBs[6] = f4.z;
            rBs[7] = f4.w;

#pragma unroll
            for (k = 0; k < 8; k++) {

                if (k < 7) {
                    f4 = *reinterpret_cast<float4*>((*psA)[k + 1] + sAi1);
                    rAsb[0] = f4.x;
                    rAsb[1] = f4.y;
                    rAsb[2] = f4.z;
                    rAsb[3] = f4.w;
                    f4 = *reinterpret_cast<float4*>((*psA)[k + 1] + sAi2);
                    rAsb[4] = f4.x;
                    rAsb[5] = f4.y;
                    rAsb[6] = f4.z;
                    rAsb[7] = f4.w;
                    f4 = *reinterpret_cast<float4*>((*psB)[k + 1] + sBi1);
                    rBsb[0] = f4.x;
                    rBsb[1] = f4.y;
                    rBsb[2] = f4.z;
                    rBsb[3] = f4.w;
                    f4 = *reinterpret_cast<float4*>((*psB)[k + 1] + sBi2);
                    rBsb[4] = f4.x;
                    rBsb[5] = f4.y;
                    rBsb[6] = f4.z;
                    rBsb[7] = f4.w;
                }

#pragma unroll
                for (m = 0; m < 8; m++) {
#pragma unroll
                    for (n = 0; n < 8; n++) {
                        rC[m][n] += rAs[m] * rBs[n];
                    }
                }

#pragma unroll
                for (m = 0; m < 8; m++) {
                    rAs[m] = rAsb[m];
                    rBs[m] = rBsb[m];
                }

            }

            // ---------------------------

            if (psA == &sA) {
                psA = &sAb;
                psB = &sBb;
            }
            else {
                psA = &sA;
                psB = &sB;
            }

        }

#pragma unroll
        for (m = 0; m < 4; m++) {
            f4.x = rC[m][0];
            f4.y = rC[m][1];
            f4.z = rC[m][2];
            f4.w = rC[m][3];
            *reinterpret_cast<float4*>(C + b * MSN + (mb + sAi1 + m) * SN + s * N + nb + sBi1) = f4;
            f4.x = rC[4 + m][0];
            f4.y = rC[4 + m][1];
            f4.z = rC[4 + m][2];
            f4.w = rC[4 + m][3];
            *reinterpret_cast<float4*>(C + b * MSN + (mb + sAi2 + m) * SN + s * N + nb + sBi1) = f4;
            f4.x = rC[m][4];
            f4.y = rC[m][5];
            f4.z = rC[m][6];
            f4.w = rC[m][7];
            *reinterpret_cast<float4*>(C + b * MSN + (mb + sAi1 + m) * SN + s * N + nb + sBi2) = f4;
            f4.x = rC[4 + m][4];
            f4.y = rC[4 + m][5];
            f4.z = rC[4 + m][6];
            f4.w = rC[4 + m][7];
            *reinterpret_cast<float4*>(C + b * MSN + (mb + sAi2 + m) * SN + s * N + nb + sBi2) = f4;
        }
    }
}

__global__ void sgemm_strided_batched_A_64x64x16_NT(float* A, float* B, float* C, int M, int N, int K, int S) {

    __shared__ __align__(16) float sA[16][64];
    __shared__ __align__(16) float sB[16][64];
    __shared__ __align__(16) float sAb[16][64];
    __shared__ __align__(16) float sBb[16][64];

    int tx = threadIdx.x;
    int mb = blockIdx.z * 64;
    int nb = blockIdx.y * 64;

    if (mb >= M || nb >= N) {
        return;
    }

    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm64 = tx % 64;
    int txb64 = tx / 64;
    int txm8 = tx % 8;
    int txm32b8 = txm32 / 8;

    int SK = S * K;
    int SN = S * N;
    int MSK = M * SK;
    int MSN = M * SN;

    int wmb = (txb32 / 2) * 32;
    int wnb = (txb32 % 2) * 32;

    int kb, k, m, n, s, b;

    b = blockIdx.x;

    float4 rA[2];
    float4 rB[2];
    float rC[8][4];
    float rAs[8];
    float rBs[4];
    float rAsb[8];
    float rBsb[4];

    float(*psA)[16][64] = &sA;
    float(*psB)[16][64] = &sB;

    for (s = 0; s < S; s++) {
#pragma unroll
        for (m = 0; m < 8; m++) {
#pragma unroll
            for (n = 0; n < 4; n++) {
                rC[m][n] = 0;
            }
        }

        rA[0] = *reinterpret_cast<float4*>(A + b * MSK + (mb + txm32) * SK + s * K + txb32 * 4);
        rA[1] = *reinterpret_cast<float4*>(A + b * MSK + (mb + 32 + txm32) * SK + s * K + txb32 * 4);
        rB[0] = *reinterpret_cast<float4*>(B + (nb + txm32) * SK + s * K + txb32 * 4);
        rB[1] = *reinterpret_cast<float4*>(B + (nb + 32 + txm32) * SK + s * K + txb32 * 4);

        for (kb = 0; kb < K; kb += 16) {

            (*psA)[txb32 * 4 + 0][txm32] = rA[0].x;
            (*psA)[txb32 * 4 + 1][txm32] = rA[0].y;
            (*psA)[txb32 * 4 + 2][txm32] = rA[0].z;
            (*psA)[txb32 * 4 + 3][txm32] = rA[0].w;
            (*psA)[txb32 * 4 + 0][32 + txm32] = rA[1].x;
            (*psA)[txb32 * 4 + 1][32 + txm32] = rA[1].y;
            (*psA)[txb32 * 4 + 2][32 + txm32] = rA[1].z;
            (*psA)[txb32 * 4 + 3][32 + txm32] = rA[1].w;
            (*psB)[txb32 * 4 + 0][txm32] = rB[0].x;
            (*psB)[txb32 * 4 + 1][txm32] = rB[0].y;
            (*psB)[txb32 * 4 + 2][txm32] = rB[0].z;
            (*psB)[txb32 * 4 + 3][txm32] = rB[0].w;
            (*psB)[txb32 * 4 + 0][32 + txm32] = rB[1].x;
            (*psB)[txb32 * 4 + 1][32 + txm32] = rB[1].y;
            (*psB)[txb32 * 4 + 2][32 + txm32] = rB[1].z;
            (*psB)[txb32 * 4 + 3][32 + txm32] = rB[1].w;

            __syncthreads();

            if (kb + 16 < K) {
                rA[0] = *reinterpret_cast<float4*>(A + b * MSK + (mb + txm32) * SK + s * K + kb + 16 + txb32 * 4);
                rA[1] = *reinterpret_cast<float4*>(A + b * MSK + (mb + 32 + txm32) * SK + s * K + kb + 16 + txb32 * 4);
                rB[0] = *reinterpret_cast<float4*>(B + (nb + txm32) * SK + s * K + kb + 16 + txb32 * 4);
                rB[1] = *reinterpret_cast<float4*>(B + (nb + 32 + txm32) * SK + s * K + kb + 16 + txb32 * 4);
            }

            for (m = 0; m < 8; m += 4) {
                float4 t = *reinterpret_cast<float4*>((*psA)[0] + wmb + m * 4 + txm32b8 * 4);
                rAs[m + 0] = t.x;
                rAs[m + 1] = t.y;
                rAs[m + 2] = t.z;
                rAs[m + 3] = t.w;
            }
            float4 t = *reinterpret_cast<float4*>((*psB)[0] + wnb + txm8 * 4);
            rBs[0] = t.x;
            rBs[1] = t.y;
            rBs[2] = t.z;
            rBs[3] = t.w;

#pragma unroll
            for (k = 0; k < 16; k++) {
                if (k < 15) {
#pragma unroll
                    for (m = 0; m < 8; m += 4) {
                        float4 t = *reinterpret_cast<float4*>((*psA)[k + 1] + wmb + m * 4 + txm32b8 * 4);
                        rAsb[m + 0] = t.x;
                        rAsb[m + 1] = t.y;
                        rAsb[m + 2] = t.z;
                        rAsb[m + 3] = t.w;
                    }
                    float4 t = *reinterpret_cast<float4*>((*psB)[k + 1] + wnb + txm8 * 4);
                    rBsb[0] = t.x;
                    rBsb[1] = t.y;
                    rBsb[2] = t.z;
                    rBsb[3] = t.w;
                }

#pragma unroll
                for (m = 0; m < 8; m++) {
#pragma unroll
                    for (n = 0; n < 4; n++) {
                        rC[m][n] += rAs[m] * rBs[n];
                    }
                }

#pragma unroll
                for (m = 0; m < 8; m++) {
                    rAs[m] = rAsb[m];
                }
#pragma unroll
                for (n = 0; n < 4; n++) {
                    rBs[n] = rBsb[n];
                }

            }

            psA = (psA == &sA) ? &sAb : &sA;
            psB = (psB == &sB) ? &sBb : &sB;

        }

#pragma unroll
        for (m = 0; m < 8; m += 4) {
#pragma unroll
            for (n = 0; n < 4; n++) {
                float4 t;
                t.x = rC[m + n][0];
                t.y = rC[m + n][1];
                t.z = rC[m + n][2];
                t.w = rC[m + n][3];
                *reinterpret_cast<float4*>(C + b * MSN + (mb + wmb + m * 4 + txm32b8 * 4 + n) * SN + s * N + nb + wnb + txm8 * 4) = t;
            }
        }
    }


}

__global__ void sgemm_strided_batched_A_32x32x16_NT(float* A, float* B, float* C, int M, int N, int K, int S) {

    __shared__ __align__(16) float sA[16][32];
    __shared__ __align__(16) float sB[16][32];
    __shared__ __align__(16) float sAb[16][32];
    __shared__ __align__(16) float sBb[16][32];

    int tx = threadIdx.x;
    int mb = blockIdx.z * 32;
    int nb = blockIdx.y * 32;

    if (mb >= M || nb >= N) {
        return;
    }

    int txm8 = tx % 8;
    int txb8 = tx / 8;
    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm32m4 = txm32 % 4;
    int txm32b4 = txm32 / 4;
    int txm16b2 = txm16 / 2;
    int txm32m2 = txm32 % 2;
    int txm32b16 = txm32 / 16;

    int SK = S * K;
    int SN = S * N;
    int MSK = M * SK;
    int MSN = M * SN;

    int wmb = 0;
    int wnb = txb32 * 16;

    float rC[4][4];
    float4 rA[2];
    float4 rB[2];
    float rAs[4];
    float rBs[4];
    float rAsb[4];
    float rBsb[4];

    float(*psA)[16][32] = &sA;
    float(*psB)[16][32] = &sB;

    int m, n, k, kb, s, b;

    b = blockIdx.x;

    for (s = 0; s < S; s++) {
#pragma unroll
        for (m = 0; m < 4; m++) {
#pragma unroll
            for (n = 0; n < 4; n++) {
                rC[m][n] = 0;
            }
        }

#pragma unroll
        for (m = 0; m < 2; m++) {
            rA[m] = *reinterpret_cast<float4*>(A + b * MSK + (mb + m * 16 + txm16) * SK + s * K + txb16 * 4);
            rB[m] = *reinterpret_cast<float4*>(B + (nb + m * 16 + txm16) * SK + s * K + txb16 * 4);
        }

        for (kb = 0; kb < K; kb += 16) {

#pragma unroll
            for (m = 0; m < 2; m++) {
                (*psA)[txb16 * 4 + 0][m * 16 + txm16] = rA[m].x;
                (*psA)[txb16 * 4 + 1][m * 16 + txm16] = rA[m].y;
                (*psA)[txb16 * 4 + 2][m * 16 + txm16] = rA[m].z;
                (*psA)[txb16 * 4 + 3][m * 16 + txm16] = rA[m].w;

                (*psB)[txb16 * 4 + 0][m * 16 + txm16] = rB[m].x;
                (*psB)[txb16 * 4 + 1][m * 16 + txm16] = rB[m].y;
                (*psB)[txb16 * 4 + 2][m * 16 + txm16] = rB[m].z;
                (*psB)[txb16 * 4 + 3][m * 16 + txm16] = rB[m].w;
            }

            __syncthreads();

            if (kb + 16 < K) {
#pragma unroll
                for (m = 0; m < 2; m++) {
                    rA[m] = *reinterpret_cast<float4*>(A + b * MSK + (mb + m * 16 + txm16) * SK + s * K + kb + 16 + txb16 * 4);
                    rB[m] = *reinterpret_cast<float4*>(B + (nb + m * 16 + txm16) * SK + s * K + kb + 16 + txb16 * 4);
                }
            }

            // COMPUTE -------------------------

            float4 tA = *reinterpret_cast<float4*>((*psA)[0] + wmb + txm16b2 * 4);
            rAs[0] = tA.x;
            rAs[1] = tA.y;
            rAs[2] = tA.z;
            rAs[3] = tA.w;
            float4 tB = *reinterpret_cast<float4*>((*psB)[0] + wnb + txm32b16 * 8 + txm32m2 * 4);
            rBs[0] = tB.x;
            rBs[1] = tB.y;
            rBs[2] = tB.z;
            rBs[3] = tB.w;

#pragma unroll
            for (k = 0; k < 16; k++) {

                if (k < 15) {
                    float4 tA = *reinterpret_cast<float4*>((*psA)[k + 1] + wmb + txm16b2 * 4);
                    rAsb[0] = tA.x;
                    rAsb[1] = tA.y;
                    rAsb[2] = tA.z;
                    rAsb[3] = tA.w;
                    float4 tB = *reinterpret_cast<float4*>((*psB)[k + 1] + wnb + txm32b16 * 8 + txm32m2 * 4);
                    rBsb[0] = tB.x;
                    rBsb[1] = tB.y;
                    rBsb[2] = tB.z;
                    rBsb[3] = tB.w;
                }

#pragma unroll
                for (m = 0; m < 4; m++) {
#pragma unroll
                    for (n = 0; n < 4; n++) {
                        rC[m][n] += rAs[m] * rBs[n];
                    }
                }

#pragma unroll
                for (m = 0; m < 4; m++) {
                    rAs[m] = rAsb[m];
                    rBs[m] = rBsb[m];
                }
            }

            // ---------------------------------

            psA = (psA == &sA) ? &sAb : &sA;
            psB = (psB == &sB) ? &sBb : &sB;

        }

#pragma unroll
        for (m = 0; m < 4; m++) {
#pragma unroll
            for (n = 0; n < 4; n += 4) {
                float4 t;
                t.x = rC[m][n + 0];
                t.y = rC[m][n + 1];
                t.z = rC[m][n + 2];
                t.w = rC[m][n + 3];
                *reinterpret_cast<float4*>(C + b * MSN + (mb + wmb + txm16b2 * 4 + m) * SN + s * N + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + n) = t;
            }
        }
    }

}


__global__ void sgemm_strided_batched_A_128x128x8_NN(float* A, float* B, float* C, int M, int N, int K, int S) {

    __shared__ __align__(16) float sA[8][132];
    __shared__ __align__(16) float sB[8][128];
    __shared__ __align__(16) float sAb[8][132];
    __shared__ __align__(16) float sBb[8][128];

    int tx = threadIdx.x;
    int mb = blockIdx.z * 128;
    int nb = blockIdx.y * 128;

    if (mb >= M || nb >= N) {
        return;
    }

    float rC[8][8];
    float4 rA;
    float4 rB;
    float rAs[8];
    float rBs[8];
    float rAsb[8];
    float rBsb[8];

    float(*psA)[8][132] = &sA;
    float(*psB)[8][128] = &sB;

    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm2 = tx % 2;
    int txb2 = tx / 2;
    int txm8 = tx % 8;
    int txb8 = tx / 8;
    int txm128 = tx % 128;
    int txb128 = tx / 128;
    int txm32b4 = txm32 / 4;

    int SK = S * K;
    int SN = S * N;
    int MSK = M * SK;
    int MSN = M * SN;

    int wnb = (txb32 / 4) * 64;
    int wmb = (txb32 % 4) * 32;

    int m, n, k, kb, kbb, s, b;

    b = blockIdx.x;

    float4 f4;

    int sAi1 = wmb + (txm16 / 4) * 4;
    int sAi2 = 16 + sAi1;
    int sBi1 = wnb + (16 * (txm32 / 16)) + (txm32 % 4) * 4;
    int sBi2 = 32 + sBi1;

    for (s = 0; s < S; s++) {

#pragma unroll
        for (m = 0; m < 8; m++) {
#pragma unroll
            for (n = 0; n < 8; n++) {
                rC[m][n] = 0;
            }
        }

        rA = *reinterpret_cast<float4*>(A + b * MSK + (mb + txb2) * SK + s * K + txm2 * 4);
        rB = *reinterpret_cast<float4*>(B + txb32 * SN + s * N + nb + txm32 * 4);

        for (kb = 0; kb < K; kb += 8) {

            (*psA)[txm2 * 4 + 0][txb2] = rA.x;
            (*psA)[txm2 * 4 + 1][txb2] = rA.y;
            (*psA)[txm2 * 4 + 2][txb2] = rA.z;
            (*psA)[txm2 * 4 + 3][txb2] = rA.w;
            (*psB)[txb32][txm32 * 4 + 0] = rB.x;
            (*psB)[txb32][txm32 * 4 + 1] = rB.y;
            (*psB)[txb32][txm32 * 4 + 2] = rB.z;
            (*psB)[txb32][txm32 * 4 + 3] = rB.w;

            __syncthreads();

            if (kb + 8 < K) {
                rA = *reinterpret_cast<float4*>(A + b * MSK + (mb + txb2) * SK + s * K + kb + 8 + txm2 * 4);
                rB = *reinterpret_cast<float4*>(B + (kb + 8 + txb32) * SN + s * N + nb + txm32 * 4);
            }

            // COMPUTE -------------------

            f4 = *reinterpret_cast<float4*>((*psA)[0] + sAi1);
            rAs[0] = f4.x;
            rAs[1] = f4.y;
            rAs[2] = f4.z;
            rAs[3] = f4.w;
            f4 = *reinterpret_cast<float4*>((*psA)[0] + sAi2);
            rAs[4] = f4.x;
            rAs[5] = f4.y;
            rAs[6] = f4.z;
            rAs[7] = f4.w;
            f4 = *reinterpret_cast<float4*>((*psB)[0] + sBi1);
            rBs[0] = f4.x;
            rBs[1] = f4.y;
            rBs[2] = f4.z;
            rBs[3] = f4.w;
            f4 = *reinterpret_cast<float4*>((*psB)[0] + sBi2);
            rBs[4] = f4.x;
            rBs[5] = f4.y;
            rBs[6] = f4.z;
            rBs[7] = f4.w;

#pragma unroll
            for (k = 0; k < 8; k++) {

                if (k < 7) {
                    f4 = *reinterpret_cast<float4*>((*psA)[k + 1] + sAi1);
                    rAsb[0] = f4.x;
                    rAsb[1] = f4.y;
                    rAsb[2] = f4.z;
                    rAsb[3] = f4.w;
                    f4 = *reinterpret_cast<float4*>((*psA)[k + 1] + sAi2);
                    rAsb[4] = f4.x;
                    rAsb[5] = f4.y;
                    rAsb[6] = f4.z;
                    rAsb[7] = f4.w;
                    f4 = *reinterpret_cast<float4*>((*psB)[k + 1] + sBi1);
                    rBsb[0] = f4.x;
                    rBsb[1] = f4.y;
                    rBsb[2] = f4.z;
                    rBsb[3] = f4.w;
                    f4 = *reinterpret_cast<float4*>((*psB)[k + 1] + sBi2);
                    rBsb[4] = f4.x;
                    rBsb[5] = f4.y;
                    rBsb[6] = f4.z;
                    rBsb[7] = f4.w;
                }

#pragma unroll
                for (m = 0; m < 8; m++) {
#pragma unroll
                    for (n = 0; n < 8; n++) {
                        rC[m][n] += rAs[m] * rBs[n];
                    }
                }

#pragma unroll
                for (m = 0; m < 8; m++) {
                    rAs[m] = rAsb[m];
                    rBs[m] = rBsb[m];
                }

            }

            // ---------------------------

            if (psA == &sA) {
                psA = &sAb;
                psB = &sBb;
            }
            else {
                psA = &sA;
                psB = &sB;
            }

        }

#pragma unroll
        for (m = 0; m < 4; m++) {
            f4.x = rC[m][0];
            f4.y = rC[m][1];
            f4.z = rC[m][2];
            f4.w = rC[m][3];
            *reinterpret_cast<float4*>(C + b * MSN + (mb + sAi1 + m) * SN + s * N + nb + sBi1) = f4;
            f4.x = rC[4 + m][0];
            f4.y = rC[4 + m][1];
            f4.z = rC[4 + m][2];
            f4.w = rC[4 + m][3];
            *reinterpret_cast<float4*>(C + b * MSN + (mb + sAi2 + m) * SN + s * N + nb + sBi1) = f4;
            f4.x = rC[m][4];
            f4.y = rC[m][5];
            f4.z = rC[m][6];
            f4.w = rC[m][7];
            *reinterpret_cast<float4*>(C + b * MSN + (mb + sAi1 + m) * SN + s * N + nb + sBi2) = f4;
            f4.x = rC[4 + m][4];
            f4.y = rC[4 + m][5];
            f4.z = rC[4 + m][6];
            f4.w = rC[4 + m][7];
            *reinterpret_cast<float4*>(C + b * MSN + (mb + sAi2 + m) * SN + s * N + nb + sBi2) = f4;
        }
    }

}

__global__ void sgemm_strided_batched_A_128x16x8_NN(float* A, float* B, float* C, int M, int N, int K, int S) {

    __shared__ __align__(16) float sA[8][132];
    __shared__ __align__(16) float sB[8][16];
    __shared__ __align__(16) float sAb[8][132];
    __shared__ __align__(16) float sBb[8][16];

    int tx = threadIdx.x;
    int mb = blockIdx.z * 128;
    int nb = blockIdx.y * 16;

    if (mb >= M || nb >= N) {
        return;
    }

    float rC[4][4];
    float4 rA[2];
    float rB;
    float rAs[4];
    float rBs[4];
    float rAsb[4];
    float rBsb[4];

    float(*psA)[8][132] = &sA;
    float(*psB)[8][16] = &sB;

    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm2 = tx % 2;
    int txb2 = tx / 2;
    int txm8 = tx % 8;
    int txb8 = tx / 8;
    int txm128 = tx % 128;
    int txb128 = tx / 128;
    int txm32b4 = txm32 / 4;

    int SK = S * K;
    int SN = S * N;
    int MSK = M * SK;
    int MSN = M * SN;

    int wnb = (txb32 / 2) * 8;
    int wmb = (txb32 % 2) * 64;

    int m, n, k, kb, kbb, s, b;

    b = blockIdx.x;

    float4 t;

    int sAi1 = wmb + (txm32 / 2) * 4;
    int sBi1 = wnb + (txm32 % 2) * 4;

    for (s = 0; s < S; s++) {

#pragma unroll
        for (m = 0; m < 4; m++) {
#pragma unroll
            for (n = 0; n < 4; n++) {
                rC[m][n] = 0;
            }
        }

        rA[0] = *reinterpret_cast<float4*>(A + b * MSK + (mb + txb2) * SK + s * K + txm2 * 4);
        rA[1] = *reinterpret_cast<float4*>(A + b * MSK + (mb + 64 + txb2) * SK + s * K + txm2 * 4);
        rB = B[txb16 * SN + s * N + nb + txm16];

        for (kb = 0; kb < K; kb += 8) {

            (*psA)[txm2 * 4 + 0][txb2] = rA[0].x;
            (*psA)[txm2 * 4 + 1][txb2] = rA[0].y;
            (*psA)[txm2 * 4 + 2][txb2] = rA[0].z;
            (*psA)[txm2 * 4 + 3][txb2] = rA[0].w;
            (*psA)[txm2 * 4 + 0][64 + txb2] = rA[1].x;
            (*psA)[txm2 * 4 + 1][64 + txb2] = rA[1].y;
            (*psA)[txm2 * 4 + 2][64 + txb2] = rA[1].z;
            (*psA)[txm2 * 4 + 3][64 + txb2] = rA[1].w;

            (*psB)[txb16][txm16] = rB;

            __syncthreads();

            if (kb + 8 < K) {
                rA[0] = *reinterpret_cast<float4*>(A + b * MSK + (mb + txb2) * SK + s * K + kb + 8 + txm2 * 4);
                rA[1] = *reinterpret_cast<float4*>(A + b * MSK + (mb + 64 + txb2) * SK + s * K + kb + 8 + txm2 * 4);
                rB = B[(kb + 8 + txb16) * SN + s * N + nb + txm16];
            }

            // COMPUTE -------------------

            float4 t = *reinterpret_cast<float4*>((*psA)[0] + sAi1);
            rAs[0] = t.x;
            rAs[1] = t.y;
            rAs[2] = t.z;
            rAs[3] = t.w;
            t = *reinterpret_cast<float4*>((*psB)[0] + sBi1);
            rBs[0] = t.x;
            rBs[1] = t.y;
            rBs[2] = t.z;
            rBs[3] = t.w;

#pragma unroll
            for (k = 0; k < 8; k++) {

                if (k < 7) {
                    t = *reinterpret_cast<float4*>((*psA)[k + 1] + sAi1);
                    rAsb[0] = t.x;
                    rAsb[1] = t.y;
                    rAsb[2] = t.z;
                    rAsb[3] = t.w;
                    t = *reinterpret_cast<float4*>((*psB)[k + 1] + sBi1);
                    rBsb[0] = t.x;
                    rBsb[1] = t.y;
                    rBsb[2] = t.z;
                    rBsb[3] = t.w;
                }

#pragma unroll
                for (m = 0; m < 4; m++) {
#pragma unroll
                    for (n = 0; n < 4; n++) {
                        rC[m][n] += rAs[m] * rBs[n];
                    }
                }

#pragma unroll
                for (m = 0; m < 4; m++) {
                    rAs[m] = rAsb[m];
                    rBs[m] = rBsb[m];
                }

            }

            // ---------------------------

            if (psA == &sA) {
                psA = &sAb;
                psB = &sBb;
            }
            else {
                psA = &sA;
                psB = &sB;
            }

        }

#pragma unroll
        for (m = 0; m < 4; m++) {
            t.x = rC[m][0];
            t.y = rC[m][1];
            t.z = rC[m][2];
            t.w = rC[m][3];
            *reinterpret_cast<float4*>(C + b * MSN + (mb + sAi1 + m) * SN + s * N + nb + sBi1) = t;
        }
    }

}

__global__ void sgemm_strided_batched_A_64x64x16_NN(float* A, float* B, float* C, int M, int N, int K, int S) {

    __shared__ __align__(16) float sA[16][64];
    __shared__ __align__(16) float sB[16][64];
    __shared__ __align__(16) float sAb[16][64];
    __shared__ __align__(16) float sBb[16][64];

    int tx = threadIdx.x;
    int mb = blockIdx.z * 64;
    int nb = blockIdx.y * 64;

    if (mb >= M || nb >= N) {
        return;
    }

    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm64 = tx % 64;
    int txb64 = tx / 64;
    int txm8 = tx % 8;
    int txm32b8 = txm32 / 8;

    int SK = S * K;
    int SN = S * N;
    int MSK = M * SK;
    int MSN = M * SN;

    int wmb = (txb32 / 2) * 32;
    int wnb = (txb32 % 2) * 32;

    int kb, k, m, n, s, b;

    b = blockIdx.x;

    float4 rA[2];
    float4 rB[2];
    float rC[8][4];
    float rAs[8];
    float rBs[4];
    float rAsb[8];
    float rBsb[4];

    float(*psA)[16][64] = &sA;
    float(*psB)[16][64] = &sB;

    for (s = 0; s < S; s++) {
#pragma unroll
        for (m = 0; m < 8; m++) {
#pragma unroll
            for (n = 0; n < 4; n++) {
                rC[m][n] = 0;
            }
        }

        rA[0] = *reinterpret_cast<float4*>(A + b * MSK + (mb + txm32) * SK + s * K + txb32 * 4);
        rA[1] = *reinterpret_cast<float4*>(A + b * MSK + (mb + 32 + txm32) * SK + s * K + txb32 * 4);
        rB[0] = *reinterpret_cast<float4*>(B + txb16 * SN + s * N + nb + txm16 * 4);
        rB[1] = *reinterpret_cast<float4*>(B + (8 + txb16) * SN + s * N + nb + txm16 * 4);

        for (kb = 0; kb < K; kb += 16) {

            (*psA)[txb32 * 4 + 0][txm32] = rA[0].x;
            (*psA)[txb32 * 4 + 1][txm32] = rA[0].y;
            (*psA)[txb32 * 4 + 2][txm32] = rA[0].z;
            (*psA)[txb32 * 4 + 3][txm32] = rA[0].w;
            (*psA)[txb32 * 4 + 0][32 + txm32] = rA[1].x;
            (*psA)[txb32 * 4 + 1][32 + txm32] = rA[1].y;
            (*psA)[txb32 * 4 + 2][32 + txm32] = rA[1].z;
            (*psA)[txb32 * 4 + 3][32 + txm32] = rA[1].w;
            (*psB)[txb16][txm16 * 4 + 0] = rB[0].x;
            (*psB)[txb16][txm16 * 4 + 1] = rB[0].y;
            (*psB)[txb16][txm16 * 4 + 2] = rB[0].z;
            (*psB)[txb16][txm16 * 4 + 3] = rB[0].w;
            (*psB)[8 + txb16][txm16 * 4 + 0] = rB[1].x;
            (*psB)[8 + txb16][txm16 * 4 + 1] = rB[1].y;
            (*psB)[8 + txb16][txm16 * 4 + 2] = rB[1].z;
            (*psB)[8 + txb16][txm16 * 4 + 3] = rB[1].w;

            __syncthreads();

            if (kb + 16 < K) {
                rA[0] = *reinterpret_cast<float4*>(A + b * MSK + (mb + txm32) * SK + s * K + kb + 16 + txb32 * 4);
                rA[1] = *reinterpret_cast<float4*>(A + b * MSK + (mb + 32 + txm32) * SK + s * K + kb + 16 + txb32 * 4);
                rB[0] = *reinterpret_cast<float4*>(B + (kb + 16 + txb16) * SN + s * N + nb + txm16 * 4);
                rB[1] = *reinterpret_cast<float4*>(B + (kb + 24 + txb16) * SN + s * N + nb + txm16 * 4);
            }

            for (m = 0; m < 8; m += 4) {
                float4 t = *reinterpret_cast<float4*>((*psA)[0] + wmb + m * 4 + txm32b8 * 4);
                rAs[m + 0] = t.x;
                rAs[m + 1] = t.y;
                rAs[m + 2] = t.z;
                rAs[m + 3] = t.w;
            }
            float4 t = *reinterpret_cast<float4*>((*psB)[0] + wnb + txm8 * 4);
            rBs[0] = t.x;
            rBs[1] = t.y;
            rBs[2] = t.z;
            rBs[3] = t.w;

#pragma unroll
            for (k = 0; k < 16; k++) {
                if (k < 15) {
#pragma unroll
                    for (m = 0; m < 8; m += 4) {
                        float4 t = *reinterpret_cast<float4*>((*psA)[k + 1] + wmb + m * 4 + txm32b8 * 4);
                        rAsb[m + 0] = t.x;
                        rAsb[m + 1] = t.y;
                        rAsb[m + 2] = t.z;
                        rAsb[m + 3] = t.w;
                    }
                    float4 t = *reinterpret_cast<float4*>((*psB)[k + 1] + wnb + txm8 * 4);
                    rBsb[0] = t.x;
                    rBsb[1] = t.y;
                    rBsb[2] = t.z;
                    rBsb[3] = t.w;
                }

#pragma unroll
                for (m = 0; m < 8; m++) {
#pragma unroll
                    for (n = 0; n < 4; n++) {
                        rC[m][n] += rAs[m] * rBs[n];
                    }
                }

#pragma unroll
                for (m = 0; m < 8; m++) {
                    rAs[m] = rAsb[m];
                }
#pragma unroll
                for (n = 0; n < 4; n++) {
                    rBs[n] = rBsb[n];
                }

            }

            psA = (psA == &sA) ? &sAb : &sA;
            psB = (psB == &sB) ? &sBb : &sB;

        }

#pragma unroll
        for (m = 0; m < 8; m += 4) {
#pragma unroll
            for (n = 0; n < 4; n++) {
                float4 t;
                t.x = rC[m + n][0];
                t.y = rC[m + n][1];
                t.z = rC[m + n][2];
                t.w = rC[m + n][3];
                *reinterpret_cast<float4*>(C + b * MSN + (mb + wmb + m * 4 + txm32b8 * 4 + n) * SN + s * N + nb + wnb + txm8 * 4) = t;
            }
        }
    }


}

__global__ void sgemm_strided_batched_A_32x32x16_NN(float* A, float* B, float* C, int M, int N, int K, int S) {

    __shared__ __align__(16) float sA[16][32];
    __shared__ __align__(16) float sB[16][32];
    __shared__ __align__(16) float sAb[16][32];
    __shared__ __align__(16) float sBb[16][32];

    int tx = threadIdx.x;
    int mb = blockIdx.z * 32;
    int nb = blockIdx.y * 32;

    if (mb >= M || nb >= N) {
        return;
    }

    int txm8 = tx % 8;
    int txb8 = tx / 8;
    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm32m4 = txm32 % 4;
    int txm32b4 = txm32 / 4;
    int txm16b2 = txm16 / 2;
    int txm32m2 = txm32 % 2;
    int txm32b16 = txm32 / 16;

    int SK = S * K;
    int SN = S * N;
    int MSK = M * SK;
    int MSN = M * SN;

    int wmb = 0;
    int wnb = txb32 * 16;

    float rC[4][4];
    float4 rA[2];
    float4 rB[2];
    float rAs[4];
    float rBs[4];
    float rAsb[4];
    float rBsb[4];

    float(*psA)[16][32] = &sA;
    float(*psB)[16][32] = &sB;

    int m, n, k, kb, s, b;

    b = blockIdx.x;

    for (s = 0; s < S; s++) {
#pragma unroll
        for (m = 0; m < 4; m++) {
#pragma unroll
            for (n = 0; n < 4; n++) {
                rC[m][n] = 0;
            }
        }

#pragma unroll
        for (m = 0; m < 2; m++) {
            rA[m] = *reinterpret_cast<float4*>(A + b * MSK + (mb + m * 16 + txm16) * SK + s * K + txb16 * 4);
            rB[m] = *reinterpret_cast<float4*>(B + (m * 8 + txb8) * SN + s * N + nb + txm8 * 4);
        }

        for (kb = 0; kb < K; kb += 16) {

#pragma unroll
            for (m = 0; m < 2; m++) {
                (*psA)[txb16 * 4 + 0][m * 16 + txm16] = rA[m].x;
                (*psA)[txb16 * 4 + 1][m * 16 + txm16] = rA[m].y;
                (*psA)[txb16 * 4 + 2][m * 16 + txm16] = rA[m].z;
                (*psA)[txb16 * 4 + 3][m * 16 + txm16] = rA[m].w;

                (*psB)[m * 8 + txb8][txm8 * 4 + 0] = rB[m].x;
                (*psB)[m * 8 + txb8][txm8 * 4 + 1] = rB[m].y;
                (*psB)[m * 8 + txb8][txm8 * 4 + 2] = rB[m].z;
                (*psB)[m * 8 + txb8][txm8 * 4 + 3] = rB[m].w;
            }

            __syncthreads();

            if (kb + 16 < K) {
#pragma unroll
                for (m = 0; m < 2; m++) {
                    rA[m] = *reinterpret_cast<float4*>(A + b * MSK + (mb + m * 16 + txm16) * SK + s * K + kb + 16 + txb16 * 4);
                    rB[m] = *reinterpret_cast<float4*>(B + (m * 8 + kb + 16 + txb8) * SN + s * N + nb + txm8 * 4);
                }
            }

            // COMPUTE -------------------------

            float4 tA = *reinterpret_cast<float4*>((*psA)[0] + wmb + txm16b2 * 4);
            rAs[0] = tA.x;
            rAs[1] = tA.y;
            rAs[2] = tA.z;
            rAs[3] = tA.w;
            float4 tB = *reinterpret_cast<float4*>((*psB)[0] + wnb + txm32b16 * 8 + txm32m2 * 4);
            rBs[0] = tB.x;
            rBs[1] = tB.y;
            rBs[2] = tB.z;
            rBs[3] = tB.w;

#pragma unroll
            for (k = 0; k < 16; k++) {

                if (k < 15) {
                    float4 tA = *reinterpret_cast<float4*>((*psA)[k + 1] + wmb + txm16b2 * 4);
                    rAsb[0] = tA.x;
                    rAsb[1] = tA.y;
                    rAsb[2] = tA.z;
                    rAsb[3] = tA.w;
                    float4 tB = *reinterpret_cast<float4*>((*psB)[k + 1] + wnb + txm32b16 * 8 + txm32m2 * 4);
                    rBsb[0] = tB.x;
                    rBsb[1] = tB.y;
                    rBsb[2] = tB.z;
                    rBsb[3] = tB.w;
                }

#pragma unroll
                for (m = 0; m < 4; m++) {
#pragma unroll
                    for (n = 0; n < 4; n++) {
                        rC[m][n] += rAs[m] * rBs[n];
                    }
                }

#pragma unroll
                for (m = 0; m < 4; m++) {
                    rAs[m] = rAsb[m];
                    rBs[m] = rBsb[m];
                }
            }

            // ---------------------------------

            psA = (psA == &sA) ? &sAb : &sA;
            psB = (psB == &sB) ? &sBb : &sB;

        }

#pragma unroll
        for (m = 0; m < 4; m++) {
#pragma unroll
            for (n = 0; n < 4; n += 4) {
                float4 t;
                t.x = rC[m][n + 0];
                t.y = rC[m][n + 1];
                t.z = rC[m][n + 2];
                t.w = rC[m][n + 3];
                *reinterpret_cast<float4*>(C + b * MSN + (mb + wmb + txm16b2 * 4 + m) * SN + s * N + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + n) = t;
            }
        }
    }

}

__global__ void sgemm_strided_batched_A_16x16x16_NN(float* A, float* B, float* C, int M, int N, int K, int S) {

    __shared__ __align__(16) float sA[16][16];
    __shared__ __align__(16) float sB[16][16];
    __shared__ __align__(16) float sAb[16][16];
    __shared__ __align__(16) float sBb[16][16];

    int tx = threadIdx.x;
    int mb = blockIdx.z * 16;
    int nb = blockIdx.y * 16;

    if (mb >= M || nb >= N) {
        return;
    }

    int txm4 = tx % 4;
    int txb4 = tx / 4;
    int txm8 = tx % 8;
    int txb8 = tx / 8;
    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm32m4 = txm32 % 4;
    int txm32b4 = txm32 / 4;
    int txm16b2 = txm16 / 2;
    int txm32m2 = txm32 % 2;
    int txm32b16 = txm32 / 16;

    int SK = S * K;
    int SN = S * N;
    int MSK = M * SK;
    int MSN = M * SN;

    int wmb = 0;
    int wnb = txb32 * 8;

    float rC[2][2];
    float4 rA;
    float4 rB;
    float rAs[2];
    float rBs[2];
    float rAsb[2];
    float rBsb[2];

    float(*psA)[16][16] = &sA;
    float(*psB)[16][16] = &sB;

    int m, n, k, kb, s, b;

    b = blockIdx.x;

    for (s = 0; s < S; s++) {
#pragma unroll
        for (m = 0; m < 2; m++) {
#pragma unroll
            for (n = 0; n < 2; n++) {
                rC[m][n] = 0;
            }
        }

        rA = *reinterpret_cast<float4*>(A + b * MSK + (mb + txb4) * SK + s * K + txm4 * 4);
        rB = *reinterpret_cast<float4*>(B + txb4 * SN + s * N + nb + txm4 * 4);

        for (kb = 0; kb < K; kb += 16) {

            (*psA)[txm4 * 4 + 0][txb4] = rA.x;
            (*psA)[txm4 * 4 + 1][txb4] = rA.y;
            (*psA)[txm4 * 4 + 2][txb4] = rA.z;
            (*psA)[txm4 * 4 + 3][txb4] = rA.w;

            (*psB)[txb4][txm4 * 4 + 0] = rB.x;
            (*psB)[txb4][txm4 * 4 + 1] = rB.y;
            (*psB)[txb4][txm4 * 4 + 2] = rB.z;
            (*psB)[txb4][txm4 * 4 + 3] = rB.w;

            __syncthreads();

            if (kb + 16 < K) {
                rA = *reinterpret_cast<float4*>(A + b * MSK + (mb + txb4) * SK + s * K + kb + 16 + txm4 * 4);
                rB = *reinterpret_cast<float4*>(B + (kb + 16 + txb4) * SN + s * N + nb + txm4 * 4);
            }

            // COMPUTE -------------------------

            float2 tA = *reinterpret_cast<float2*>((*psA)[0] + wmb + txm32b4 * 2);
            rAs[0] = tA.x;
            rAs[1] = tA.y;
            float2 tB = *reinterpret_cast<float2*>((*psB)[0] + wnb + txm32m4 * 2);
            rBs[0] = tB.x;
            rBs[1] = tB.y;

#pragma unroll
            for (k = 0; k < 16; k++) {

                if (k < 15) {
                    float2 tA = *reinterpret_cast<float2*>((*psA)[k + 1] + wmb + txm32b4 * 2);
                    rAsb[0] = tA.x;
                    rAsb[1] = tA.y;
                    float2 tB = *reinterpret_cast<float2*>((*psB)[k + 1] + wnb + txm32m4 * 2);
                    rBsb[0] = tB.x;
                    rBsb[1] = tB.y;
                }

#pragma unroll
                for (m = 0; m < 2; m++) {
#pragma unroll
                    for (n = 0; n < 2; n++) {
                        rC[m][n] += rAs[m] * rBs[n];
                    }
                }

#pragma unroll
                for (m = 0; m < 2; m++) {
                    rAs[m] = rAsb[m];
                    rBs[m] = rBsb[m];
                }
            }

            // ---------------------------------

            psA = (psA == &sA) ? &sAb : &sA;
            psB = (psB == &sB) ? &sBb : &sB;

        }

#pragma unroll
        for (m = 0; m < 2; m++) {
#pragma unroll
            for (n = 0; n < 2; n += 2) {
                float2 t;
                t.x = rC[m][n + 0];
                t.y = rC[m][n + 1];
                *reinterpret_cast<float2*>(C + b * MSN + (mb + wmb + txm32b4 * 2 + m) * SN + s * N + nb + wnb + txm32m4 * 2 + n) = t;
            }
        }
    }

}



DLLEXPORT void cuda_sgemm(float* A, float* B, float* C, int M, int N, int K, int kern_id, int tA, int tB) {

    if (kern_id == 3) {
        int aligned = ((unsigned long)A % 16 == 0 && (unsigned long)B % 16 == 0 && (unsigned long)C % 16 == 0 && N % 4 == 0) ? 1 : 0;
        dim3 gridDims(4, (int)ceil((float)N / 256), (int)ceil((float)M / 256));
        dim3 blockDims(256, 1, 1);
        sgemm_128x128x1 << < gridDims, blockDims >> > (A, B, C, M, N, K, aligned);
        return;
    }

    if (tA == 0 && tB == 0) {
        int aligned = ((unsigned long)A % 16 == 0 && (unsigned long)B % 16 == 0 && (unsigned long)C % 16 == 0 && K % 4 == 0 && N % 4 == 0) ? 1 : 0;
        if (kern_id == 0) {
            dim3 gridDims(4, (int)ceil((float)N / 256), (int)ceil((float)M / 256));
            dim3 blockDims(256, 1, 1);
            sgemm_128x128x8_NN << < gridDims, blockDims >> > (A, B, C, M, N, K, aligned);
        }
        else if (kern_id == 1) {
            dim3 gridDims(4, (int)ceil((float)N / 128), (int)ceil((float)M / 128));
            dim3 blockDims(128, 1, 1);
            sgemm_64x64x16_NN << < gridDims, blockDims >> > (A, B, C, M, N, K, aligned);
        }
        else if (kern_id == 2) {
            dim3 gridDims(4, (int)ceil((float)N / 64), (int)ceil((float)M / 64));
            dim3 blockDims(64, 1, 1);
            sgemm_32x32x16_NN << < gridDims, blockDims >> > (A, B, C, M, N, K, aligned);
        }
        else if (kern_id == 4) {
            aligned = ((unsigned long)A % 16 == 0 && (unsigned long)B % 16 == 0 && (unsigned long)C % 16 == 0 && N % 4 == 0) ? 1 : 0;
            dim3 gridDims((int)ceil((float)N / 4), 1, 1);
            dim3 blockDims(256, 1, 1);
            sgemm_1x4x256_NN << < gridDims, blockDims >> > (A, B, C, M, N, K, aligned);
        }
        else if (kern_id == 5) {
            dim3 gridDims((int)ceil((float)M / 4), (int)ceil((float)N / 4), 1);
            dim3 blockDims(256, 1, 1);            
            sgemm_4x4x256_NN << < gridDims, blockDims >> > (A, B, C, M, N, K, aligned);
        }
    }
    else if (tA == 1 && tB == 0) {
        int aligned = ((unsigned long)A % 16 == 0 && (unsigned long)B % 16 == 0 && (unsigned long)C % 16 == 0 && M % 4 == 0 && N % 4 == 0) ? 1 : 0;
        if (kern_id == 0) {
            dim3 gridDims(4, (int)ceil((float)N / 256), (int)ceil((float)M / 256));
            dim3 blockDims(256, 1, 1);
            sgemm_128x128x8_TN << < gridDims, blockDims >> > (A, B, C, M, N, K, aligned);
        }
        else if (kern_id == 1) {
            dim3 gridDims(4, (int)ceil((float)N / 128), (int)ceil((float)M / 128));
            dim3 blockDims(128, 1, 1);
            sgemm_64x64x16_TN << < gridDims, blockDims >> > (A, B, C, M, N, K, aligned);
        }
        else if (kern_id == 2) {
            dim3 gridDims(4, (int)ceil((float)N / 64), (int)ceil((float)M / 64));
            dim3 blockDims(64, 1, 1);
            sgemm_32x32x16_TN << < gridDims, blockDims >> > (A, B, C, M, N, K, aligned);
        }
        else if (kern_id == 5) {
            dim3 gridDims((int)ceil((float)M / 4), (int)ceil((float)N / 4), 1);
            dim3 blockDims(256, 1, 1);
            sgemm_4x4x256_TN << < gridDims, blockDims >> > (A, B, C, M, N, K, aligned);
        }
    }
    else if (tA == 0 && tB == 1) {
        int aligned = ((unsigned long)A % 16 == 0 && (unsigned long)B % 16 == 0 && (unsigned long)C % 16 == 0 && K % 4 == 0 && N % 4 == 0) ? 1 : 0;
        if (kern_id == 0) {
            dim3 gridDims(4, (int)ceil((float)N / 256), (int)ceil((float)M / 256));
            dim3 blockDims(256, 1, 1);
            sgemm_128x128x8_NT << < gridDims, blockDims >> > (A, B, C, M, N, K, aligned);
        }
        else if (kern_id == 1) {
            dim3 gridDims(4, (int)ceil((float)N / 128), (int)ceil((float)M / 128));
            dim3 blockDims(128, 1, 1);
            sgemm_64x64x16_NT << < gridDims, blockDims >> > (A, B, C, M, N, K, aligned);
        }
        else if (kern_id == 2) {
            dim3 gridDims(4, (int)ceil((float)N / 64), (int)ceil((float)M / 64));
            dim3 blockDims(64, 1, 1);
            sgemm_32x32x16_NT << < gridDims, blockDims >> > (A, B, C, M, N, K, aligned);
        }
        else if (kern_id == 5) {
            dim3 gridDims((int)ceil((float)M / 4), (int)ceil((float)N / 4), 1);
            dim3 blockDims(256, 1, 1);
            sgemm_4x4x256_NT << < gridDims, blockDims >> > (A, B, C, M, N, K, aligned);
        }
    }

}

DLLEXPORT void cuda_sgemm_strided(float* A, float* B, float* C, int M, int N, int K, int stride, int tA, int tB) {

    if (tA == 0 && tB == 0) {
        if (M >= 512 && N >= 512) {
            dim3 gridDims(4, (int)ceil((float)N / 256), (int)ceil((float)M / 256));
            dim3 blockDims(256, 1, 1);
            sgemm_strided_128x128x8_NN << < gridDims, blockDims >> > (A, B, C, M, N, K, stride);
        }
        else if (M >= 512 && N <= 64) {
            dim3 gridDims(4, (int)ceil((float)N / 32), (int)ceil((float)M / 256));
            dim3 blockDims(128, 1, 1);
            sgemm_strided_128x16x8_NN << < gridDims, blockDims >> > (A, B, C, M, N, K, stride);
        }
        else if (M >= 256 && N >= 256) {
            dim3 gridDims(4, (int)ceil((float)N / 128), (int)ceil((float)M / 128));
            dim3 blockDims(128, 1, 1);
            sgemm_strided_64x64x16_NN << < gridDims, blockDims >> > (A, B, C, M, N, K, stride);
        }
        else if (M >= 128 && N >= 128) {
            dim3 gridDims(4, (int)ceil((float)N / 64), (int)ceil((float)M / 64));
            dim3 blockDims(64, 1, 1);
            sgemm_strided_32x32x16_NN << < gridDims, blockDims >> > (A, B, C, M, N, K, stride);
        }
        else {
            dim3 gridDims(4, (int)ceil((float)N / 32), (int)ceil((float)M / 32));
            dim3 blockDims(64, 1, 1);
            sgemm_strided_16x16x16_NN << < gridDims, blockDims >> > (A, B, C, M, N, K, stride);
        }
    }
    else if (tA == 0 && tB == 1) {
        if (M >= 128 && N >= 128) {
            dim3 gridDims(4, (int)ceil((float)N / 256), (int)ceil((float)M / 256));
            dim3 blockDims(256, 1, 1);
            sgemm_strided_128x128x8_NT << < gridDims, blockDims >> > (A, B, C, M, N, K, stride);
        }
        else if (M >= 64 && N >= 64) {
            dim3 gridDims(4, (int)ceil((float)N / 128), (int)ceil((float)M / 128));
            dim3 blockDims(128, 1, 1);
            sgemm_strided_64x64x16_NT << < gridDims, blockDims >> > (A, B, C, M, N, K, stride);
        }
        else {
            dim3 gridDims(4, (int)ceil((float)N / 64), (int)ceil((float)M / 64));
            dim3 blockDims(64, 1, 1);
            sgemm_strided_32x32x16_NT << < gridDims, blockDims >> > (A, B, C, M, N, K, stride);
        }
    }
    else if (tA == 1 && tB == 0) {
        if (M >= 512 && N >= 512) {
            dim3 gridDims(4, (int)ceil((float)N / 256), (int)ceil((float)M / 256));
            dim3 blockDims(256, 1, 1);
            sgemm_strided_128x128x8_TN << < gridDims, blockDims >> > (A, B, C, M, N, K, stride);
        }
        else if (M >= 256 && N >= 256) {
            dim3 gridDims(4, (int)ceil((float)N / 128), (int)ceil((float)M / 128));
            dim3 blockDims(128, 1, 1);
            sgemm_strided_64x64x16_TN << < gridDims, blockDims >> > (A, B, C, M, N, K, stride);
        }
        else if (M >= 128 && N >= 128){
            dim3 gridDims(4, (int)ceil((float)N / 64), (int)ceil((float)M / 64));
            dim3 blockDims(64, 1, 1);
            sgemm_strided_32x32x16_TN << < gridDims, blockDims >> > (A, B, C, M, N, K, stride);
        }
        else {
            dim3 gridDims(1, (int)ceil((float)N / 8), (int)ceil((float)M / 8));
            dim3 blockDims(64, 1, 1);
            sgemm_strided_8x8x64_TN << < gridDims, blockDims >> > (A, B, C, M, N, K, stride);
        }
    }

}

DLLEXPORT void cuda_sgemm_strided_batched_K(float* A, float* B, float* T, int M, int N, int K, int Kb, int stride, int tA, int tB) {

    int Kc = K / Kb;

    if (tA == 1 && tB == 0) {
        if (N <= 16) {
            dim3 gridDims(Kb, (int)ceil((float)M / 32), (int)ceil((float)N / 16));
            dim3 blockDims(64, 1, 1);
            sgemm_strided_batched_K_32x16x16_TN << < gridDims, blockDims >> > (A, B, T, M, N, K, stride, Kc);
        }
        else {
            dim3 gridDims(Kb, (int)ceil((float)M / 32), (int)ceil((float)N / 32));
            dim3 blockDims(64, 1, 1);
            sgemm_strided_batched_K_32x32x16_TN << < gridDims, blockDims >> > (A, B, T, M, N, K, stride, Kc);
        }
    }
    else {

    }

}

DLLEXPORT void cuda_sgemm_strided_batched_A(float* A, float* B, float* C, int M, int N, int K, int stride, int tA, int tB, int Ab) {

    if (tA == 0 && tB == 1) {
        if (M >= 256 && N >= 256) {
            dim3 gridDims(Ab, (int)ceil((float)N / 128), (int)ceil((float)M / 128));
            dim3 blockDims(256, 1, 1);
            sgemm_strided_batched_A_128x128x8_NT << < gridDims, blockDims >> > (A, B, C, M, N, K, stride);
        }
        else if (M >= 128 && N >= 128) {
            dim3 gridDims(Ab, (int)ceil((float)N / 64), (int)ceil((float)M / 64));
            dim3 blockDims(128, 1, 1);
            sgemm_strided_batched_A_64x64x16_NT << < gridDims, blockDims >> > (A, B, C, M, N, K, stride);
        }
        else {
            dim3 gridDims(Ab, (int)ceil((float)N / 32), (int)ceil((float)M / 32));
            dim3 blockDims(64, 1, 1);
            sgemm_strided_batched_A_32x32x16_NT << < gridDims, blockDims >> > (A, B, C, M, N, K, stride);
        }
    }
    else if (tA == 0 && tB == 0) {
        if (M >= 256 && N >= 256) {
            dim3 gridDims(Ab, (int)ceil((float)N / 128), (int)ceil((float)M / 128));
            dim3 blockDims(256, 1, 1);
            sgemm_strided_batched_A_128x128x8_NN << < gridDims, blockDims >> > (A, B, C, M, N, K, stride);
        }
        else if (M >= 256 && N <= 32) {
            dim3 gridDims(Ab, (int)ceil((float)N / 16), (int)ceil((float)M / 128));
            dim3 blockDims(128, 1, 1);
            sgemm_strided_batched_A_128x16x8_NN << < gridDims, blockDims >> > (A, B, C, M, N, K, stride);
        }
        else if (M >= 128 && N >= 128) {
            dim3 gridDims(Ab, (int)ceil((float)N / 64), (int)ceil((float)M / 64));
            dim3 blockDims(128, 1, 1);
            sgemm_strided_batched_A_64x64x16_NN << < gridDims, blockDims >> > (A, B, C, M, N, K, stride);
        }
        else if (M >= 64 && N >= 64) {
            dim3 gridDims(Ab, (int)ceil((float)N / 32), (int)ceil((float)M / 32));
            dim3 blockDims(64, 1, 1);
            sgemm_strided_batched_A_32x32x16_NN << < gridDims, blockDims >> > (A, B, C, M, N, K, stride);
        }
        else {
            dim3 gridDims(Ab, (int)ceil((float)N / 16), (int)ceil((float)M / 16));
            dim3 blockDims(64, 1, 1);
            sgemm_strided_batched_A_16x16x16_NN << < gridDims, blockDims >> > (A, B, C, M, N, K, stride);
        }
    }
}