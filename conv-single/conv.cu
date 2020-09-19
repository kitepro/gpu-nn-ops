#include<time.h>
#include<iostream>

#define DLLEXPORT extern "C" __declspec(dllexport)

#define FULL_MASK 0xFFFFFFFF

struct float8 {
    float a, b, c, d, e, f, g, h;
};


__global__ void conv_sgemm_128x128x8_f(float* A, float* B, float* C, int H, int W, int Ch, int* M, int* N, int* K, int* offsetsB, int* offsetsC, int* Ks128, int minK) {


    __shared__ __align__(16) float sA[8][128];
    __shared__ __align__(16) float sB[8][128];
    __shared__ __align__(16) float sAb[8][128];
    __shared__ __align__(16) float sBb[8][128];

    int tx = threadIdx.x;
    int mb = blockIdx.x * 128;
    int nb = blockIdx.y * 128;
    int Kt = Ks128[blockIdx.z];

    if (mb >= M[Kt] || nb >= N[Kt]) { return; }

    B += offsetsB[Kt];
    C += offsetsC[Kt];

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

    int wnb = (txb32 / 4) * 64;
    int wmb = (txb32 % 4) * 32;

    int WC = W * Ch;

    int m, n, k, kb, hbb, pbase, p;

    int wc = ceil((float)W / (Kt + minK));
    pbase = ((mb + txm128) % wc) * (Kt + minK) * Ch;
    p = pbase + txb128 * 4;
    hbb = ((mb + txm128) / wc) * (Kt + minK);
    int hcap = min(hbb + (Kt + minK), H);
    int cap = (hbb < H) ? pbase + (Kt + minK) * Ch : 0;

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

    if (mb + txm128 < M[Kt]) {
        rA.x = (p < cap) ? ((p < WC) ? A[hbb * WC + p++] : (0 * p++)) : ((hbb + (p - pbase) / (cap - pbase) < hcap) ? ((pbase + (p - pbase) % (cap - pbase) < WC) ? A[(hbb += (p - pbase) / (cap - pbase)) * WC + (p = pbase + (p - pbase) % (cap - pbase))++] : 0 * (hbb += (p - pbase) / (cap - pbase)) * (p = pbase + (p - pbase) % (cap - pbase))++) : 0);
        rA.y = (p < cap) ? ((p < WC) ? A[hbb * WC + p++] : (0 * p++)) : ((hbb + (p - pbase) / (cap - pbase) < hcap) ? ((pbase + (p - pbase) % (cap - pbase) < WC) ? A[(hbb += (p - pbase) / (cap - pbase)) * WC + (p = pbase + (p - pbase) % (cap - pbase))++] : 0 * (hbb += (p - pbase) / (cap - pbase)) * (p = pbase + (p - pbase) % (cap - pbase))++) : 0);
        rA.z = (p < cap) ? ((p < WC) ? A[hbb * WC + p++] : (0 * p++)) : ((hbb + (p - pbase) / (cap - pbase) < hcap) ? ((pbase + (p - pbase) % (cap - pbase) < WC) ? A[(hbb += (p - pbase) / (cap - pbase)) * WC + (p = pbase + (p - pbase) % (cap - pbase))++] : 0 * (hbb += (p - pbase) / (cap - pbase)) * (p = pbase + (p - pbase) % (cap - pbase))++) : 0);
        rA.w = (p < cap) ? ((p < WC) ? A[hbb * WC + p++] : (0 * p++)) : ((hbb + (p - pbase) / (cap - pbase) < hcap) ? ((pbase + (p - pbase) % (cap - pbase) < WC) ? A[(hbb += (p - pbase) / (cap - pbase)) * WC + (p = pbase + (p - pbase) % (cap - pbase))++] : 0 * (hbb += (p - pbase) / (cap - pbase)) * (p = pbase + (p - pbase) % (cap - pbase))++) : 0);
        p += 4;
    }
    else {
        rA.x = rA.y = rA.z = rA.w = 0;
    }

    if (nb + txm128 < N[Kt]) {
        if (txb128 * 4 + 4 <= K[Kt]) {
            rB.x = B[(nb + txm128) * K[Kt] + txb128 * 4 + 0];
            rB.y = B[(nb + txm128) * K[Kt] + txb128 * 4 + 1];
            rB.z = B[(nb + txm128) * K[Kt] + txb128 * 4 + 2];
            rB.w = B[(nb + txm128) * K[Kt] + txb128 * 4 + 3];
            //rB = *reinterpret_cast<float4*>(B + (nb + txm128) * K[Kt] + txb128 * 4);
        }
        else if (txb128 * 4 + 3 <= K[Kt]) {
            rB.x = B[(nb + txm128) * K[Kt] + txb128 * 4 + 0];
            rB.y = B[(nb + txm128) * K[Kt] + txb128 * 4 + 1];
            rB.z = B[(nb + txm128) * K[Kt] + txb128 * 4 + 2];
            rB.w = 0;
        }
        else if (txb128 * 4 + 2 <= K[Kt]) {
            rB.x = B[(nb + txm128) * K[Kt] + txb128 * 4 + 0];
            rB.y = B[(nb + txm128) * K[Kt] + txb128 * 4 + 1];
            rB.z = rB.w = 0;
        }
        else if (txb128 * 4 + 1 <= K[Kt]) {
            rB.x = B[(nb + txm128) * K[Kt] + txb128 * 4 + 0];
            rB.y = rB.z = rB.w = 0;
        }
        else {
            rB.x = rB.y = rB.z = rB.w = 0;
        }
    }
    else {
        rB.x = rB.y = rB.z = rB.w = 0;
    }

    for (kb = 0; kb < K[Kt]; kb += 8) {

        (*psA)[txb128 * 4 + 0][txm128] = rA.x;
        (*psA)[txb128 * 4 + 1][txm128] = rA.y;
        (*psA)[txb128 * 4 + 2][txm128] = rA.z;
        (*psA)[txb128 * 4 + 3][txm128] = rA.w;

        (*psB)[txb128 * 4 + 0][txm128] = rB.x;
        (*psB)[txb128 * 4 + 1][txm128] = rB.y;
        (*psB)[txb128 * 4 + 2][txm128] = rB.z;
        (*psB)[txb128 * 4 + 3][txm128] = rB.w;

        __syncthreads();

        if (kb + 8 < K[Kt]) {
            if (mb + txm128 < M[Kt]) {
                rA.x = (p < cap) ? ((p < WC) ? A[hbb * WC + p++] : (0 * p++)) : ((hbb + (p - pbase) / (cap - pbase) < hcap) ? ((pbase + (p - pbase) % (cap - pbase) < WC) ? A[(hbb += (p - pbase) / (cap - pbase)) * WC + (p = pbase + (p - pbase) % (cap - pbase))++] : 0 * (hbb += (p - pbase) / (cap - pbase)) * (p = pbase + (p - pbase) % (cap - pbase))++) : 0);
                rA.y = (p < cap) ? ((p < WC) ? A[hbb * WC + p++] : (0 * p++)) : ((hbb + (p - pbase) / (cap - pbase) < hcap) ? ((pbase + (p - pbase) % (cap - pbase) < WC) ? A[(hbb += (p - pbase) / (cap - pbase)) * WC + (p = pbase + (p - pbase) % (cap - pbase))++] : 0 * (hbb += (p - pbase) / (cap - pbase)) * (p = pbase + (p - pbase) % (cap - pbase))++) : 0);
                rA.z = (p < cap) ? ((p < WC) ? A[hbb * WC + p++] : (0 * p++)) : ((hbb + (p - pbase) / (cap - pbase) < hcap) ? ((pbase + (p - pbase) % (cap - pbase) < WC) ? A[(hbb += (p - pbase) / (cap - pbase)) * WC + (p = pbase + (p - pbase) % (cap - pbase))++] : 0 * (hbb += (p - pbase) / (cap - pbase)) * (p = pbase + (p - pbase) % (cap - pbase))++) : 0);
                rA.w = (p < cap) ? ((p < WC) ? A[hbb * WC + p++] : (0 * p++)) : ((hbb + (p - pbase) / (cap - pbase) < hcap) ? ((pbase + (p - pbase) % (cap - pbase) < WC) ? A[(hbb += (p - pbase) / (cap - pbase)) * WC + (p = pbase + (p - pbase) % (cap - pbase))++] : 0 * (hbb += (p - pbase) / (cap - pbase)) * (p = pbase + (p - pbase) % (cap - pbase))++) : 0);
                p += 4;
            }
            else {
                rA.x = rA.y = rA.z = rA.w = 0;
            }

            if (nb + txm128 < N[Kt]) {
                if (kb + 8 + txb128 * 4 + 4 <= K[Kt]) {
                    rB.x = B[(nb + txm128) * K[Kt] + kb + 8 + txb128 * 4 + 0];
                    rB.y = B[(nb + txm128) * K[Kt] + kb + 8 + txb128 * 4 + 1];
                    rB.z = B[(nb + txm128) * K[Kt] + kb + 8 + txb128 * 4 + 2];
                    rB.w = B[(nb + txm128) * K[Kt] + kb + 8 + txb128 * 4 + 3];
                    //rB = *reinterpret_cast<float4*>(B + (nb + txm128) * K[Kt] + kb + 8 + txb128 * 4);
                }
                else if (kb + 8 + txb128 * 4 + 3 <= K[Kt]) {
                    rB.x = B[(nb + txm128) * K[Kt] + kb + 8 + txb128 * 4 + 0];
                    rB.y = B[(nb + txm128) * K[Kt] + kb + 8 + txb128 * 4 + 1];
                    rB.z = B[(nb + txm128) * K[Kt] + kb + 8 + txb128 * 4 + 2];
                    rB.w = 0;
                }
                else if (kb + 8 + txb128 * 4 + 2 <= K[Kt]) {
                    rB.x = B[(nb + txm128) * K[Kt] + kb + 8 + txb128 * 4 + 0];
                    rB.y = B[(nb + txm128) * K[Kt] + kb + 8 + txb128 * 4 + 1];
                    rB.z = rB.w = 0;
                }
                else if (kb + 8 + txb128 * 4 + 1 <= K[Kt]) {
                    rB.x = B[(nb + txm128) * K[Kt] + kb + 8 + txb128 * 4 + 0];
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

    int t1 = N[Kt] - nb - sBi1;
    int t2 = N[Kt] - nb - sBi2;

#pragma unroll
    for (m = 0; m < 8; m++) {
        if (mb + sAi1 + m < M[Kt]) {
            float4 t, z;
            t.x = rC[m][0];
            t.y = rC[m][1];
            t.z = rC[m][2];
            t.w = rC[m][3];
            if (4 <= t1) {
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi1 + 0] = t.x;
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi1 + 1] = t.y;
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi1 + 2] = t.z;
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi1 + 3] = t.w;
                //*reinterpret_cast<float4*>(C + (mb + sAi1 + m) * N[Kt] + nb + sBi1) = t;
            }
            else if (3 <= t1) {
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi1 + 0] = t.x;
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi1 + 1] = t.y;
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi1 + 2] = t.z;
            }
            else if (2 <= t1) {
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi1 + 0] = t.x;
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi1 + 1] = t.y;
            }
            else if (1 <= t1) {
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi1 + 0] = t.x;
            }
            z.x = rC[m][4];
            z.y = rC[m][5];
            z.z = rC[m][6];
            z.w = rC[m][7];
            if (4 <= t2) {
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi2 + 0] = z.x;
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi2 + 1] = z.y;
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi2 + 2] = z.z;
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi2 + 3] = z.w;
                //*reinterpret_cast<float4*>(C + (mb + sAi1 + m) * N[Kt] + nb + sBi2) = z;
            }
            else if (3 <= t2) {
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi2 + 0] = z.x;
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi2 + 1] = z.y;
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi2 + 2] = z.z;
            }
            else if (2 <= t2) {
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi2 + 0] = z.x;
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi2 + 1] = z.y;
            }
            else if (1 <= t2) {
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi2 + 0] = z.x;
            }
        }
    }

}

__global__ void conv_sgemm_32x32x8_f(float* A, float* B, float* C, int H, int W, int Ch, int* M, int* N, int* K, int* offsetsB, int* offsetsC, int* Ks32, int minK) {

    __shared__ __align__(16) float sA[8][32];
    __shared__ __align__(16) float sB[8][32];
    __shared__ __align__(16) float sAb[8][32];
    __shared__ __align__(16) float sBb[8][32];

    int tx = threadIdx.x;
    int mb = blockIdx.x * 32;
    int nb = blockIdx.y * 32;
    int Kt = Ks32[blockIdx.z];

    if (mb >= M[Kt] || nb >= N[Kt]) { return; }

    B += offsetsB[Kt];
    C += offsetsC[Kt];

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
    float4 rA;
    float4 rB;
    float rAs[4];
    float rBs[4];
    float rAsb[4];
    float rBsb[4];

    float(*psA)[8][32] = &sA;
    float(*psB)[8][32] = &sB;

    int WC = W * Ch;

    int m, n, k, kb;

    int wc = ceil((float)W / (Kt + minK));
    int wcap = wc * (Kt + minK) * Ch;
    int wb = ((mb + txm32) % wc) * (Kt + minK) * Ch;
    int hb = ((mb + txm32) / wc) * (Kt + minK);
    int wp = (txb32 * 4) % ((Kt + minK) * Ch);
    int hp = (txb32 * 4) / ((Kt + minK) * Ch);
    if (hp >= Kt + minK) { hp = H; }

    /*
    int m, n, k, kb, hbb, pbase, p;

    int wc = ceil((float)W / (Kt + minK));
    pbase = ((mb + txm32) % wc) * (Kt + minK) * Ch;
    p = pbase + txb32 * 4;
    hbb = ((mb + txm32) / wc) * (Kt + minK);
    int hcap = min(hbb + (Kt + minK), H);
    int cap = pbase + (Kt + minK) * Ch;
    */

#pragma unroll
    for (m = 0; m < 4; m++) {
#pragma unroll
        for (n = 0; n < 4; n++) {
            rC[m][n] = 0;
        }
    }

    if (hb + hp < H) {
        if (wb + wp < WC) {
            rA.x = A[0];
        }
        else {
            rA.x = 0;
        }
    }
    else {
        rA.x = 0;
    }

    rA.x = (hb + hp < H) ? ((wp < (Kt + minK)* Ch) ? ((wb + wp < WC) ? A[(hb + hp) * WC + wb + wp++] : 0 * wp++) : ((hb + hp + 1 < H && hp + 1 < Kt + minK) ? A[(hb + ++hp) * WC + wb + (wp = 0)++] : 0 * (hp = H))) : 0;
    rA.y = (hb + hp < H) ? ((wp < (Kt + minK)* Ch) ? ((wb + wp < WC) ? A[(hb + hp) * WC + wb + wp++] : 0 * wp++) : ((hb + hp + 1 < H && hp + 1 < Kt + minK) ? A[(hb + ++hp) * WC + wb + (wp = 0)++] : 0 * (hp = H))) : 0;
    rA.z = (hb + hp < H) ? ((wp < (Kt + minK)* Ch) ? ((wb + wp < WC) ? A[(hb + hp) * WC + wb + wp++] : 0 * wp++) : ((hb + hp + 1 < H && hp + 1 < Kt + minK) ? A[(hb + ++hp) * WC + wb + (wp = 0)++] : 0 * (hp = H))) : 0;
    rA.w = (hb + hp < H) ? ((wp < (Kt + minK)* Ch) ? ((wb + wp < WC) ? A[(hb + hp) * WC + wb + wp++] : 0 * wp++) : ((hb + hp + 1 < H && hp + 1 < Kt + minK) ? A[(hb + ++hp) * WC + wb + (wp = 0)++] : 0 * (hp = H))) : 0;

    wp += 4;
    if (wp >= (Kt + minK) * Ch) {
        hp += wp / ((Kt + minK) * Ch);
        wp = wp % ((Kt + minK) * Ch);
        if (hp >= Kt + minK) {
            hb = H;
        }
    }

    /*
    if (mb + txm32 < M[Kt]) {
        rA.x = (p < cap) ?
                    ((p < WC) ?
                        A[hbb * WC + p++] :
                        (0 * p++)) :
                    ((hbb + (p - pbase) / (cap - pbase) < hcap) ?
                        ((pbase + (p - pbase) % (cap - pbase) < WC) ?
                            A[(hbb += (p - pbase) / (cap - pbase)) * WC + (p = pbase + (p - pbase) % (cap - pbase))++] :
                            0 * (hbb += (p - pbase) / (cap - pbase)) * (p = pbase + (p - pbase) % (cap - pbase))++) :
                        0);
        rA.y = (p < cap) ? ((p < WC) ? A[hbb * WC + p++] : (0 * p++)) : ((hbb + (p - pbase) / (cap - pbase) < hcap) ? ((pbase + (p - pbase) % (cap - pbase) < WC) ? A[(hbb += (p - pbase) / (cap - pbase)) * WC + (p = pbase + (p - pbase) % (cap - pbase))++] : 0 * (hbb += (p - pbase) / (cap - pbase)) * (p = pbase + (p - pbase) % (cap - pbase))++) : 0);
        rA.z = (p < cap) ? ((p < WC) ? A[hbb * WC + p++] : (0 * p++)) : ((hbb + (p - pbase) / (cap - pbase) < hcap) ? ((pbase + (p - pbase) % (cap - pbase) < WC) ? A[(hbb += (p - pbase) / (cap - pbase)) * WC + (p = pbase + (p - pbase) % (cap - pbase))++] : 0 * (hbb += (p - pbase) / (cap - pbase)) * (p = pbase + (p - pbase) % (cap - pbase))++) : 0);
        rA.w = (p < cap) ? ((p < WC) ? A[hbb * WC + p++] : (0 * p++)) : ((hbb + (p - pbase) / (cap - pbase) < hcap) ? ((pbase + (p - pbase) % (cap - pbase) < WC) ? A[(hbb += (p - pbase) / (cap - pbase)) * WC + (p = pbase + (p - pbase) % (cap - pbase))++] : 0 * (hbb += (p - pbase) / (cap - pbase)) * (p = pbase + (p - pbase) % (cap - pbase))++) : 0);
        p += 4;
    }
    else {
        rA.x = rA.y = rA.z = rA.w = 0;
    }
    */

    if (nb + txm32 < N[Kt]) {
        if (txb32 * 4 + 4 <= K[Kt]) {
            rB.x = B[(nb + txm32) * K[Kt] + txb32 * 4 + 0];
            rB.y = B[(nb + txm32) * K[Kt] + txb32 * 4 + 1];
            rB.z = B[(nb + txm32) * K[Kt] + txb32 * 4 + 2];
            rB.w = B[(nb + txm32) * K[Kt] + txb32 * 4 + 3];
            //rB = *reinterpret_cast<float4*>(B + (nb + txm32) * K[Kt] + txb32 * 4);
        }
        else if (txb32 * 4 + 3 <= K[Kt]) {
            rB.x = B[(nb + txm32) * K[Kt] + txb32 * 4 + 0];
            rB.y = B[(nb + txm32) * K[Kt] + txb32 * 4 + 1];
            rB.z = B[(nb + txm32) * K[Kt] + txb32 * 4 + 2];
            rB.w = 0;
        }
        else if (txb32 * 4 + 2 <= K[Kt]) {
            rB.x = B[(nb + txm32) * K[Kt] + txb32 * 4 + 0];
            rB.y = B[(nb + txm32) * K[Kt] + txb32 * 4 + 1];
            rB.z = rB.w = 0;
        }
        else if (txb32 * 4 + 1 <= K[Kt]) {
            rB.x = B[(nb + txm32) * K[Kt] + txb32 * 4 + 0];
            rB.y = rB.z = rB.w = 0;
        }
        else {
            rB.x = rB.y = rB.z = rB.w = 0;
        }
    }
    else {
        rB.x = rB.y = rB.z = rB.w = 0;
    }

    for (kb = 0; kb < K[Kt]; kb += 8) {

        (*psA)[txb32 * 4 + 0][txm32] = rA.x;
        (*psA)[txb32 * 4 + 1][txm32] = rA.y;
        (*psA)[txb32 * 4 + 2][txm32] = rA.z;
        (*psA)[txb32 * 4 + 3][txm32] = rA.w;

        (*psB)[txb32 * 4 + 0][txm32] = rB.x;
        (*psB)[txb32 * 4 + 1][txm32] = rB.y;
        (*psB)[txb32 * 4 + 2][txm32] = rB.z;
        (*psB)[txb32 * 4 + 3][txm32] = rB.w;

        __syncthreads();

        if (kb + 8 < K[Kt]) {

            rA.x = (hb + hp < H) ? ((wp < (Kt + minK)* Ch) ? ((wb + wp < WC) ? A[(hb + hp) * WC + wb + wp++] : 0 * wp++) : ((hb + hp + 1 < H && hp + 1 < Kt + minK) ? A[(hb + ++hp) * WC + wb + (wp = 0)++] : 0 * (hp = H))) : 0;
            rA.y = (hb + hp < H) ? ((wp < (Kt + minK)* Ch) ? ((wb + wp < WC) ? A[(hb + hp) * WC + wb + wp++] : 0 * wp++) : ((hb + hp + 1 < H && hp + 1 < Kt + minK) ? A[(hb + ++hp) * WC + wb + (wp = 0)++] : 0 * (hp = H))) : 0;
            rA.z = (hb + hp < H) ? ((wp < (Kt + minK)* Ch) ? ((wb + wp < WC) ? A[(hb + hp) * WC + wb + wp++] : 0 * wp++) : ((hb + hp + 1 < H && hp + 1 < Kt + minK) ? A[(hb + ++hp) * WC + wb + (wp = 0)++] : 0 * (hp = H))) : 0;
            rA.w = (hb + hp < H) ? ((wp < (Kt + minK)* Ch) ? ((wb + wp < WC) ? A[(hb + hp) * WC + wb + wp++] : 0 * wp++) : ((hb + hp + 1 < H && hp + 1 < Kt + minK) ? A[(hb + ++hp) * WC + wb + (wp = 0)++] : 0 * (hp = H))) : 0;

            wp += 4;
            if (wp >= (Kt + minK) * Ch) {
                hp += wp / ((Kt + minK) * Ch);
                wp = wp % ((Kt + minK) * Ch);
                if (hp >= Kt + minK) {
                    hb = H;
                }
            }
            /*
            if (mb + txm32 < M[Kt]) {
                rA.x = (p < cap) ? ((p < WC) ? A[hbb * WC + p++] : (0 * p++)) : ((hbb + (p - pbase) / (cap - pbase) < hcap) ? ((pbase + (p - pbase) % (cap - pbase) < WC) ? A[(hbb += (p - pbase) / (cap - pbase)) * WC + (p = pbase + (p - pbase) % (cap - pbase))++] : 0 * (hbb += (p - pbase) / (cap - pbase)) * (p = pbase + (p - pbase) % (cap - pbase))++) : 0);
                rA.y = (p < cap) ? ((p < WC) ? A[hbb * WC + p++] : (0 * p++)) : ((hbb + (p - pbase) / (cap - pbase) < hcap) ? ((pbase + (p - pbase) % (cap - pbase) < WC) ? A[(hbb += (p - pbase) / (cap - pbase)) * WC + (p = pbase + (p - pbase) % (cap - pbase))++] : 0 * (hbb += (p - pbase) / (cap - pbase)) * (p = pbase + (p - pbase) % (cap - pbase))++) : 0);
                rA.z = (p < cap) ? ((p < WC) ? A[hbb * WC + p++] : (0 * p++)) : ((hbb + (p - pbase) / (cap - pbase) < hcap) ? ((pbase + (p - pbase) % (cap - pbase) < WC) ? A[(hbb += (p - pbase) / (cap - pbase)) * WC + (p = pbase + (p - pbase) % (cap - pbase))++] : 0 * (hbb += (p - pbase) / (cap - pbase)) * (p = pbase + (p - pbase) % (cap - pbase))++) : 0);
                rA.w = (p < cap) ? ((p < WC) ? A[hbb * WC + p++] : (0 * p++)) : ((hbb + (p - pbase) / (cap - pbase) < hcap) ? ((pbase + (p - pbase) % (cap - pbase) < WC) ? A[(hbb += (p - pbase) / (cap - pbase)) * WC + (p = pbase + (p - pbase) % (cap - pbase))++] : 0 * (hbb += (p - pbase) / (cap - pbase)) * (p = pbase + (p - pbase) % (cap - pbase))++) : 0);
                p += 4;
            }
            else {
                rA.x = rA.y = rA.z = rA.w = 0;
            }
            */

            if (nb + txm32 < N[Kt]) {
                if (kb + 8 + txb32 * 4 + 4 <= K[Kt]) {
                    rB.x = B[(nb + txm32) * K[Kt] + kb + 8 + txb32 * 4 + 0];
                    rB.y = B[(nb + txm32) * K[Kt] + kb + 8 + txb32 * 4 + 1];
                    rB.z = B[(nb + txm32) * K[Kt] + kb + 8 + txb32 * 4 + 2];
                    rB.w = B[(nb + txm32) * K[Kt] + kb + 8 + txb32 * 4 + 3];
                    //rB = *reinterpret_cast<float4*>(B + (nb + txm32) * K[Kt] + kb + 8 + txb32 * 4);
                }
                else if (kb + 8 + txb32 * 4 + 3 <= K[Kt]) {
                    rB.x = B[(nb + txm32) * K[Kt] + kb + 8 + txb32 * 4 + 0];
                    rB.y = B[(nb + txm32) * K[Kt] + kb + 8 + txb32 * 4 + 1];
                    rB.z = B[(nb + txm32) * K[Kt] + kb + 8 + txb32 * 4 + 2];
                    rB.w = 0;
                }
                else if (kb + 8 + txb32 * 4 + 2 <= K[Kt]) {
                    rB.x = B[(nb + txm32) * K[Kt] + kb + 8 + txb32 * 4 + 0];
                    rB.y = B[(nb + txm32) * K[Kt] + kb + 8 + txb32 * 4 + 1];
                    rB.z = rB.w = 0;
                }
                else if (kb + 8 + txb32 * 4 + 1 <= K[Kt]) {
                    rB.x = B[(nb + txm32) * K[Kt] + kb + 8 + txb32 * 4 + 0];
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
        for (k = 0; k < 8; k++) {

            if (k < 7) {
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


    int t1 = M[Kt] - mb - wmb - txm16b2 * 4;
    int t2 = N[Kt] - nb - wnb - txm32b16 * 8 - txm32m2 * 4;

#pragma unroll
    for (m = 0; m < 4; m++) {
        if (m < t1) {
            if (4 <= t2) {
                C[(mb + wmb + txm16b2 * 4 + m) * N[Kt] + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + 0] = rC[m][0];
                C[(mb + wmb + txm16b2 * 4 + m) * N[Kt] + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + 1] = rC[m][1];
                C[(mb + wmb + txm16b2 * 4 + m) * N[Kt] + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + 2] = rC[m][2];
                C[(mb + wmb + txm16b2 * 4 + m) * N[Kt] + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + 3] = rC[m][3];
                /*
                float4 t;
                t.x = rC[m][0];
                t.y = rC[m][1];
                t.z = rC[m][2];
                t.w = rC[m][3];
                *reinterpret_cast<float4*>(C + (mb + wmb + txm16b2 * 4 + m) * N[Kt] + nb + wnb + txm32b16 * 8 + txm32m2 * 4) = t;
                */
            }
            else if (3 <= t2) {
                C[(mb + wmb + txm16b2 * 4 + m) * N[Kt] + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + 0] = rC[m][0];
                C[(mb + wmb + txm16b2 * 4 + m) * N[Kt] + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + 1] = rC[m][1];
                C[(mb + wmb + txm16b2 * 4 + m) * N[Kt] + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + 2] = rC[m][2];
            }
            else if (2 <= t2) {
                C[(mb + wmb + txm16b2 * 4 + m) * N[Kt] + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + 0] = rC[m][0];
                C[(mb + wmb + txm16b2 * 4 + m) * N[Kt] + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + 1] = rC[m][1];
            }
            else if (1 <= t2) {
                C[(mb + wmb + txm16b2 * 4 + m) * N[Kt] + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + 0] = rC[m][0];
            }
        }
    }
}

__global__ void conv_sgemm_4x4x256_f(float* A, float* B, float* C, int H, int W, int Ch, int* M, int* N, int* K, int* offsetsB, int* offsetsC, int* Ks4, int minK) {

    __shared__ __align__(16) float sA[4][256];
    __shared__ __align__(16) float sB[4][256];
    __shared__ float sC[8][4][4];

    int tx = threadIdx.x;
    int mb = blockIdx.x * 4;
    int nb = blockIdx.y * 4;
    int Kt = Ks4[blockIdx.z];

    if (mb >= M[Kt] || nb >= N[Kt]) { return; }

    int txm64 = tx % 64;
    int txb64 = tx / 64;
    int txm32 = tx % 32;
    int txb32 = tx / 32;

    B += offsetsB[Kt];
    C += offsetsC[Kt];

    int WC = W * Ch;

    float4 rA, rB;
    float rC[4][4];

    int m, n, k, kb, hbb, pbase, p;

    int wc = ceil((float)W / (Kt + minK));
    pbase = ((mb + txb64) % wc) * (Kt + minK) * Ch;
    p = pbase + txm64 * 4;
    hbb = ((mb + txb64) / wc) * (Kt + minK);
    int hcap = min(hbb + (Kt + minK), H);
    int cap = (hbb < H) ? pbase + (Kt + minK) * Ch : 0 * (hcap = -99999999);

    int t1, t2;

    float(*psA)[4][256] = &sA;
    float(*psB)[4][256] = &sB;

    if (mb + txb64 < M[Kt]) {
        rA.x = (p < cap) ? ((p < WC) ? A[hbb * WC + p++] : (0 * p++)) : ((hbb + (p - pbase) / (cap - pbase) < hcap) ? ((pbase + (p - pbase) % (cap - pbase) < WC) ? A[(hbb += (p - pbase) / (cap - pbase)) * WC + (p = pbase + (p - pbase) % (cap - pbase))++] : 0 * (hbb += (p - pbase) / (cap - pbase)) * (p = pbase + (p - pbase) % (cap - pbase))++) : 0);
        rA.y = (p < cap) ? ((p < WC) ? A[hbb * WC + p++] : (0 * p++)) : ((hbb + (p - pbase) / (cap - pbase) < hcap) ? ((pbase + (p - pbase) % (cap - pbase) < WC) ? A[(hbb += (p - pbase) / (cap - pbase)) * WC + (p = pbase + (p - pbase) % (cap - pbase))++] : 0 * (hbb += (p - pbase) / (cap - pbase)) * (p = pbase + (p - pbase) % (cap - pbase))++) : 0);
        rA.z = (p < cap) ? ((p < WC) ? A[hbb * WC + p++] : (0 * p++)) : ((hbb + (p - pbase) / (cap - pbase) < hcap) ? ((pbase + (p - pbase) % (cap - pbase) < WC) ? A[(hbb += (p - pbase) / (cap - pbase)) * WC + (p = pbase + (p - pbase) % (cap - pbase))++] : 0 * (hbb += (p - pbase) / (cap - pbase)) * (p = pbase + (p - pbase) % (cap - pbase))++) : 0);
        rA.w = (p < cap) ? ((p < WC) ? A[hbb * WC + p++] : (0 * p++)) : ((hbb + (p - pbase) / (cap - pbase) < hcap) ? ((pbase + (p - pbase) % (cap - pbase) < WC) ? A[(hbb += (p - pbase) / (cap - pbase)) * WC + (p = pbase + (p - pbase) % (cap - pbase))++] : 0 * (hbb += (p - pbase) / (cap - pbase)) * (p = pbase + (p - pbase) % (cap - pbase))++) : 0);
        p += 252;
    }
    else {
        rA.x = rA.y = rA.z = rA.w = 0;
    }


    if (nb + txb64 < N[Kt]) {
        if (txm64 * 4 + 4 <= K[Kt]) {
            rB.x = B[(nb + txb64) * K[Kt] + txm64 * 4 + 0];
            rB.y = B[(nb + txb64) * K[Kt] + txm64 * 4 + 1];
            rB.z = B[(nb + txb64) * K[Kt] + txm64 * 4 + 2];
            rB.w = B[(nb + txb64) * K[Kt] + txm64 * 4 + 3];
            //rB = *reinterpret_cast<float4*>(B + (nb + txb64) * K[Kt] + txm64 * 4);
        }
        else if (txm64 * 4 + 3 <= K[Kt]) {
            rB.x = B[(nb + txb64) * K[Kt] + txm64 * 4 + 0];
            rB.y = B[(nb + txb64) * K[Kt] + txm64 * 4 + 1];
            rB.z = B[(nb + txb64) * K[Kt] + txm64 * 4 + 2];
            rB.w = 0;
        }
        else if (txm64 * 4 + 2 <= K[Kt]) {
            rB.x = B[(nb + txb64) * K[Kt] + txm64 * 4 + 0];
            rB.y = B[(nb + txb64) * K[Kt] + txm64 * 4 + 1];
            rB.z = rB.w = 0;
        }
        else if (txm64 * 4 + 1 <= K[Kt]) {
            rB.x = B[(nb + txb64) * K[Kt] + txm64 * 4 + 0];
            rB.y = rB.z = rB.w = 0;
        }
        else {
            rB.x = rB.y = rB.z = rB.w = 0;
        }
    }
    else {
        rB.x = rB.y = rB.z = rB.w = 0;
    }

    for (m = 0; m < 4; m++) {
        for (n = 0; n < 4; n++) {
            rC[m][n] = 0;
        }
    }

    for (kb = 0; kb < K[Kt]; kb += 256) {

        __syncthreads();

        (*psA)[txb64][txm64 * 4 + 0] = rA.x;
        (*psA)[txb64][txm64 * 4 + 1] = rA.y;
        (*psA)[txb64][txm64 * 4 + 2] = rA.z;
        (*psA)[txb64][txm64 * 4 + 3] = rA.w;

        (*psB)[txb64][txm64 * 4 + 0] = rB.x;
        (*psB)[txb64][txm64 * 4 + 1] = rB.y;
        (*psB)[txb64][txm64 * 4 + 2] = rB.z;
        (*psB)[txb64][txm64 * 4 + 3] = rB.w;

        __syncthreads();

        if (kb + 256 < K[Kt]) {
            if (mb + txb64 < M[Kt]) {
                rA.x = (p < cap) ? ((p < WC) ? A[hbb * WC + p++] : (0 * p++)) : ((hbb + (p - pbase) / (cap - pbase) < hcap) ? ((pbase + (p - pbase) % (cap - pbase) < WC) ? A[(hbb += (p - pbase) / (cap - pbase)) * WC + (p = pbase + (p - pbase) % (cap - pbase))++] : 0 * (hbb += (p - pbase) / (cap - pbase)) * (p = pbase + (p - pbase) % (cap - pbase))++) : 0);
                rA.y = (p < cap) ? ((p < WC) ? A[hbb * WC + p++] : (0 * p++)) : ((hbb + (p - pbase) / (cap - pbase) < hcap) ? ((pbase + (p - pbase) % (cap - pbase) < WC) ? A[(hbb += (p - pbase) / (cap - pbase)) * WC + (p = pbase + (p - pbase) % (cap - pbase))++] : 0 * (hbb += (p - pbase) / (cap - pbase)) * (p = pbase + (p - pbase) % (cap - pbase))++) : 0);
                rA.z = (p < cap) ? ((p < WC) ? A[hbb * WC + p++] : (0 * p++)) : ((hbb + (p - pbase) / (cap - pbase) < hcap) ? ((pbase + (p - pbase) % (cap - pbase) < WC) ? A[(hbb += (p - pbase) / (cap - pbase)) * WC + (p = pbase + (p - pbase) % (cap - pbase))++] : 0 * (hbb += (p - pbase) / (cap - pbase)) * (p = pbase + (p - pbase) % (cap - pbase))++) : 0);
                rA.w = (p < cap) ? ((p < WC) ? A[hbb * WC + p++] : (0 * p++)) : ((hbb + (p - pbase) / (cap - pbase) < hcap) ? ((pbase + (p - pbase) % (cap - pbase) < WC) ? A[(hbb += (p - pbase) / (cap - pbase)) * WC + (p = pbase + (p - pbase) % (cap - pbase))++] : 0 * (hbb += (p - pbase) / (cap - pbase)) * (p = pbase + (p - pbase) % (cap - pbase))++) : 0);
                p += 252;
            }
            else {
                rA.x = rA.y = rA.z = rA.w = 0;
            }

            if (nb + txb64 < N[Kt]) {
                if (kb + 256 + txm64 * 4 + 4 <= K[Kt]) {
                    rB.x = B[(nb + txb64) * K[Kt] + kb + 256 + txm64 * 4 + 0];
                    rB.y = B[(nb + txb64) * K[Kt] + kb + 256 + txm64 * 4 + 1];
                    rB.z = B[(nb + txb64) * K[Kt] + kb + 256 + txm64 * 4 + 2];
                    rB.w = B[(nb + txb64) * K[Kt] + kb + 256 + txm64 * 4 + 3];
                    //rB = *reinterpret_cast<float4*>(B + (nb + txb64) * K[Kt] + kb + 256 + txm64 * 4);
                }
                else if (kb + 256 + txm64 * 4 + 3 <= K[Kt]) {
                    rB.x = B[(nb + txb64) * K[Kt] + kb + 256 + txm64 * 4 + 0];
                    rB.y = B[(nb + txb64) * K[Kt] + kb + 256 + txm64 * 4 + 1];
                    rB.z = B[(nb + txb64) * K[Kt] + kb + 256 + txm64 * 4 + 2];
                    rB.w = 0;
                }
                else if (kb + 256 + txm64 * 4 + 2 <= K[Kt]) {
                    rB.x = B[(nb + txb64) * K[Kt] + kb + 256 + txm64 * 4 + 0];
                    rB.y = B[(nb + txb64) * K[Kt] + kb + 256 + txm64 * 4 + 1];
                    rB.z = rB.w = 0;
                }
                else if (kb + 256 + txm64 * 4 + 1 <= K[Kt]) {
                    rB.x = B[(nb + txb64) * K[Kt] + kb + 256 + txm64 * 4 + 0];
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

        // COMPUTE

#pragma unroll
        for (m = 0; m < 4; m++) {
#pragma unroll
            for (n = 0; n < 4; n++) {
                rC[m][n] += (*psA)[m][tx] * (*psB)[n][tx];
            }
        }

    }

#pragma unroll
    for (m = 0; m < 4; m++) {
#pragma unroll
        for (n = 0; n < 4; n++) {
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
            if (txm32 == 0) {
                sC[txb32][m][n] = rC[m][n];
            }
        }
    }

    __syncthreads();

    if (tx == 0) {
#pragma unroll
        for (m = 0; m < 4; m++) {
            if (mb + m < M[Kt]) {
                if (nb + 4 <= N[Kt]) {
                    C[(mb + m) * N[Kt] + nb + 0] = rC[m][0] + sC[1][m][0] + sC[2][m][0] + sC[3][m][0] + sC[4][m][0] + sC[5][m][0] + sC[6][m][0] + sC[7][m][0];
                    C[(mb + m) * N[Kt] + nb + 1] = rC[m][1] + sC[1][m][1] + sC[2][m][1] + sC[3][m][1] + sC[4][m][1] + sC[5][m][1] + sC[6][m][1] + sC[7][m][1];
                    C[(mb + m) * N[Kt] + nb + 2] = rC[m][2] + sC[1][m][2] + sC[2][m][2] + sC[3][m][2] + sC[4][m][2] + sC[5][m][2] + sC[6][m][2] + sC[7][m][2];
                    C[(mb + m) * N[Kt] + nb + 3] = rC[m][3] + sC[1][m][3] + sC[2][m][3] + sC[3][m][3] + sC[4][m][3] + sC[5][m][3] + sC[6][m][3] + sC[7][m][3];
                    /*
                    rA.x = rC[m][0] + sC[1][m][0] + sC[2][m][0] + sC[3][m][0] + sC[4][m][0] + sC[5][m][0] + sC[6][m][0] + sC[7][m][0];
                    rA.y = rC[m][1] + sC[1][m][1] + sC[2][m][1] + sC[3][m][1] + sC[4][m][1] + sC[5][m][1] + sC[6][m][1] + sC[7][m][1];
                    rA.z = rC[m][2] + sC[1][m][2] + sC[2][m][2] + sC[3][m][2] + sC[4][m][2] + sC[5][m][2] + sC[6][m][2] + sC[7][m][2];
                    rA.w = rC[m][3] + sC[1][m][3] + sC[2][m][3] + sC[3][m][3] + sC[4][m][3] + sC[5][m][3] + sC[6][m][3] + sC[7][m][3];
                    *reinterpret_cast<float4*>(C + (mb + m) * N[Kt] + nb) = rA;
                    */
                }
                else if (nb + 3 <= N[Kt]) {
                    C[(mb + m) * N[Kt] + nb + 0] = rC[m][0] + sC[1][m][0] + sC[2][m][0] + sC[3][m][0] + sC[4][m][0] + sC[5][m][0] + sC[6][m][0] + sC[7][m][0];
                    C[(mb + m) * N[Kt] + nb + 1] = rC[m][1] + sC[1][m][1] + sC[2][m][1] + sC[3][m][1] + sC[4][m][1] + sC[5][m][1] + sC[6][m][1] + sC[7][m][1];
                    C[(mb + m) * N[Kt] + nb + 2] = rC[m][2] + sC[1][m][2] + sC[2][m][2] + sC[3][m][2] + sC[4][m][2] + sC[5][m][2] + sC[6][m][2] + sC[7][m][2];
                }
                else if (nb + 2 <= N[Kt]) {
                    C[(mb + m) * N[Kt] + nb + 0] = rC[m][0] + sC[1][m][0] + sC[2][m][0] + sC[3][m][0] + sC[4][m][0] + sC[5][m][0] + sC[6][m][0] + sC[7][m][0];
                    C[(mb + m) * N[Kt] + nb + 1] = rC[m][1] + sC[1][m][1] + sC[2][m][1] + sC[3][m][1] + sC[4][m][1] + sC[5][m][1] + sC[6][m][1] + sC[7][m][1];
                }
                else if (nb + 1 <= N[Kt]) {
                    C[(mb + m) * N[Kt] + nb + 0] = rC[m][0] + sC[1][m][0] + sC[2][m][0] + sC[3][m][0] + sC[4][m][0] + sC[5][m][0] + sC[6][m][0] + sC[7][m][0];
                }
            }
        }
    }

}


__global__ void conv_sgemm_128x128x8_k(float* A, float* B, float* C, int H, int W, int Ch, int* M, int* N, int* K, int* offsetsA, int* offsetsC, int* Ks128, int minK) {


    __shared__ __align__(16) float sA[8][132];
    __shared__ __align__(16) float sB[8][132];
    __shared__ __align__(16) float sAb[8][132];
    __shared__ __align__(16) float sBb[8][132];

    int tx = threadIdx.x;
    int mb = blockIdx.x * 128;
    int nb = blockIdx.y * 128;
    int Kt = Ks128[blockIdx.z];

    if (mb >= M[Kt] || nb >= N[Kt]) { return; }

    A += offsetsA[Kt];
    C += offsetsC[Kt];

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

    int WC = W * Ch;

    int m, n, k, kb, hb, wb, hp, wp, wcap, wpb, hpb, p;

    int wc = ceil((float)W / (Kt + minK));
    wcap = wc * (Kt + minK) * Ch;
    hb = (txb32 / wc) * (Kt + minK);
    wb = (txb32 % wc) * (Kt + minK) * Ch;
    hp = (nb + txm32 * 4) / ((Kt + minK) * Ch);
    wp = (nb + txm32 * 4) % ((Kt + minK) * Ch);
    wpb = wp;
    hpb = hp;
    if (hp >= Kt + minK) {
        hb = H;
    }

    float4 f4;
    float8 f8;

    int sAi1 = wmb + (txm32 % 2) * 8 + (txm32 / 16) * 16;
    int sBi1 = wnb + (txm32 / 2) * 4;
    int sBi2 = wnb + (32 + (txm32 / 2) * 4) % 64;

#pragma unroll
    for (m = 0; m < 8; m++) {
        if (mb + sAi1 + m < M[Kt]) {
            int t1 = N[Kt] - nb - sBi1;
            int t2 = N[Kt] - nb - sBi2;
            if (4 <= t1) {
                rC[m][0] = C[(mb + sAi1 + m) * N[Kt] + nb + sBi1 + 0];
                rC[m][1] = C[(mb + sAi1 + m) * N[Kt] + nb + sBi1 + 1];
                rC[m][2] = C[(mb + sAi1 + m) * N[Kt] + nb + sBi1 + 2];
                rC[m][3] = C[(mb + sAi1 + m) * N[Kt] + nb + sBi1 + 3];
            }
            else if (3 <= t1) {
                rC[m][0] = C[(mb + sAi1 + m) * N[Kt] + nb + sBi1 + 0];
                rC[m][1] = C[(mb + sAi1 + m) * N[Kt] + nb + sBi1 + 1];
                rC[m][2] = C[(mb + sAi1 + m) * N[Kt] + nb + sBi1 + 2];
                rC[m][3] = 0;
            }
            else if (2 <= t1) {
                rC[m][0] = C[(mb + sAi1 + m) * N[Kt] + nb + sBi1 + 0];
                rC[m][1] = C[(mb + sAi1 + m) * N[Kt] + nb + sBi1 + 1];
                rC[m][2] = rC[m][3] = 0;
            }
            else if (1 <= t1) {
                rC[m][0] = C[(mb + sAi1 + m) * N[Kt] + nb + sBi1 + 0];
                rC[m][1] = rC[m][2] = rC[m][3] = 0;
            }
            else {
                rC[m][0] = rC[m][1] = rC[m][2] = rC[m][3] = 0;                
            }
            if (4 <= t2) {
                rC[m][4] = C[(mb + sAi1 + m) * N[Kt] + nb + sBi2 + 0];
                rC[m][5] = C[(mb + sAi1 + m) * N[Kt] + nb + sBi2 + 1];
                rC[m][6] = C[(mb + sAi1 + m) * N[Kt] + nb + sBi2 + 2];
                rC[m][7] = C[(mb + sAi1 + m) * N[Kt] + nb + sBi2 + 3];
            }
            else if (3 <= t2) {
                rC[m][4] = C[(mb + sAi1 + m) * N[Kt] + nb + sBi2 + 0];
                rC[m][5] = C[(mb + sAi1 + m) * N[Kt] + nb + sBi2 + 1];
                rC[m][6] = C[(mb + sAi1 + m) * N[Kt] + nb + sBi2 + 2];
                rC[m][7] = 0;
            }
            else if (2 <= t2) {
                rC[m][4] = C[(mb + sAi1 + m) * N[Kt] + nb + sBi2 + 0];
                rC[m][5] = C[(mb + sAi1 + m) * N[Kt] + nb + sBi2 + 1];
                rC[m][6] = rC[m][7] = 0;
            }
            else if (1 <= t2) {
                rC[m][4] = C[(mb + sAi1 + m) * N[Kt] + nb + sBi2 + 0];
                rC[m][5] = rC[m][6] = rC[m][7] = 0;
            }
            else {
                rC[m][4] = rC[m][5] = rC[m][6] = rC[m][7] = 0;
            }
        }
        else {
            rC[m][0] = rC[m][1] = rC[m][2] = rC[m][3] = rC[m][4] = rC[m][5] = rC[m][6] = rC[m][7] = 0;
        }
    }

    int t1, t2;

    if (txb32 < K[Kt]) {
        t1 = M[Kt] - mb - txm32 * 4;
        if (4 <= t1) {
            rA.x = A[txb32 * M[Kt] + mb + txm32 * 4 + 0];
            rA.y = A[txb32 * M[Kt] + mb + txm32 * 4 + 1];
            rA.z = A[txb32 * M[Kt] + mb + txm32 * 4 + 2];
            rA.w = A[txb32 * M[Kt] + mb + txm32 * 4 + 3];
        }
        else if (3 <= t1) {
            rA.x = A[txb32 * M[Kt] + mb + txm32 * 4 + 0];
            rA.y = A[txb32 * M[Kt] + mb + txm32 * 4 + 1];
            rA.z = A[txb32 * M[Kt] + mb + txm32 * 4 + 2];
            rA.w = 0;
        }
        else if (2 <= t1) {
            rA.x = A[txb32 * M[Kt] + mb + txm32 * 4 + 0];
            rA.y = A[txb32 * M[Kt] + mb + txm32 * 4 + 1];
            rA.z = rA.w = 0;
        }
        else if (1 <= t1) {
            rA.x = A[txb32 * M[Kt] + mb + txm32 * 4 + 0];
            rA.y = rA.z = rA.w = 0;
        }
        else {
            rA.x = rA.y = rA.z = rA.w = 0;
        }
    }
    else {
        rA.x = rA.y = rA.z = rA.w = 0;
    }


    rB.x = (hb + hp < H) ? ((wp < (Kt + minK)* Ch) ? ((wb + wp < WC) ? B[(hb + hp) * WC + (wb + wp++)] : 0 * wp++) : ((hb + hp + 1 < H && hp + 1 < Kt + minK) ? B[(hb + ++hp) * WC + wb + (wp = 0)++] : 0)) : 0;
    rB.y = (hb + hp < H) ? ((wp < (Kt + minK)* Ch) ? ((wb + wp < WC) ? B[(hb + hp) * WC + (wb + wp++)] : 0 * wp++) : ((hb + hp + 1 < H && hp + 1 < Kt + minK) ? B[(hb + ++hp) * WC + wb + (wp = 0)++] : 0)) : 0;
    rB.z = (hb + hp < H) ? ((wp < (Kt + minK)* Ch) ? ((wb + wp < WC) ? B[(hb + hp) * WC + (wb + wp++)] : 0 * wp++) : ((hb + hp + 1 < H && hp + 1 < Kt + minK) ? B[(hb + ++hp) * WC + wb + (wp = 0)++] : 0)) : 0;
    rB.w = (hb + hp < H) ? ((wp < (Kt + minK)* Ch) ? ((wb + wp < WC) ? B[(hb + hp) * WC + (wb + wp++)] : 0 * wp++) : ((hb + hp + 1 < H && hp + 1 < Kt + minK) ? B[(hb + ++hp) * WC + wb + (wp = 0)++] : 0)) : 0;
    
    wp = wpb;
    hp = hpb;
    wb += 8 * (Kt + minK) * Ch;
    if (wb >= wcap) {
        hb += (wb / wcap) * (Kt + minK);
        wb = wb % wcap;
    }

    for (kb = 0; kb < K[Kt]; kb += 8) {

        (*psA)[txb32][txm32 * 4 + 0] = rA.x;
        (*psA)[txb32][txm32 * 4 + 1] = rA.y;
        (*psA)[txb32][txm32 * 4 + 2] = rA.z;
        (*psA)[txb32][txm32 * 4 + 3] = rA.w;

        (*psB)[txb32][txm32 * 4 + 0] = rB.x;
        (*psB)[txb32][txm32 * 4 + 1] = rB.y;
        (*psB)[txb32][txm32 * 4 + 2] = rB.z;
        (*psB)[txb32][txm32 * 4 + 3] = rB.w;

        __syncthreads();

        if (kb + 8 < K[Kt]) {

            if (txb32 + kb + 8 < K[Kt]) {
                t1 = M[Kt] - mb - txm32 * 4;
                if (4 <= t1) {
                    rA.x = A[(kb + 8 + txb32) * M[Kt] + mb + txm32 * 4 + 0];
                    rA.y = A[(kb + 8 + txb32) * M[Kt] + mb + txm32 * 4 + 1];
                    rA.z = A[(kb + 8 + txb32) * M[Kt] + mb + txm32 * 4 + 2];
                    rA.w = A[(kb + 8 + txb32) * M[Kt] + mb + txm32 * 4 + 3];
                }
                else if (3 <= t1) {
                    rA.x = A[(kb + 8 + txb32) * M[Kt] + mb + txm32 * 4 + 0];
                    rA.y = A[(kb + 8 + txb32) * M[Kt] + mb + txm32 * 4 + 1];
                    rA.z = A[(kb + 8 + txb32) * M[Kt] + mb + txm32 * 4 + 2];
                    rA.w = 0;
                }
                else if (2 <= t1) {
                    rA.x = A[(kb + 8 + txb32) * M[Kt] + mb + txm32 * 4 + 0];
                    rA.y = A[(kb + 8 + txb32) * M[Kt] + mb + txm32 * 4 + 1];
                    rA.z = rA.w = 0;
                }
                else if (1 <= t1) {
                    rA.x = A[(kb + 8 + txb32) * M[Kt] + mb + txm32 * 4 + 0];
                    rA.y = rA.z = rA.w = 0;
                }
                else {
                    rA.x = rA.y = rA.z = rA.w = 0;
                }
            }
            else {
                rA.x = rA.y = rA.z = rA.w = 0;
            }

            rB.x = (hb + hp < H) ? ((wp < (Kt + minK)* Ch) ? ((wb + wp < WC) ? B[(hb + hp) * WC + (wb + wp++)] : 0 * wp++) : ((hb + hp + 1 < H && hp + 1 < Kt + minK) ? B[(hb + ++hp) * WC + wb + (wp = 0)++] : 0)) : 0;
            rB.y = (hb + hp < H) ? ((wp < (Kt + minK)* Ch) ? ((wb + wp < WC) ? B[(hb + hp) * WC + (wb + wp++)] : 0 * wp++) : ((hb + hp + 1 < H && hp + 1 < Kt + minK) ? B[(hb + ++hp) * WC + wb + (wp = 0)++] : 0)) : 0;
            rB.z = (hb + hp < H) ? ((wp < (Kt + minK)* Ch) ? ((wb + wp < WC) ? B[(hb + hp) * WC + (wb + wp++)] : 0 * wp++) : ((hb + hp + 1 < H && hp + 1 < Kt + minK) ? B[(hb + ++hp) * WC + wb + (wp = 0)++] : 0)) : 0;
            rB.w = (hb + hp < H) ? ((wp < (Kt + minK)* Ch) ? ((wb + wp < WC) ? B[(hb + hp) * WC + (wb + wp++)] : 0 * wp++) : ((hb + hp + 1 < H && hp + 1 < Kt + minK) ? B[(hb + ++hp) * WC + wb + (wp = 0)++] : 0)) : 0;

            wp = wpb;
            hp = hpb;
            wb += 8 * (Kt + minK) * Ch;
            if (wb >= wcap) {
                hb += (wb / wcap) * (Kt + minK);
                wb = wb % wcap;
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

    t1 = N[Kt] - nb - sBi1;
    t2 = N[Kt] - nb - sBi2;

#pragma unroll
    for (m = 0; m < 8; m++) {
        if (mb + sAi1 + m < M[Kt]) {
            float4 t, z;
            t.x = rC[m][0];
            t.y = rC[m][1];
            t.z = rC[m][2];
            t.w = rC[m][3];
            if (4 <= t1) {
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi1 + 0] = t.x;
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi1 + 1] = t.y;
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi1 + 2] = t.z;
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi1 + 3] = t.w;
                //*reinterpret_cast<float4*>(C + (mb + sAi1 + m) * N[Kt] + nb + sBi1) = t;
            }
            else if (3 <= t1) {
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi1 + 0] = t.x;
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi1 + 1] = t.y;
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi1 + 2] = t.z;
            }
            else if (2 <= t1) {
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi1 + 0] = t.x;
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi1 + 1] = t.y;
            }
            else if (1 <= t1) {
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi1 + 0] = t.x;
            }
            z.x = rC[m][4];
            z.y = rC[m][5];
            z.z = rC[m][6];
            z.w = rC[m][7];
            if (4 <= t2) {
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi2 + 0] = z.x;
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi2 + 1] = z.y;
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi2 + 2] = z.z;
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi2 + 3] = z.w;
                //*reinterpret_cast<float4*>(C + (mb + sAi1 + m) * N[Kt] + nb + sBi2) = z;
            }
            else if (3 <= t2) {
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi2 + 0] = z.x;
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi2 + 1] = z.y;
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi2 + 2] = z.z;
            }
            else if (2 <= t2) {
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi2 + 0] = z.x;
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi2 + 1] = z.y;
            }
            else if (1 <= t2) {
                C[(mb + sAi1 + m) * N[Kt] + nb + sBi2 + 0] = z.x;
            }
        }
    }

}

__global__ void conv_sgemm_32x32x8_k(float* A, float* B, float* C, int H, int W, int Ch, int* M, int* N, int* K, int* offsetsA, int* offsetsC, int* Ks32, int minK) {

    __shared__ __align__(16) float sA[8][32];
    __shared__ __align__(16) float sB[8][32];
    __shared__ __align__(16) float sAb[8][32];
    __shared__ __align__(16) float sBb[8][32];

    int tx = threadIdx.x;
    int mb = blockIdx.x * 32;
    int nb = blockIdx.y * 32;
    int Kt = Ks32[blockIdx.z];

    if (mb >= M[Kt] || nb >= N[Kt]) { return; }

    A += offsetsA[Kt];
    C += offsetsC[Kt];

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
    float4 rA;
    float4 rB;
    float rAs[4];
    float rBs[4];
    float rAsb[4];
    float rBsb[4];

    float(*psA)[8][32] = &sA;
    float(*psB)[8][32] = &sB;

    int WC = W * Ch;

    int m, n, k, kb, hb, wb, hp, wp, wcap, wpb, hpb, p;

    int wc = ceil((float)W / (Kt + minK));
    wcap = wc * (Kt + minK) * Ch;
    hb = (txb8 / wc) * (Kt + minK);
    wb = (txb8 % wc) * (Kt + minK) * Ch;
    hp = (nb + txm8 * 4) / ((Kt + minK) * Ch);
    wp = (nb + txm8 * 4) % ((Kt + minK) * Ch);
    wpb = wp;
    hpb = hp;
    if (hp >= Kt + minK) {
        hb = H;
    }

    int t1 = M[Kt] - mb - wmb - txm16b2 * 4;
    int t2 = N[Kt] - nb - wnb - txm32b16 * 8 - txm32m2 * 4;

#pragma unroll
    for (m = 0; m < 4; m++) {
        if (m < t1) {
            if (4 <= t2) {
                rC[m][0] = C[(mb + wmb + txm16b2 * 4 + m) * N[Kt] + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + 0];
                rC[m][1] = C[(mb + wmb + txm16b2 * 4 + m) * N[Kt] + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + 1];
                rC[m][2] = C[(mb + wmb + txm16b2 * 4 + m) * N[Kt] + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + 2];
                rC[m][3] = C[(mb + wmb + txm16b2 * 4 + m) * N[Kt] + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + 3];
            }
            else if (3 <= t2) {
                rC[m][0] = C[(mb + wmb + txm16b2 * 4 + m) * N[Kt] + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + 0];
                rC[m][1] = C[(mb + wmb + txm16b2 * 4 + m) * N[Kt] + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + 1];
                rC[m][2] = C[(mb + wmb + txm16b2 * 4 + m) * N[Kt] + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + 2];
                rC[m][3] = 0;
            }
            else if (2 <= t2) {
                rC[m][0] = C[(mb + wmb + txm16b2 * 4 + m) * N[Kt] + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + 0];
                rC[m][1] = C[(mb + wmb + txm16b2 * 4 + m) * N[Kt] + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + 1];
                rC[m][2] = rC[m][3] = 0;
            }
            else if (1 <= t2) {
                rC[m][0] = C[(mb + wmb + txm16b2 * 4 + m) * N[Kt] + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + 0];
                rC[m][1] = rC[m][2] = rC[m][3] = 0;
            }
            else {
                rC[m][0] = rC[m][1] = rC[m][2] = rC[m][3] = 0;
            }
        }
        else {
            rC[m][0] = rC[m][1] = rC[m][2] = rC[m][3] = 0;
        }
    }

    if (txb8 < K[Kt]) {
        t1 = M[Kt] - mb - txm8 * 4;
        if (4 <= t1) {
            rA.x = A[txb8 * M[Kt] + mb + txm8 * 4 + 0];
            rA.y = A[txb8 * M[Kt] + mb + txm8 * 4 + 1];
            rA.z = A[txb8 * M[Kt] + mb + txm8 * 4 + 2];
            rA.w = A[txb8 * M[Kt] + mb + txm8 * 4 + 3];
            //rA = *reinterpret_cast<float4*>(A + txb8 * M[Kt] + mb + txm8 * 4);
        }
        else if (3 <= t1) {
            rA.x = A[txb8 * M[Kt] + mb + txm8 * 4 + 0];
            rA.y = A[txb8 * M[Kt] + mb + txm8 * 4 + 1];
            rA.z = A[txb8 * M[Kt] + mb + txm8 * 4 + 2];
            rA.w = 0;
        }
        else if (2 <= t1) {
            rA.x = A[txb8 * M[Kt] + mb + txm8 * 4 + 0];
            rA.y = A[txb8 * M[Kt] + mb + txm8 * 4 + 1];
            rA.z = rA.w = 0;
        }
        else if (1 <= t1) {
            rA.x = A[txb8 * M[Kt] + mb + txm8 * 4 + 0];
            rA.y = rA.z = rA.w = 0;
        }
        else {
            rA.x = rA.y = rA.z = rA.w = 0;
        }
    }
    else {
        rA.x = rA.y = rA.z = rA.w = 0;
    }

    rB.x = (hb + hp < H) ? ((wp < (Kt + minK)* Ch) ? ((wb + wp < WC) ? B[(hb + hp) * WC + (wb + wp++)] : 0 * wp++) : ((hb + hp + 1 < H && hp + 1 < Kt + minK) ? B[(hb + ++hp) * WC + wb + (wp = 0)++] : 0)) : 0;
    rB.y = (hb + hp < H) ? ((wp < (Kt + minK)* Ch) ? ((wb + wp < WC) ? B[(hb + hp) * WC + (wb + wp++)] : 0 * wp++) : ((hb + hp + 1 < H && hp + 1 < Kt + minK) ? B[(hb + ++hp) * WC + wb + (wp = 0)++] : 0)) : 0;
    rB.z = (hb + hp < H) ? ((wp < (Kt + minK)* Ch) ? ((wb + wp < WC) ? B[(hb + hp) * WC + (wb + wp++)] : 0 * wp++) : ((hb + hp + 1 < H && hp + 1 < Kt + minK) ? B[(hb + ++hp) * WC + wb + (wp = 0)++] : 0)) : 0;
    rB.w = (hb + hp < H) ? ((wp < (Kt + minK)* Ch) ? ((wb + wp < WC) ? B[(hb + hp) * WC + (wb + wp++)] : 0 * wp++) : ((hb + hp + 1 < H && hp + 1 < Kt + minK) ? B[(hb + ++hp) * WC + wb + (wp = 0)++] : 0)) : 0;

    wp = wpb;
    hp = hpb;
    wb += 8 * (Kt + minK) * Ch;
    if (wb >= wcap) {
        hb += (wb / wcap) * (Kt + minK);
        wb = wb % wcap;
    }

    for (kb = 0; kb < K[Kt]; kb += 8) {

        (*psA)[txb8][txm8 * 4 + 0] = rA.x;
        (*psA)[txb8][txm8 * 4 + 1] = rA.y;
        (*psA)[txb8][txm8 * 4 + 2] = rA.z;
        (*psA)[txb8][txm8 * 4 + 3] = rA.w;

        (*psB)[txb8][txm8 * 4 + 0] = rB.x;
        (*psB)[txb8][txm8 * 4 + 1] = rB.y;
        (*psB)[txb8][txm8 * 4 + 2] = rB.z;
        (*psB)[txb8][txm8 * 4 + 3] = rB.w;

        __syncthreads();

        if (kb + 8 < K[Kt]) {
            if (kb + 8 + txb8 < K[Kt]) {
                t1 = M[Kt] - mb - txm8 * 4;
                if (4 <= t1) {
                    rA.x = A[(kb + 8 + txb8) * M[Kt] + mb + txm8 * 4 + 0];
                    rA.y = A[(kb + 8 + txb8) * M[Kt] + mb + txm8 * 4 + 1];
                    rA.z = A[(kb + 8 + txb8) * M[Kt] + mb + txm8 * 4 + 2];
                    rA.w = A[(kb + 8 + txb8) * M[Kt] + mb + txm8 * 4 + 3];
                    //rA = *reinterpret_cast<float4*>(A + (kb + 8 + txb8) * M[Kt] + mb + txm8 * 4);
                }
                else if (3 <= t1) {
                    rA.x = A[(kb + 8 + txb8) * M[Kt] + mb + txm8 * 4 + 0];
                    rA.y = A[(kb + 8 + txb8) * M[Kt] + mb + txm8 * 4 + 1];
                    rA.z = A[(kb + 8 + txb8) * M[Kt] + mb + txm8 * 4 + 2];
                    rA.w = 0;
                }
                else if (2 <= t1) {
                    rA.x = A[(kb + 8 + txb8) * M[Kt] + mb + txm8 * 4 + 0];
                    rA.y = A[(kb + 8 + txb8) * M[Kt] + mb + txm8 * 4 + 1];
                    rA.z = rA.w = 0;
                }
                else if (1 <= t1) {
                    rA.x = A[(kb + 8 + txb8) * M[Kt] + mb + txm8 * 4 + 0];
                    rA.y = rA.z = rA.w = 0;
                }
                else {
                    rA.x = rA.y = rA.z = rA.w = 0;
                }
            }
            else {
                rA.x = rA.y = rA.z = rA.w = 0;
            }

            rB.x = (hb + hp < H) ? ((wp < (Kt + minK)* Ch) ? ((wb + wp < WC) ? B[(hb + hp) * WC + (wb + wp++)] : 0 * wp++) : ((hb + hp + 1 < H && hp + 1 < Kt + minK) ? B[(hb + ++hp) * WC + wb + (wp = 0)++] : 0)) : 0;
            rB.y = (hb + hp < H) ? ((wp < (Kt + minK)* Ch) ? ((wb + wp < WC) ? B[(hb + hp) * WC + (wb + wp++)] : 0 * wp++) : ((hb + hp + 1 < H && hp + 1 < Kt + minK) ? B[(hb + ++hp) * WC + wb + (wp = 0)++] : 0)) : 0;
            rB.z = (hb + hp < H) ? ((wp < (Kt + minK)* Ch) ? ((wb + wp < WC) ? B[(hb + hp) * WC + (wb + wp++)] : 0 * wp++) : ((hb + hp + 1 < H && hp + 1 < Kt + minK) ? B[(hb + ++hp) * WC + wb + (wp = 0)++] : 0)) : 0;
            rB.w = (hb + hp < H) ? ((wp < (Kt + minK)* Ch) ? ((wb + wp < WC) ? B[(hb + hp) * WC + (wb + wp++)] : 0 * wp++) : ((hb + hp + 1 < H && hp + 1 < Kt + minK) ? B[(hb + ++hp) * WC + wb + (wp = 0)++] : 0)) : 0;

            wp = wpb;
            hp = hpb;
            wb += 8 * (Kt + minK) * Ch;
            if (wb >= wcap) {
                hb += (wb / wcap) * (Kt + minK);
                wb = wb % wcap;
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
        for (k = 0; k < 8; k++) {

            if (k < 7) {
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


    t1 = M[Kt] - mb - wmb - txm16b2 * 4;
    t2 = N[Kt] - nb - wnb - txm32b16 * 8 - txm32m2 * 4;

#pragma unroll
    for (m = 0; m < 4; m++) {
        if (m < t1) {
            if (4 <= t2) {
                C[(mb + wmb + txm16b2 * 4 + m) * N[Kt] + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + 0] = rC[m][0];
                C[(mb + wmb + txm16b2 * 4 + m) * N[Kt] + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + 1] = rC[m][1];
                C[(mb + wmb + txm16b2 * 4 + m) * N[Kt] + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + 2] = rC[m][2];
                C[(mb + wmb + txm16b2 * 4 + m) * N[Kt] + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + 3] = rC[m][3];
                /*
                float4 t;
                t.x = rC[m][0];
                t.y = rC[m][1];
                t.z = rC[m][2];
                t.w = rC[m][3];
                *reinterpret_cast<float4*>(C + (mb + wmb + txm16b2 * 4 + m) * N[Kt] + nb + wnb + txm32b16 * 8 + txm32m2 * 4) = t;
                */
            }
            else if (3 <= t2) {
                C[(mb + wmb + txm16b2 * 4 + m) * N[Kt] + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + 0] = rC[m][0];
                C[(mb + wmb + txm16b2 * 4 + m) * N[Kt] + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + 1] = rC[m][1];
                C[(mb + wmb + txm16b2 * 4 + m) * N[Kt] + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + 2] = rC[m][2];
            }
            else if (2 <= t2) {
                C[(mb + wmb + txm16b2 * 4 + m) * N[Kt] + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + 0] = rC[m][0];
                C[(mb + wmb + txm16b2 * 4 + m) * N[Kt] + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + 1] = rC[m][1];
            }
            else if (1 <= t2) {
                C[(mb + wmb + txm16b2 * 4 + m) * N[Kt] + nb + wnb + txm32b16 * 8 + txm32m2 * 4 + 0] = rC[m][0];
            }
        }
    }
}

__global__ void conv_sgemm_4x4x256_k(float* A, float* B, float* C, int H, int W, int Ch, int* M, int* N, int* K, int* offsetsA, int* offsetsC, int* Ks4, int minK) {

    __shared__ __align__(16) float sA[4][256];
    __shared__ __align__(16) float sB[4][256];
    __shared__ float sC[8][4][4];

    int tx = threadIdx.x;
    int mb = blockIdx.x * 4;
    int nb = blockIdx.y * 4;
    int Kt = Ks4[blockIdx.z];

    if (mb >= M[Kt] || nb >= N[Kt]) { return; }

    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm64 = tx % 64;
    int txb64 = tx / 64;

    A += offsetsA[Kt];
    C += offsetsC[Kt];

    float4 rA, rB;
    float rC[4][4];

    int WC = W * Ch;

    int m, n, k, kb, hb, wb, hp, wp, wcap, wpb, hpb, p;

    int wc = ceil((float)W / (Kt + minK));
    wcap = wc * (Kt + minK) * Ch;
    hb = (tx / wc) * (Kt + minK);
    wb = (tx % wc) * (Kt + minK) * Ch;
    hp = nb / ((Kt + minK) * Ch);
    wp = nb % ((Kt + minK) * Ch);
    wpb = wp;
    hpb = hp;
    if (hp >= Kt + minK) {
        hb = H;
    }

    int t1, t2;

    float(*psA)[4][256] = &sA;
    float(*psB)[4][256] = &sB;

    if (tx == 0) {
#pragma unroll
        for (m = 0; m < 4; m++) {
            if (mb + m < M[Kt]) {
                if (nb + 4 <= N[Kt]) {
                    rC[m][0] = C[(mb + m) * N[Kt] + nb + 0];
                    rC[m][1] = C[(mb + m) * N[Kt] + nb + 1];
                    rC[m][2] = C[(mb + m) * N[Kt] + nb + 2];
                    rC[m][3] = C[(mb + m) * N[Kt] + nb + 3];
                }
                else if (nb + 3 <= N[Kt]) {
                    rC[m][0] = C[(mb + m) * N[Kt] + nb + 0];
                    rC[m][1] = C[(mb + m) * N[Kt] + nb + 1];
                    rC[m][2] = C[(mb + m) * N[Kt] + nb + 2];
                    rC[m][3] = 0;
                }
                else if (nb + 2 <= N[Kt]) {
                    rC[m][0] = C[(mb + m) * N[Kt] + nb + 0];
                    rC[m][1] = C[(mb + m) * N[Kt] + nb + 1];
                    rC[m][2] = rC[m][3] = 0;
                }
                else if (nb + 1 <= N[Kt]) {
                    rC[m][0] = C[(mb + m) * N[Kt] + nb + 0];
                    rC[m][1] = rC[m][2] = rC[m][3] = 0;
                }
                else {
                    rC[m][0] = rC[m][1] = rC[m][2] = rC[m][3] = 0;
                }
            }
            else {
                rC[m][0] = rC[m][1] = rC[m][2] = rC[m][3] = 0;
            }
        }
    }
    else {
#pragma unroll
        for (m = 0; m < 4; m++) {
#pragma unroll
            for (n = 0; n < 4; n++) {
                rC[m][n] = 0;
            }
        }
    }

    if (tx < K[Kt]) {
        t1 = M[Kt] - mb;
        if (4 <= t1) {
            rA.x = A[tx * M[Kt] + mb + 0];
            rA.y = A[tx * M[Kt] + mb + 1];
            rA.z = A[tx * M[Kt] + mb + 2];
            rA.w = A[tx * M[Kt] + mb + 3];
        }
        else if (3 <= t1) {
            rA.x = A[tx * M[Kt] + mb + 0];
            rA.y = A[tx * M[Kt] + mb + 1];
            rA.z = A[tx * M[Kt] + mb + 2];
            rA.w = 0;
        }
        else if (2 <= t1) {
            rA.x = A[tx * M[Kt] + mb + 0];
            rA.y = A[tx * M[Kt] + mb + 1];
            rA.z = rA.w = 0;
        }
        else if (1 <= t1) {
            rA.x = A[tx * M[Kt] + mb + 0];
            rA.y = rA.z = rA.w = 0;
        }
        else {
            rA.x = rA.y = rA.z = rA.w = 0;
        }
    }
    else {
        rA.x = rA.y = rA.z = rA.w = 0;
    }

    rB.x = (hb + hp < H) ? ((wp < (Kt + minK)* Ch) ? ((wb + wp < WC) ? B[(hb + hp) * WC + (wb + wp++)] : 0 * wp++) : ((hb + hp + 1 < H && hp + 1 < Kt + minK) ? B[(hb + ++hp) * WC + wb + (wp = 0)++] : 0)) : 0;
    rB.y = (hb + hp < H) ? ((wp < (Kt + minK)* Ch) ? ((wb + wp < WC) ? B[(hb + hp) * WC + (wb + wp++)] : 0 * wp++) : ((hb + hp + 1 < H && hp + 1 < Kt + minK) ? B[(hb + ++hp) * WC + wb + (wp = 0)++] : 0)) : 0;
    rB.z = (hb + hp < H) ? ((wp < (Kt + minK)* Ch) ? ((wb + wp < WC) ? B[(hb + hp) * WC + (wb + wp++)] : 0 * wp++) : ((hb + hp + 1 < H && hp + 1 < Kt + minK) ? B[(hb + ++hp) * WC + wb + (wp = 0)++] : 0)) : 0;
    rB.w = (hb + hp < H) ? ((wp < (Kt + minK)* Ch) ? ((wb + wp < WC) ? B[(hb + hp) * WC + (wb + wp++)] : 0 * wp++) : ((hb + hp + 1 < H && hp + 1 < Kt + minK) ? B[(hb + ++hp) * WC + wb + (wp = 0)++] : 0)) : 0;

    wp = wpb;
    hp = hpb;
    wb += 256 * (Kt + minK) * Ch; 
    if (wb >= wcap) {
        hb += (wb / wcap) * (Kt + minK);
        wb = wb % wcap;
    }


    for (kb = 0; kb < K[Kt]; kb += 256) {

        __syncthreads();

        (*psA)[0][tx] = rA.x;
        (*psA)[1][tx] = rA.y;
        (*psA)[2][tx] = rA.z;
        (*psA)[3][tx] = rA.w;

        (*psB)[0][tx] = rB.x;
        (*psB)[1][tx] = rB.y;
        (*psB)[2][tx] = rB.z;
        (*psB)[3][tx] = rB.w;

        __syncthreads();

        if (kb + 256 < K[Kt]) {
            if (tx < K[Kt]) {
                t1 = M[Kt] - mb;
                if (4 <= t1) {
                    rA.x = A[(kb + 256 + tx) * M[Kt] + mb + 0];
                    rA.y = A[(kb + 256 + tx) * M[Kt] + mb + 1];
                    rA.z = A[(kb + 256 + tx) * M[Kt] + mb + 2];
                    rA.w = A[(kb + 256 + tx) * M[Kt] + mb + 3];
                    //rA = *reinterpret_cast<float4*>(A + (kb + 256 + tx) * M[Kt] + mb);
                }
                else if (3 <= t1) {
                    rA.x = A[(kb + 256 + tx) * M[Kt] + mb + 0];
                    rA.y = A[(kb + 256 + tx) * M[Kt] + mb + 1];
                    rA.z = A[(kb + 256 + tx) * M[Kt] + mb + 2];
                    rA.w = 0;
                }
                else if (2 <= t1) {
                    rA.x = A[(kb + 256 + tx) * M[Kt] + mb + 0];
                    rA.y = A[(kb + 256 + tx) * M[Kt] + mb + 1];
                    rA.z = rA.w = 0;
                }
                else if (1 <= t1) {
                    rA.x = A[(kb + 256 + tx) * M[Kt] + mb + 0];
                    rA.y = rA.z = rA.w = 0;
                }
                else {
                    rA.x = rA.y = rA.z = rA.w = 0;
                }
            }
            else {
                rA.x = rA.y = rA.z = rA.w = 0;
            }

            rB.x = (hb + hp < H) ? ((wp < (Kt + minK)* Ch) ? ((wb + wp < WC) ? B[(hb + hp) * WC + (wb + wp++)] : 0 * wp++) : ((hb + hp + 1 < H && hp + 1 < Kt + minK) ? B[(hb + ++hp) * WC + wb + (wp = 0)++] : 0)) : 0;
            rB.y = (hb + hp < H) ? ((wp < (Kt + minK)* Ch) ? ((wb + wp < WC) ? B[(hb + hp) * WC + (wb + wp++)] : 0 * wp++) : ((hb + hp + 1 < H && hp + 1 < Kt + minK) ? B[(hb + ++hp) * WC + wb + (wp = 0)++] : 0)) : 0;
            rB.z = (hb + hp < H) ? ((wp < (Kt + minK)* Ch) ? ((wb + wp < WC) ? B[(hb + hp) * WC + (wb + wp++)] : 0 * wp++) : ((hb + hp + 1 < H && hp + 1 < Kt + minK) ? B[(hb + ++hp) * WC + wb + (wp = 0)++] : 0)) : 0;
            rB.w = (hb + hp < H) ? ((wp < (Kt + minK)* Ch) ? ((wb + wp < WC) ? B[(hb + hp) * WC + (wb + wp++)] : 0 * wp++) : ((hb + hp + 1 < H && hp + 1 < Kt + minK) ? B[(hb + ++hp) * WC + wb + (wp = 0)++] : 0)) : 0;

            wp = wpb;
            hp = hpb;
            wb += 256 * (Kt + minK) * Ch;
            if (wb >= wcap) {
                hb += (wb / wcap) * (Kt + minK);
                wb = wb % wcap;
            }
        }

        // COMPUTE

#pragma unroll
        for (m = 0; m < 4; m++) {
#pragma unroll
            for (n = 0; n < 4; n++) {
                rC[m][n] += (*psA)[m][tx] * (*psB)[n][tx];
            }
        }

    }

#pragma unroll
    for (m = 0; m < 4; m++) {
#pragma unroll
        for (n = 0; n < 4; n++) {
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
            if (txm32 == 0) {
                sC[txb32][m][n] = rC[m][n];
            }
        }
    }

    __syncthreads();

    if (tx == 0) {
#pragma unroll
        for (m = 0; m < 4; m++) {
            if (mb + m < M[Kt]) {
                if (nb + 4 <= N[Kt]) {
                    C[(mb + m) * N[Kt] + nb + 0] = rC[m][0] + sC[1][m][0] + sC[2][m][0] + sC[3][m][0] + sC[4][m][0] + sC[5][m][0] + sC[6][m][0] + sC[7][m][0];
                    C[(mb + m) * N[Kt] + nb + 1] = rC[m][1] + sC[1][m][1] + sC[2][m][1] + sC[3][m][1] + sC[4][m][1] + sC[5][m][1] + sC[6][m][1] + sC[7][m][1];
                    C[(mb + m) * N[Kt] + nb + 2] = rC[m][2] + sC[1][m][2] + sC[2][m][2] + sC[3][m][2] + sC[4][m][2] + sC[5][m][2] + sC[6][m][2] + sC[7][m][2];
                    C[(mb + m) * N[Kt] + nb + 3] = rC[m][3] + sC[1][m][3] + sC[2][m][3] + sC[3][m][3] + sC[4][m][3] + sC[5][m][3] + sC[6][m][3] + sC[7][m][3];
                    /*
                    rA.x = rC[m][0] + sC[1][m][0] + sC[2][m][0] + sC[3][m][0] + sC[4][m][0] + sC[5][m][0] + sC[6][m][0] + sC[7][m][0];
                    rA.y = rC[m][1] + sC[1][m][1] + sC[2][m][1] + sC[3][m][1] + sC[4][m][1] + sC[5][m][1] + sC[6][m][1] + sC[7][m][1];
                    rA.z = rC[m][2] + sC[1][m][2] + sC[2][m][2] + sC[3][m][2] + sC[4][m][2] + sC[5][m][2] + sC[6][m][2] + sC[7][m][2];
                    rA.w = rC[m][3] + sC[1][m][3] + sC[2][m][3] + sC[3][m][3] + sC[4][m][3] + sC[5][m][3] + sC[6][m][3] + sC[7][m][3];
                    *reinterpret_cast<float4*>(C + (mb + m) * N[Kt] + nb) = rA;
                    */
                }
                else if (nb + 3 <= N[Kt]) {
                    C[(mb + m) * N[Kt] + nb + 0] = rC[m][0] + sC[1][m][0] + sC[2][m][0] + sC[3][m][0] + sC[4][m][0] + sC[5][m][0] + sC[6][m][0] + sC[7][m][0];
                    C[(mb + m) * N[Kt] + nb + 1] = rC[m][1] + sC[1][m][1] + sC[2][m][1] + sC[3][m][1] + sC[4][m][1] + sC[5][m][1] + sC[6][m][1] + sC[7][m][1];
                    C[(mb + m) * N[Kt] + nb + 2] = rC[m][2] + sC[1][m][2] + sC[2][m][2] + sC[3][m][2] + sC[4][m][2] + sC[5][m][2] + sC[6][m][2] + sC[7][m][2];
                }
                else if (nb + 2 <= N[Kt]) {
                    C[(mb + m) * N[Kt] + nb + 0] = rC[m][0] + sC[1][m][0] + sC[2][m][0] + sC[3][m][0] + sC[4][m][0] + sC[5][m][0] + sC[6][m][0] + sC[7][m][0];
                    C[(mb + m) * N[Kt] + nb + 1] = rC[m][1] + sC[1][m][1] + sC[2][m][1] + sC[3][m][1] + sC[4][m][1] + sC[5][m][1] + sC[6][m][1] + sC[7][m][1];
                }
                else if (nb + 1 <= N[Kt]) {
                    C[(mb + m) * N[Kt] + nb + 0] = rC[m][0] + sC[1][m][0] + sC[2][m][0] + sC[3][m][0] + sC[4][m][0] + sC[5][m][0] + sC[6][m][0] + sC[7][m][0];
                }
            }
        }
    }

}


/*
__global__ void conv_sgemm_128x128x8_i(float* A, float* B, float* C, int H, int W, int Ch, int* M, int* N, int* K, int* offsetsA, int* offsetsB, int* Ks128) {


    __shared__ __align__(16) float sA[8][132];
    __shared__ __align__(16) float sB[8][132];
    __shared__ __align__(16) float sAb[8][132];
    __shared__ __align__(16) float sBb[8][132];

    int tx = threadIdx.x;
    int mb = blockIdx.x * 128;
    int nb = blockIdx.y * 128;
    int Kt = Ks128[blockIdx.z];

    A += offsetsA[Kt];
    B += offsetsB[Kt];

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

    int WC = W * Ch;

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

    int t1, t2;

    if (mb + txm128 < M[Kt]) {
        t1 = K[Kt] - txb128 * 4;
        if (4 <= t1) {
            rA.x = A[(mb + txm128) * K[Kt] + txb128 * 4 + 0];
            rA.y = A[(mb + txm128) * K[Kt] + txb128 * 4 + 1];
            rA.z = A[(mb + txm128) * K[Kt] + txb128 * 4 + 2];
            rA.w = A[(mb + txm128) * K[Kt] + txb128 * 4 + 3];
            //rA = *reinterpret_cast<float4*>(A + (mb + txm128) * K[Kt] + txb128 * 4);
        }
        else if (3 <= t1) {
            rA.x = A[(mb + txm128) * K[Kt] + txb128 * 4 + 0];
            rA.y = A[(mb + txm128) * K[Kt] + txb128 * 4 + 1];
            rA.z = A[(mb + txm128) * K[Kt] + txb128 * 4 + 2];
            rA.w = 0;
        }
        else if (2 <= t1) {
            rA.x = A[(mb + txm128) * K[Kt] + txb128 * 4 + 0];
            rA.y = A[(mb + txm128) * K[Kt] + txb128 * 4 + 1];
            rA.z = rA.w = 0;
        }
        else if (1 <= t1) {
            rA.x = A[(mb + txm128) * K[Kt] + txb128 * 4 + 0];
            rA.y = rA.z = rA.w = 0;
        }
        else {
            rA.x = rA.y = rA.z = rA.w = 0;
        }
    }
    else {
        rA.x = rA.y = rA.z = rA.w = 0;
    }

    if (txb32 < K[Kt]) {
        t1 = N[Kt] - nb - txm32 * 4;
        if (4 <= t1) {
            rB.x = B[txb32 * N[Kt] + nb + txm32 * 4 + 0];
            rB.y = B[txb32 * N[Kt] + nb + txm32 * 4 + 1];
            rB.z = B[txb32 * N[Kt] + nb + txm32 * 4 + 2];
            rB.w = B[txb32 * N[Kt] + nb + txm32 * 4 + 3];
            //rB = *reinterpret_cast<float4*>(B + txb32 * N[Kt] + nb + txm32 * 4);
        }
        else if (3 <= t1) {
            rB.x = B[txb32 * N[Kt] + nb + txm32 * 4 + 0];
            rB.y = B[txb32 * N[Kt] + nb + txm32 * 4 + 1];
            rB.z = B[txb32 * N[Kt] + nb + txm32 * 4 + 2];
            rB.w = 0;
        }
        else if (2 <= t1) {
            rB.x = B[txb32 * N[Kt] + nb + txm32 * 4 + 0];
            rB.y = B[txb32 * N[Kt] + nb + txm32 * 4 + 1];
            rB.z = rB.w = 0;
        }
        else if (1 <= t1) {
            rB.x = B[txb32 * N[Kt] + nb + txm32 * 4 + 0];
            rB.y = rB.z = rB.w = 0;
        }
    }
    else {
        rB.x = rB.y = rB.z = rB.w = 0;
    }

    for (kb = 0; kb < K[Kt]; kb += 8) {

        (*psA)[txb128 * 4 + 0][txm128] = rA.x;
        (*psA)[txb128 * 4 + 1][txm128] = rA.y;
        (*psA)[txb128 * 4 + 2][txm128] = rA.z;
        (*psA)[txb128 * 4 + 3][txm128] = rA.w;

        (*psB)[txb32][txm32 * 4 + 0] = rB.x;
        (*psB)[txb32][txm32 * 4 + 1] = rB.y;
        (*psB)[txb32][txm32 * 4 + 2] = rB.z;
        (*psB)[txb32][txm32 * 4 + 3] = rB.w;

        __syncthreads();

        if (kb + 8 < K[Kt]) {

            if (mb + txm128 < M[Kt]) {
                t1 = K[Kt] - txb128 * 4 - kb - 8;
                if (4 <= t1) {
                    rA.x = A[(mb + txm128) * K[Kt] + kb + 8 + txb128 * 4 + 0];
                    rA.y = A[(mb + txm128) * K[Kt] + kb + 8 + txb128 * 4 + 1];
                    rA.z = A[(mb + txm128) * K[Kt] + kb + 8 + txb128 * 4 + 2];
                    rA.w = A[(mb + txm128) * K[Kt] + kb + 8 + txb128 * 4 + 3];
                    //rA = *reinterpret_cast<float4*>(A + (mb + txm128) * K[Kt] + kb + 8 + txb128 * 4);
                }
                else if (3 <= t1) {
                    rA.x = A[(mb + txm128) * K[Kt] + kb + 8 + txb128 * 4 + 0];
                    rA.y = A[(mb + txm128) * K[Kt] + kb + 8 + txb128 * 4 + 1];
                    rA.z = A[(mb + txm128) * K[Kt] + kb + 8 + txb128 * 4 + 2];
                    rA.w = 0;
                }
                else if (2 <= t1) {
                    rA.x = A[(mb + txm128) * K[Kt] + kb + 8 + txb128 * 4 + 0];
                    rA.y = A[(mb + txm128) * K[Kt] + kb + 8 + txb128 * 4 + 1];
                    rA.z = rA.w = 0;
                }
                else if (1 <= t1) {
                    rA.x = A[(mb + txm128) * K[Kt] + kb + 8 + txb128 * 4 + 0];
                    rA.y = rA.z = rA.w = 0;
                }
                else {
                    rA.x = rA.y = rA.z = rA.w = 0;
                }
            }
            else {
                rA.x = rA.y = rA.z = rA.w = 0;
            }

            if (kb + 8 + txb32 < K[Kt]) {
                t1 = N[Kt] - nb - txm32 * 4;
                if (4 <= t1) {
                    rB.x = B[(kb + 8 + txb32) * N[Kt] + nb + txm32 * 4 + 0];
                    rB.y = B[(kb + 8 + txb32) * N[Kt] + nb + txm32 * 4 + 1];
                    rB.z = B[(kb + 8 + txb32) * N[Kt] + nb + txm32 * 4 + 2];
                    rB.w = B[(kb + 8 + txb32) * N[Kt] + nb + txm32 * 4 + 3];
                    //rB = *reinterpret_cast<float4*>(B + (kb + 8 + txb32) * N[Kt] + nb + txm32 * 4);
                }
                else if (3 <= t1) {
                    rB.x = B[(kb + 8 + txb32) * N[Kt] + nb + txm32 * 4 + 0];
                    rB.y = B[(kb + 8 + txb32) * N[Kt] + nb + txm32 * 4 + 1];
                    rB.z = B[(kb + 8 + txb32) * N[Kt] + nb + txm32 * 4 + 2];
                    rB.w = 0;
                }
                else if (2 <= t1) {
                    rB.x = B[(kb + 8 + txb32) * N[Kt] + nb + txm32 * 4 + 0];
                    rB.y = B[(kb + 8 + txb32) * N[Kt] + nb + txm32 * 4 + 1];
                    rB.z = rB.w = 0;
                }
                else if (1 <= t1) {
                    rB.x = B[(kb + 8 + txb32) * N[Kt] + nb + txm32 * 4 + 0];
                    rB.y = rB.z = rB.w = 0;
                }
            }
            else {
                rB.x = rB.y = rB.z = rB.w = 0;
            }
        }

        // COMPUTE -------------------

        if (mb + sAi1 < M[Kt] && (nb + sBi1 < N[Kt] || nb + sBi2 < N[Kt])) {

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

    t1 = N[Kt] - nb - sBi1;
    t2 = N[Kt] - nb - sBi2;

#pragma unroll
    for (m = 0; m < 8; m++) {
        if (mb + sAi1 + m < M[Kt]) {
            float4 t;
            t.x = rC[m][0];
            t.y = rC[m][1];
            t.z = rC[m][2];
            t.w = rC[m][3];

            int b = (mb + sAi1 + m);
            int nw = ceil((float)W / Kt);
            int h = (b / nw) * Kt;
            int w = (b % nw) * Kt;

            int c = (nb + sBi1) % Ch;
            int nh = h + ((nb + sBi1) / Ch) / Kt;
            nw = w + ((nb + sBi1) / Ch) % Kt;
            int p = 0;

            if (4 <= t1) {
                atomicAdd(C + nh * WC + nw * Ch + c + p++, t.x);
                p = (p >= Ch) ? 0 * ((nw + 1 >= W) ? (nw = 0) * nh++ : nw++) : p;
                atomicAdd(C + nh * WC + nw * Ch + c + p++, t.y);
                p = (p >= Ch) ? 0 * ((nw + 1 >= W) ? (nw = 0) * nh++ : nw++) : p;
                atomicAdd(C + nh * WC + nw * Ch + c + p++, t.z);
                p = (p >= Ch) ? 0 * ((nw + 1 >= W) ? (nw = 0) * nh++ : nw++) : p;
                atomicAdd(C + nh * WC + nw * Ch + c + p++, t.w);
            }
            else if (3 <= t1) {
                atomicAdd(C + nh * WC + nw * Ch + c + p++, t.x);
                p = (p >= Ch) ? 0 * ((nw + 1 >= W) ? (nw = 0) * nh++ : nw++) : p;
                atomicAdd(C + nh * WC + nw * Ch + c + p++, t.y);
                p = (p >= Ch) ? 0 * ((nw + 1 >= W) ? (nw = 0) * nh++ : nw++) : p;
                atomicAdd(C + nh * WC + nw * Ch + c + p++, t.z);
            }
            else if (2 <= t1) {
                atomicAdd(C + nh * WC + nw * Ch + c + p++, t.x);
                p = (p >= Ch) ? 0 * ((nw + 1 >= W) ? (nw = 0) * nh++ : nw++) : p;
                atomicAdd(C + nh * WC + nw * Ch + c + p++, t.y);
            }
            else if (1 <= t1) {
                atomicAdd(C + nh * WC + nw * Ch + c + p++, t.x);
            }


            t.x = rC[m][4];
            t.y = rC[m][5];
            t.z = rC[m][6];
            t.w = rC[m][7];

            c = (nb + sBi2) % Ch;
            nh = h + ((nb + sBi2) / Ch) / (Kt + minK);
            nw = w + ((nb + sBi2) / Ch) % (Kt + minK);
            p = 0;

            if (4 <= t2) {
                atomicAdd(C + nh * WC + nw * Ch + c + p++, t.x);
                p = (p >= Ch) ? 0 * ((nw + 1 >= W) ? (nw = 0) * nh++ : nw++) : p;
                atomicAdd(C + nh * WC + nw * Ch + c + p++, t.y);
                p = (p >= Ch) ? 0 * ((nw + 1 >= W) ? (nw = 0) * nh++ : nw++) : p;
                atomicAdd(C + nh * WC + nw * Ch + c + p++, t.z);
                p = (p >= Ch) ? 0 * ((nw + 1 >= W) ? (nw = 0) * nh++ : nw++) : p;
                atomicAdd(C + nh * WC + nw * Ch + c + p++, t.w);
            }
            else if (3 <= t2) {
                atomicAdd(C + nh * WC + nw * Ch + c + p++, t.x);
                p = (p >= Ch) ? 0 * ((nw + 1 >= W) ? (nw = 0) * nh++ : nw++) : p;
                atomicAdd(C + nh * WC + nw * Ch + c + p++, t.y);
                p = (p >= Ch) ? 0 * ((nw + 1 >= W) ? (nw = 0) * nh++ : nw++) : p;
                atomicAdd(C + nh * WC + nw * Ch + c + p++, t.z);
            }
            else if (2 <= t2) {
                atomicAdd(C + nh * WC + nw * Ch + c + p++, t.x);
                p = (p >= Ch) ? 0 * ((nw + 1 >= W) ? (nw = 0) * nh++ : nw++) : p;
                atomicAdd(C + nh * WC + nw * Ch + c + p++, t.y);
            }
            else if (1 <= t2) {
                atomicAdd(C + nh * WC + nw * Ch + c + p++, t.x);
            }
        }
    }

}

__global__ void conv_sgemm_32x32x8_i(float* A, float* B, float* C, int H, int W, int Ch, int* M, int* N, int* K, int* offsetsA, int* offsetsB, int* Ks32) {

    __shared__ __align__(16) float sA[8][32];
    __shared__ __align__(16) float sB[8][32];
    __shared__ __align__(16) float sAb[8][32];
    __shared__ __align__(16) float sBb[8][32];

    int tx = threadIdx.x;
    int mb = blockIdx.x * 32;
    int nb = blockIdx.y * 32;
    int Kt = Ks32[blockIdx.z];

    A += offsetsA[Kt];
    B += offsetsB[Kt];

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
    float4 rA;
    float4 rB;
    float rAs[4];
    float rBs[4];
    float rAsb[4];
    float rBsb[4];

    float(*psA)[8][32] = &sA;
    float(*psB)[8][32] = &sB;

    int m, n, k, kb;

    int WC = W * Ch;

#pragma unroll
    for (m = 0; m < 4; m++) {
#pragma unroll
        for (n = 0; n < 4; n++) {
            rC[m][n] = 0;
        }
    }

    int t1, t2;

    if (mb + txm32 < M[Kt]) {
        if (txb32 * 4 + 4 <= K[Kt]) {
            rA.x = A[(mb + txm32) * K[Kt] + txb32 * 4 + 0];
            rA.y = A[(mb + txm32) * K[Kt] + txb32 * 4 + 1];
            rA.z = A[(mb + txm32) * K[Kt] + txb32 * 4 + 2];
            rA.w = A[(mb + txm32) * K[Kt] + txb32 * 4 + 3];
            //rA = *reinterpret_cast<float4*>(A + (mb + txm32) * K[Kt] + txb32 * 4);
        }
        else if (txb32 * 4 + 3 <= K[Kt]) {
            rA.x = A[(mb + txm32) * K[Kt] + txb32 * 4 + 0];
            rA.y = A[(mb + txm32) * K[Kt] + txb32 * 4 + 1];
            rA.z = A[(mb + txm32) * K[Kt] + txb32 * 4 + 2];
            rA.w = 0;
        }
        else if (txb32 * 4 + 2 <= K[Kt]) {
            rA.x = A[(mb + txm32) * K[Kt] + txb32 * 4 + 0];
            rA.y = A[(mb + txm32) * K[Kt] + txb32 * 4 + 1];
            rA.z = rA.w = 0;
        }
        else if (txb32 * 4 + 1 <= K[Kt]) {
            rA.x = A[(mb + txm32) * K[Kt] + txb32 * 4 + 0];
            rA.y = rA.z = rA.w = 0;
        }
        else {
            rA.x = rA.y = rA.z = rA.w = 0;
        }
    }
    else {
        rA.x = rA.y = rA.z = rA.w = 0;
    }

    if (txb8 < K[Kt]) {
        t1 = N[Kt] - nb - txm8 * 4;
        if (4 <= t1) {
            rB.x = B[txb8 * N[Kt] + nb + txm8 * 4 + 0];
            rB.y = B[txb8 * N[Kt] + nb + txm8 * 4 + 1];
            rB.z = B[txb8 * N[Kt] + nb + txm8 * 4 + 2];
            rB.w = B[txb8 * N[Kt] + nb + txm8 * 4 + 3];
            //rB = *reinterpret_cast<float4*>(B + txb8 * N[Kt] + nb + txm8 * 4);
        }
        else if (3 <= t1) {
            rB.x = B[txb8 * N[Kt] + nb + txm8 * 4 + 0];
            rB.y = B[txb8 * N[Kt] + nb + txm8 * 4 + 1];
            rB.z = B[txb8 * N[Kt] + nb + txm8 * 4 + 2];
            rB.w = 0;
        }
        else if (2 <= t1) {
            rB.x = B[txb8 * N[Kt] + nb + txm8 * 4 + 0];
            rB.y = B[txb8 * N[Kt] + nb + txm8 * 4 + 1];
            rB.z = rB.w = 0;
        }
        else if (1 <= t1) {
            rB.x = B[txb8 * N[Kt] + nb + txm8 * 4 + 0];
            rB.y = rB.z = rB.w = 0;
        }
        else {
            rB.x = rB.y = rB.z = rB.w = 0;
        }
    }
    else {
        rB.x = rB.y = rB.z = rB.w = 0;
    }

    for (kb = 0; kb < K[Kt]; kb += 8) {

        (*psA)[txb32 * 4 + 0][txm32] = rA.x;
        (*psA)[txb32 * 4 + 1][txm32] = rA.y;
        (*psA)[txb32 * 4 + 2][txm32] = rA.z;
        (*psA)[txb32 * 4 + 3][txm32] = rA.w;

        (*psB)[txb8][txm8 * 4 + 0] = rB.x;
        (*psB)[txb8][txm8 * 4 + 1] = rB.y;
        (*psB)[txb8][txm8 * 4 + 2] = rB.z;
        (*psB)[txb8][txm8 * 4 + 3] = rB.w;

        __syncthreads();

        if (kb + 8 < K[Kt]) {
            if (mb + txm32 < M[Kt]) {
                if (kb + 8 + txb32 * 4 + 4 <= K[Kt]) {
                    rA.x = A[(mb + txm32) * K[Kt] + kb + 8 + txb32 * 4 + 0];
                    rA.y = A[(mb + txm32) * K[Kt] + kb + 8 + txb32 * 4 + 1];
                    rA.z = A[(mb + txm32) * K[Kt] + kb + 8 + txb32 * 4 + 2];
                    rA.w = A[(mb + txm32) * K[Kt] + kb + 8 + txb32 * 4 + 3];
                    //rA = *reinterpret_cast<float4*>(A + (mb + txm32) * K[Kt] + kb + 8 + txb32 * 4);
                }
                else if (kb + 8 + txb32 * 4 + 3 <= K[Kt]) {
                    rA.x = A[(mb + txm32) * K[Kt] + kb + 8 + txb32 * 4 + 0];
                    rA.y = A[(mb + txm32) * K[Kt] + kb + 8 + txb32 * 4 + 1];
                    rA.z = A[(mb + txm32) * K[Kt] + kb + 8 + txb32 * 4 + 2];
                    rA.w = 0;
                }
                else if (kb + 8 + txb32 * 4 + 2 <= K[Kt]) {
                    rA.x = A[(mb + txm32) * K[Kt] + kb + 8 + txb32 * 4 + 0];
                    rA.y = A[(mb + txm32) * K[Kt] + kb + 8 + txb32 * 4 + 1];
                    rA.z = rA.w = 0;
                }
                else if (kb + 8 + txb32 * 4 + 1 <= K[Kt]) {
                    rA.x = A[(mb + txm32) * K[Kt] + kb + 8 + txb32 * 4 + 0];
                    rA.y = rA.z = rA.w = 0;
                }
                else {
                    rA.x = rA.y = rA.z = rA.w = 0;
                }
            }
            else {
                rA.x = rA.y = rA.z = rA.w = 0;
            }

            if (kb + 8 + txb8 < K[Kt]) {
                t1 = N[Kt] - nb - txm8 * 4;
                if (4 <= t1) {
                    rB.x = B[(kb + 8 + txb8) * N[Kt] + nb + txm8 * 4 + 0];
                    rB.y = B[(kb + 8 + txb8) * N[Kt] + nb + txm8 * 4 + 1];
                    rB.z = B[(kb + 8 + txb8) * N[Kt] + nb + txm8 * 4 + 2];
                    rB.w = B[(kb + 8 + txb8) * N[Kt] + nb + txm8 * 4 + 3];
                    //rB = *reinterpret_cast<float4*>(B + (kb + 8 + txb8) * N[Kt] + nb + txm8 * 4);
                }
                else if (3 <= t1) {
                    rB.x = B[(kb + 8 + txb8) * N[Kt] + nb + txm8 * 4 + 0];
                    rB.y = B[(kb + 8 + txb8) * N[Kt] + nb + txm8 * 4 + 1];
                    rB.z = B[(kb + 8 + txb8) * N[Kt] + nb + txm8 * 4 + 2];
                    rB.w = 0;
                }
                else if (2 <= t1) {
                    rB.x = B[(kb + 8 + txb8) * N[Kt] + nb + txm8 * 4 + 0];
                    rB.y = B[(kb + 8 + txb8) * N[Kt] + nb + txm8 * 4 + 1];
                    rB.z = rB.w = 0;
                }
                else if (1 <= t1) {
                    rB.x = B[(kb + 8 + txb8) * N[Kt] + nb + txm8 * 4 + 0];
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

        // COMPUTE -------------------------

        if (mb + wmb + txm16b2 * 4 < M[Kt] && nb + wnb + txm32b16 * 8 + txm32m2 * 4 < N[Kt]) {

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
            for (k = 0; k < 8; k++) {

                if (k < 8) {
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
        }

        // ---------------------------------

        psA = (psA == &sA) ? &sAb : &sA;
        psB = (psB == &sB) ? &sBb : &sB;

    }


    t1 = M[Kt] - mb - wmb - txm16b2 * 4;
    t2 = N[Kt] - nb - wnb - txm32b16 * 8 - txm32m2 * 4;

#pragma unroll
    for (m = 0; m < 4; m++) {
        if (m < t1) {
            float4 t;
            t.x = rC[m][0];
            t.y = rC[m][1];
            t.z = rC[m][2];
            t.w = rC[m][3];

            int b = (M[Kt] - t1 + m);
            int nw = ceil((float)W / Kt);
            int h = (b / nw) * Kt;
            int w = (b % nw) * Kt;

            int c = (N[Kt] - t2) % Ch;
            int nh = h + ((N[Kt] - t2) / Ch) / Kt;
            nw = w + ((N[Kt] - t2) / Ch) % Kt;
            int p = 0;

            if (4 <= t2) {
                atomicAdd(C + nh * WC + nw * Ch + c + p++, t.x);
                p = (p >= Ch) ? 0 * ((nw + 1 >= W) ? (nw = 0) * nh++ : nw++) : p;
                atomicAdd(C + nh * WC + nw * Ch + c + p++, t.y);
                p = (p >= Ch) ? 0 * ((nw + 1 >= W) ? (nw = 0) * nh++ : nw++) : p;
                atomicAdd(C + nh * WC + nw * Ch + c + p++, t.z);
                p = (p >= Ch) ? 0 * ((nw + 1 >= W) ? (nw = 0) * nh++ : nw++) : p;
                atomicAdd(C + nh * WC + nw * Ch + c + p++, t.w);
            }
            else if (3 <= t2) {
                atomicAdd(C + nh * WC + nw * Ch + c + p++, t.x);
                p = (p >= Ch) ? 0 * ((nw + 1 >= W) ? (nw = 0) * nh++ : nw++) : p;
                atomicAdd(C + nh * WC + nw * Ch + c + p++, t.y);
                p = (p >= Ch) ? 0 * ((nw + 1 >= W) ? (nw = 0) * nh++ : nw++) : p;
                atomicAdd(C + nh * WC + nw * Ch + c + p++, t.z);
            }
            else if (2 <= t2) {
                atomicAdd(C + nh * WC + nw * Ch + c + p++, t.x);
                p = (p >= Ch) ? 0 * ((nw + 1 >= W) ? (nw = 0) * nh++ : nw++) : p;
                atomicAdd(C + nh * WC + nw * Ch + c + p++, t.y);
            }
            else if (1 <= t2) {
                atomicAdd(C + nh * WC + nw * Ch + c + p++, t.x);
            }
        }
    }
}

__global__ void conv_sgemm_4x4x256_i(float* A, float* B, float* C, int H, int W, int Ch, int* M, int* N, int* K, int* offsetsA, int* offsetsB, int* Ks4) {

    __shared__ __align__(16) float sA[4][256];
    __shared__ __align__(16) float sB[4][256];
    __shared__ float sC[8][4][4];

    int tx = threadIdx.x;
    int mb = blockIdx.x * 4;
    int nb = blockIdx.y * 4;
    int Kt = Ks4[blockIdx.z];

    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm64 = tx % 64;
    int txb64 = tx / 64;

    A += offsetsA[Kt];
    B += offsetsB[Kt];

    float4 rA, rB;
    float rC[4][4];

    int kb, k, m, n;
    int t1, t2;

    int WC = W * Ch;

    float(*psA)[4][256] = &sA;
    float(*psB)[4][256] = &sB;

    for (m = 0; m < 4; m++) {
        for (n = 0; n < 4; n++) {
            rC[m][n] = 0;
        }
    }

    if (mb + txb64 < M[Kt]) {
        if (txm64 * 4 + 4 <= K[Kt]) {
            rA.x = A[(mb + txb64) * K[Kt] + txm64 * 4 + 0];
            rA.y = A[(mb + txb64) * K[Kt] + txm64 * 4 + 1];
            rA.z = A[(mb + txb64) * K[Kt] + txm64 * 4 + 2];
            rA.w = A[(mb + txb64) * K[Kt] + txm64 * 4 + 3];
            //rA = *reinterpret_cast<float4*>(A + (mb + txb64) * K[Kt] + txm64 * 4);
        }
        else if (txm64 * 4 + 3 <= K[Kt]) {
            rA.x = A[(mb + txb64) * K[Kt] + txm64 * 4 + 0];
            rA.y = A[(mb + txb64) * K[Kt] + txm64 * 4 + 1];
            rA.z = A[(mb + txb64) * K[Kt] + txm64 * 4 + 2];
            rA.w = 0;
        }
        else if (txm64 * 4 + 2 <= K[Kt]) {
            rA.x = A[(mb + txb64) * K[Kt] + txm64 * 4 + 0];
            rA.y = A[(mb + txb64) * K[Kt] + txm64 * 4 + 1];
            rA.z = rA.w = 0;
        }
        else if (txm64 * 4 + 1 <= K[Kt]) {
            rA.x = A[(mb + txb64) * K[Kt] + txm64 * 4 + 0];
            rA.y = rA.z = rA.w = 0;
        }
        else {
            rA.x = rA.y = rA.z = rA.w = 0;
        }
    }
    else {
        rA.x = rA.y = rA.z = rA.w = 0;
    }

    if (tx < K[Kt]) {
        t1 = N[Kt] - nb;
        if (4 <= t1) {
            rB.x = B[tx * N[Kt] + nb + 0];
            rB.y = B[tx * N[Kt] + nb + 1];
            rB.z = B[tx * N[Kt] + nb + 2];
            rB.w = B[tx * N[Kt] + nb + 3];
            //rB = *reinterpret_cast<float4*>(B + tx * N[Kt] + nb);
        }
        else if (3 <= t1) {
            rB.x = B[tx * N[Kt] + nb + 0];
            rB.y = B[tx * N[Kt] + nb + 1];
            rB.z = B[tx * N[Kt] + nb + 2];
            rB.w = 0;
        }
        else if (2 <= t1) {
            rB.x = B[tx * N[Kt] + nb + 0];
            rB.y = B[tx * N[Kt] + nb + 1];
            rB.z = rB.w = 0;
        }
        else if (1 <= t1) {
            rB.x = B[tx * N[Kt] + nb + 0];
            rB.y = rB.z = rB.w = 0;
        }
        else {
            rB.x = rB.y = rB.z = rB.w = 0;
        }
    }
    else {
        rB.x = rB.y = rB.z = rB.w = 0;
    }


    for (kb = 0; kb < K[Kt]; kb += 256) {

        __syncthreads();

        (*psA)[txb64][txm64 * 4 + 0] = rA.x;
        (*psA)[txb64][txm64 * 4 + 1] = rA.y;
        (*psA)[txb64][txm64 * 4 + 2] = rA.z;
        (*psA)[txb64][txm64 * 4 + 3] = rA.w;

        (*psB)[0][tx] = rB.x;
        (*psB)[1][tx] = rB.y;
        (*psB)[2][tx] = rB.z;
        (*psB)[3][tx] = rB.w;

        __syncthreads();

        if (kb + 256 < K[Kt]) {
            if (mb + txb64 < M[Kt]) {
                if (kb + 256 + txm64 * 4 + 4 <= K[Kt]) {
                    rA.x = A[(mb + txb64) * K[Kt] + kb + 256 + txm64 * 4 + 0];
                    rA.y = A[(mb + txb64) * K[Kt] + kb + 256 + txm64 * 4 + 1];
                    rA.z = A[(mb + txb64) * K[Kt] + kb + 256 + txm64 * 4 + 2];
                    rA.w = A[(mb + txb64) * K[Kt] + kb + 256 + txm64 * 4 + 3];
                    //rA = *reinterpret_cast<float4*>(A + (mb + txb64) * K[Kt] + kb + 256 + txm64 * 4);
                }
                else if (kb + 256 + txm64 * 4 + 3 <= K[Kt]) {
                    rA.x = A[(mb + txb64) * K[Kt] + kb + 256 + txm64 * 4 + 0];
                    rA.y = A[(mb + txb64) * K[Kt] + kb + 256 + txm64 * 4 + 1];
                    rA.z = A[(mb + txb64) * K[Kt] + kb + 256 + txm64 * 4 + 2];
                    rA.w = 0;
                }
                else if (kb + 256 + txm64 * 4 + 2 <= K[Kt]) {
                    rA.x = A[(mb + txb64) * K[Kt] + kb + 256 + txm64 * 4 + 0];
                    rA.y = A[(mb + txb64) * K[Kt] + kb + 256 + txm64 * 4 + 1];
                    rA.z = rA.w = 0;
                }
                else if (kb + 256 + txm64 * 4 + 1 <= K[Kt]) {
                    rA.x = A[(mb + txb64) * K[Kt] + kb + 256 + txm64 * 4 + 0];
                    rA.y = rA.z = rA.w = 0;
                }
                else {
                    rA.x = rA.y = rA.z = rA.w = 0;
                }
            }
            else {
                rA.x = rA.y = rA.z = rA.w = 0;
            }

            if (kb + 256 + tx < K[Kt]) {
                t1 = N[Kt] - nb;
                if (4 <= t1) {
                    rB.x = B[(kb + 256 + tx) * N[Kt] + nb + 0];
                    rB.y = B[(kb + 256 + tx) * N[Kt] + nb + 1];
                    rB.z = B[(kb + 256 + tx) * N[Kt] + nb + 2];
                    rB.w = B[(kb + 256 + tx) * N[Kt] + nb + 3];
                    //rB = *reinterpret_cast<float4*>(B + (kb + 256 + tx) * N[Kt] + nb);
                }
                else if (3 <= t1) {
                    rB.x = B[(kb + 256 + tx) * N[Kt] + nb + 0];
                    rB.y = B[(kb + 256 + tx) * N[Kt] + nb + 1];
                    rB.z = B[(kb + 256 + tx) * N[Kt] + nb + 2];
                    rB.w = 0;
                }
                else if (2 <= t1) {
                    rB.x = B[(kb + 256 + tx) * N[Kt] + nb + 0];
                    rB.y = B[(kb + 256 + tx) * N[Kt] + nb + 1];
                    rB.z = rB.w = 0;
                }
                else if (1 <= t1) {
                    rB.x = B[(kb + 256 + tx) * N[Kt] + nb + 0];
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

        // COMPUTE

#pragma unroll
        for (m = 0; m < 4; m++) {
#pragma unroll
            for (n = 0; n < 4; n++) {
                rC[m][n] += (*psA)[m][tx] * (*psB)[n][tx];
            }
        }

    }

#pragma unroll
    for (m = 0; m < 4; m++) {
#pragma unroll
        for (n = 0; n < 4; n++) {
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
            if (txm32 == 0) {
                sC[txb32][m][n] = rC[m][n];
            }
        }
    }

    __syncthreads();

    if (tx == 0) {
#pragma unroll
        for (m = 0; m < 4; m++) {
            if (mb + m < M[Kt]) {

                float4 t;

                t.x = rC[m][0] + sC[1][m][0] + sC[2][m][0] + sC[3][m][0] + sC[4][m][0] + sC[5][m][0] + sC[6][m][0] + sC[7][m][0];
                t.y = rC[m][1] + sC[1][m][1] + sC[2][m][1] + sC[3][m][1] + sC[4][m][1] + sC[5][m][1] + sC[6][m][1] + sC[7][m][1];
                t.z = rC[m][2] + sC[1][m][2] + sC[2][m][2] + sC[3][m][2] + sC[4][m][2] + sC[5][m][2] + sC[6][m][2] + sC[7][m][2];
                t.w = rC[m][3] + sC[1][m][3] + sC[2][m][3] + sC[3][m][3] + sC[4][m][3] + sC[5][m][3] + sC[6][m][3] + sC[7][m][3];

                int b = (mb + m);
                int nw = ceil((float)W / Kt);
                int h = (b / nw) * Kt;
                int w = (b % nw) * Kt;

                int c = nb % Ch;
                int nh = h + (nb / Ch) / Kt;
                nw = w + (nb / Ch) % Kt;
                int p = 0;

                if (nb + 4 <= N[Kt]) {
                    atomicAdd(C + nh * WC + nw * Ch + c + p++, t.x);
                    p = (p >= Ch) ? 0 * ((nw + 1 >= W) ? (nw = 0) * nh++ : nw++) : p;
                    atomicAdd(C + nh * WC + nw * Ch + c + p++, t.y);
                    p = (p >= Ch) ? 0 * ((nw + 1 >= W) ? (nw = 0) * nh++ : nw++) : p;
                    atomicAdd(C + nh * WC + nw * Ch + c + p++, t.z);
                    p = (p >= Ch) ? 0 * ((nw + 1 >= W) ? (nw = 0) * nh++ : nw++) : p;
                    atomicAdd(C + nh * WC + nw * Ch + c + p++, t.w);
                }
                if (nb + 3 <= N[Kt]) {
                    atomicAdd(C + nh * WC + nw * Ch + c + p++, t.x);
                    p = (p >= Ch) ? 0 * ((nw + 1 >= W) ? (nw = 0) * nh++ : nw++) : p;
                    atomicAdd(C + nh * WC + nw * Ch + c + p++, t.y);
                    p = (p >= Ch) ? 0 * ((nw + 1 >= W) ? (nw = 0) * nh++ : nw++) : p;
                    atomicAdd(C + nh * WC + nw * Ch + c + p++, t.z);
                }
                if (nb + 2 <= N[Kt]) {
                    atomicAdd(C + nh * WC + nw * Ch + c + p++, t.x);
                    p = (p >= Ch) ? 0 * ((nw + 1 >= W) ? (nw = 0) * nh++ : nw++) : p;
                    atomicAdd(C + nh * WC + nw * Ch + c + p++, t.y);
                }
                if (nb + 1 <= N[Kt]) {
                    atomicAdd(C + nh * WC + nw * Ch + c + p++, t.x);
                }
            }
        }
    }

}
*/


__global__ void conv_add_bias(float* out, float* bias, int* Ms, int* Ns, int* biasOffsets, int* outOffsets) {

    int tx = threadIdx.x;
    int Kt = blockIdx.z;

    int m = blockIdx.x;
    int n = blockIdx.y * 256 + tx;

    if (m >= Ms[Kt] || n >= Ns[Kt]) {
        return;
    }

    bias += biasOffsets[Kt];
    out += outOffsets[Kt];

    out[m * Ns[Kt] + n] += bias[n];

}

__global__ void conv_bias_grad(float* out, float* bias, int* Ms, int* Ns, int* biasOffsets, int* outOffsets) {

    int tx = threadIdx.x;
    int Kt = blockIdx.z;
    int n = blockIdx.x * 256 + tx;

    if (n >= Ns[Kt]) {
        return;
    }

    bias += biasOffsets[Kt];
    out += outOffsets[Kt];

    float s = 0;
    for (int m = 0; m < Ms[Kt]; m++) {
        s += out[m * Ns[Kt] + n];
    }
    bias[n] = s;

}


DLLEXPORT void cuda_conv_sgemm_forward(float* dimg, float* dkern, float* dout, int H, int W, int C, int* M, int* N, int* K, int* offsetsB, int* offsetsC, int* dims128, int* dims32, int* dims4, int* Ks128, int* Ks32, int* Ks4, int minK) {

    if (dims128[0] != 0 && dims128[1] != 0 && dims128[2] != 0) {
        conv_sgemm_128x128x8_f << <dim3((int)ceil((float)dims128[0] / 128), (int)ceil((float)dims128[1] / 128), dims128[2]), dim3(256, 1, 1) >> > (dimg, dkern, dout, H, W, C, M, N, K, offsetsB, offsetsC, Ks128, minK);
    }
    if (dims32[0] != 0 && dims32[1] != 0 && dims32[2] != 0) {
        conv_sgemm_32x32x8_f << <dim3((int)ceil((float)dims32[0] / 32), (int)ceil((float)dims32[1] / 32), dims32[2]), dim3(64, 1, 1) >> > (dimg, dkern, dout, H, W, C, M, N, K, offsetsB, offsetsC, Ks32, minK);
    }
    if (dims4[0] != 0 && dims4[1] != 0 && dims4[2] != 0) {
        conv_sgemm_4x4x256_f << <dim3((int)ceil((float)dims4[0] / 4), (int)ceil((float)dims4[1] / 4), dims4[2]), dim3(256, 1, 1) >> > (dimg, dkern, dout, H, W, C, M, N, K, offsetsB, offsetsC, Ks4, minK);
    }

}

DLLEXPORT void cuda_conv_sgemm_kerngrad(float* dimg, float* dkern, float* dout, int H, int W, int C, int* M, int* N, int* K, int* offsetsB, int* offsetsC, int* dims128, int* dims32, int* dims4, int* Ks128, int* Ks32, int* Ks4, int minK) {

    if (dims128[0] != 0 && dims128[1] != 0 && dims128[2] != 0) {
        conv_sgemm_128x128x8_k << <dim3((int)ceil((float)dims128[0] / 128), (int)ceil((float)dims128[1] / 128), dims128[2]), dim3(256, 1, 1) >> > (dout, dimg, dkern, H, W, C, N, K, M, offsetsC, offsetsB, Ks128, minK);
    }
    if (dims32[0] != 0 && dims32[1] != 0 && dims32[2] != 0) {
        conv_sgemm_32x32x8_k << <dim3((int)ceil((float)dims32[0] / 32), (int)ceil((float)dims32[1] / 32), dims32[2]), dim3(64, 1, 1) >> > (dout, dimg, dkern, H, W, C, N, K, M, offsetsC, offsetsB, Ks32, minK);
    }
    if (dims4[0] != 0 && dims4[1] != 0 && dims4[2] != 0) {
        conv_sgemm_4x4x256_k << <dim3((int)ceil((float)dims4[0] / 4), (int)ceil((float)dims4[1] / 4), dims4[2]), dim3(256, 1, 1) >> > (dout, dimg, dkern, H, W, C, N, K, M, offsetsC, offsetsB, Ks4, minK);
    }

}

DLLEXPORT void cuda_conv_sgemm_inpgrad(float* dimg, float* dkern, float* dout, int H, int W, int C, int* M, int* N, int* K, int* offsetsB, int* offsetsC, int* dims128, int* dims32, int* dims4, int* Ks128, int* Ks32, int* Ks4) {

    if (dims128[0] != 0 && dims128[1] != 0 && dims128[2] != 0) {
        //conv_sgemm_128x128x8_i << <dim3((int)ceil((float)hM[0] / 128), (int)ceil((float)maxN128 / 128), sgemm128 + 1), dim3(256, 1, 1) >> > (dout, dkern, dimg, H, W, C, M, K, N, offsetsC, offsetsB, minK);
    }
    if (dims32[0] != 0 && dims32[1] != 0 && dims32[2] != 0) {
        //conv_sgemm_32x32x8_i << <dim3((int)ceil((float)hM[sgemm128 + 1] / 32), (int)ceil((float)maxN32 / 32), sgemm32 - sgemm128), dim3(64, 1, 1) >> > (dout, dkern, dimg, H, W, C, M, K, N, offsetsC, offsetsB, minK, sgemm128 + 1);
    }
    if (dims4[0] != 0 && dims4[1] != 0 && dims4[2] != 0) {
        //conv_sgemm_4x4x256_i << <dim3((int)ceil((float)hM[sgemm32 + 1] / 4), (int)ceil((float)maxN4 / 4), kc - sgemm32), dim3(256, 1, 1) >> > (dout, dkern, dimg, H, W, C, M, K, N, offsetsC, offsetsB, minK, sgemm32 + 1);
    }

}


DLLEXPORT void cuda_conv_add_bias(float* dout, float* dbias, int* Ms, int* Ns, int* biasOffsets, int* outOffsets, int* dims) {

    conv_add_bias << <dim3((int)ceil((float)dims[0]), (int)ceil((float)dims[1] / 256), dims[2]), dim3(256, 1, 1) >> > (dout, dbias, Ms, Ns, biasOffsets, outOffsets);

}

DLLEXPORT void cuda_conv_bias_grad(float* dout, float* dbias, int* Ms, int* Ns, int* biasOffsets, int* outOffsets, int* dims) {

    conv_bias_grad << <dim3((int)ceil((float)dims[1] / 256), 1, dims[2]), dim3(256, 1, 1) >> > (dout, dbias, Ms, Ns, biasOffsets, outOffsets);

}