// Reference C implementation to verify matmul_forward_naive
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void matmul_forward_naive(float* out,
                         const float* inp, const float* weight, const float* bias,
                         int B, int T, int C, int OC) {
    // the most naive implementation of matrix multiplication
    // inp is (B,T,C), weight is (OC, C), bias is (OC) or NULL
    // out will be (B,T,OC)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int bt = b * T + t;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                for (int i = 0; i < C; i++) {
                    val += inp[bt * C + i] * weight[o*C + i];
                }
                out[bt * OC + o] = val;
            }
        }
    }
}

int main() {
    printf("Testing matmul_forward_naive (C reference)...\n");

    // Test dimensions: B=2, T=2, C=3, OC=4
    int B = 2, T = 2, C = 3, OC = 4;

    // Allocate arrays
    float* out = (float*)malloc(B * T * OC * sizeof(float));
    float* inp = (float*)malloc(B * T * C * sizeof(float));
    float* weight = (float*)malloc(OC * C * sizeof(float));
    float* bias = (float*)malloc(OC * sizeof(float));

    // Initialize inp with sequential values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for (int i = 0; i < B * T * C; i++) {
        inp[i] = i + 1.0f;
    }

    // Initialize weight with sequential values
    for (int i = 0; i < OC * C; i++) {
        weight[i] = i + 1.0f;
    }

    // Initialize bias with sequential values
    for (int i = 0; i < OC; i++) {
        bias[i] = i + 0.1f;
    }

    // Run matmul_forward_naive
    matmul_forward_naive(out, inp, weight, bias, B, T, C, OC);

    // Print first few outputs for verification
    printf("First output values:\n");
    printf("  out[0] = %f\n", out[0]);
    printf("  out[1] = %f\n", out[1]);
    printf("  out[2] = %f\n", out[2]);
    printf("  out[3] = %f\n", out[3]);

    // Clean up
    free(out);
    free(inp);
    free(weight);
    free(bias);

    printf("matmul_forward_naive test completed!\n");
    return 0;
}
