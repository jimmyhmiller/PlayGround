// Reference C implementation to verify layernorm_forward
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            float* x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd (reciprocal standard deviation)
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalize
                float o = n * weight[i] + bias[i]; // scale and shift
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

int main() {
    printf("Testing layernorm_forward (C reference)...\n");

    // Test dimensions: B=2, T=2, C=3
    int B = 2, T = 2, C = 3;

    // Allocate arrays
    float* out = (float*)calloc(B * T * C, sizeof(float));
    float* mean = (float*)calloc(B * T, sizeof(float));
    float* rstd = (float*)calloc(B * T, sizeof(float));
    float* inp = (float*)malloc(B * T * C * sizeof(float));
    float* weight = (float*)malloc(C * sizeof(float));
    float* bias = (float*)malloc(C * sizeof(float));

    // Initialize inp with sequential values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for (int i = 0; i < B * T * C; i++) {
        inp[i] = i + 1.0f;
    }

    // Initialize weight and bias
    for (int i = 0; i < C; i++) {
        weight[i] = 1.0f;
        bias[i] = 0.0f;
    }

    // Run layernorm_forward
    layernorm_forward(out, mean, rstd, inp, weight, bias, B, T, C);

    // Print first few outputs for verification
    printf("First output values:\n");
    printf("  out[0] = %f\n", out[0]);
    printf("  out[1] = %f\n", out[1]);
    printf("  out[2] = %f\n", out[2]);
    printf("  mean[0] = %f\n", mean[0]);
    printf("  rstd[0] = %f\n", rstd[0]);

    // Clean up
    free(out);
    free(mean);
    free(rstd);
    free(inp);
    free(weight);
    free(bias);

    printf("layernorm_forward test completed!\n");
    return 0;
}
