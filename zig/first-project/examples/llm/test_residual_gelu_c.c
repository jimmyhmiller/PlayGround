// Reference C implementation to verify residual_forward and gelu_forward
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void residual_forward(float* out, float* inp1, float* inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
void gelu_forward(float* out, float* inp, int N) {
    // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}

int main() {
    printf("Testing residual_forward (C reference)...\n");

    // Test dimensions
    int N = 6;
    float* out = (float*)malloc(N * sizeof(float));
    float* inp1 = (float*)malloc(N * sizeof(float));
    float* inp2 = (float*)malloc(N * sizeof(float));

    // Initialize inp1: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    for (int i = 0; i < N; i++) {
        inp1[i] = i + 1.0f;
    }

    // Initialize inp2: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    for (int i = 0; i < N; i++) {
        inp2[i] = (i + 1.0f) * 0.1f;
    }

    // Run residual_forward
    residual_forward(out, inp1, inp2, N);

    // Print outputs
    printf("First output values:\n");
    printf("  out[0] = %f (expected 1.1)\n", out[0]);
    printf("  out[1] = %f (expected 2.2)\n", out[1]);
    printf("  out[2] = %f (expected 3.3)\n", out[2]);

    // Clean up
    free(out);
    free(inp1);
    free(inp2);

    printf("residual_forward test completed!\n\n");

    // Test gelu_forward
    printf("Testing gelu_forward (C reference)...\n");

    N = 4;
    out = (float*)malloc(N * sizeof(float));
    float* inp = (float*)malloc(N * sizeof(float));

    // Initialize inp with test values: [-1.0, 0.0, 1.0, 2.0]
    inp[0] = -1.0f;
    inp[1] = 0.0f;
    inp[2] = 1.0f;
    inp[3] = 2.0f;

    // Run gelu_forward
    gelu_forward(out, inp, N);

    // Print outputs
    printf("First output values:\n");
    printf("  out[0] = %f\n", out[0]);
    printf("  out[1] = %f\n", out[1]);
    printf("  out[2] = %f\n", out[2]);
    printf("  out[3] = %f\n", out[3]);

    // Clean up
    free(out);
    free(inp);

    printf("gelu_forward test completed!\n");
    return 0;
}
