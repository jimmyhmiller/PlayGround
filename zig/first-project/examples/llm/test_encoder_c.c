// Reference C implementation to verify encoder_forward
#include <stdio.h>
#include <stdlib.h>

void encoder_forward(float* out,
                   int* inp, float* wte, float* wpe,
                   int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            // get the index of the token at inp[b, t]
            int ix = inp[b * T + t];
            // seek to the position in wte corresponding to the token
            float* wte_ix = wte + ix * C;
            // seek to the position in wpe corresponding to the position
            float* wpe_t = wpe + t * C;
            // add the two vectors and store the result in out[b,t,:]
            for (int i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

int main() {
    printf("Testing encoder_forward (C reference)...\n");

    // Test dimensions: B=2, T=3, C=4
    int B = 2, T = 3, C = 4, V = 10;

    // Allocate arrays
    float* out = (float*)calloc(B * T * C, sizeof(float));
    int* inp = (int*)malloc(B * T * sizeof(int));
    float* wte = (float*)malloc(V * C * sizeof(float));
    float* wpe = (float*)malloc(T * C * sizeof(float));

    // Initialize inp with token ids: [1, 2, 3, 4, 5, 6]
    inp[0] = 1; inp[1] = 2; inp[2] = 3;
    inp[3] = 4; inp[4] = 5; inp[5] = 6;

    // Initialize wte (token embeddings) with sequential values
    for (int i = 0; i < V * C; i++) {
        wte[i] = i * 0.1f + 1.0f;
    }

    // Initialize wpe (position embeddings) with sequential values
    for (int i = 0; i < T * C; i++) {
        wpe[i] = i * 0.01f + 0.5f;
    }

    // Run encoder_forward
    encoder_forward(out, inp, wte, wpe, B, T, C);

    // Print first few outputs for verification
    printf("First output values:\n");
    printf("  out[0] = %f\n", out[0]);
    printf("  out[1] = %f\n", out[1]);
    printf("  out[2] = %f\n", out[2]);
    printf("  out[3] = %f\n", out[3]);

    // Clean up
    free(out);
    free(inp);
    free(wte);
    free(wpe);

    printf("encoder_forward test completed!\n");
    return 0;
}
