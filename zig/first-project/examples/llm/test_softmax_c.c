// Reference C implementation to verify softmax_forward
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void softmax_forward(float* probs, float* logits, int B, int T, int V, int Vp) {
    // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,Vp) of the unnormalized log probabilities
    // Vp is the padded vocab size (for efficiency), V is the actual vocab size
    // example: Vp is 50304 and V is 50257
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // probs <- softmax(logits)
            float* logits_bt = logits + b * T * Vp + t * Vp;
            float* probs_bt = probs + b * T * Vp + t * Vp;

            // maxval is only calculated and subtracted for numerical stability
            float maxval = -10000.0f; // TODO something better
            for (int i = 0; i < V; i++) {
                if (logits_bt[i] > maxval) {
                    maxval = logits_bt[i];
                }
            }
            float sum = 0.0f;
            for (int i = 0; i < V; i++) {
                probs_bt[i] = expf(logits_bt[i] - maxval);
                sum += probs_bt[i];
            }
            // normalize the probabilities to sum to 1.0
            for (int i = 0; i < V; i++) {
                probs_bt[i] /= sum;
            }
            // for extra super safety we may wish to include this too,
            // forcing the probabilities here to be zero, but it shouldn't matter
            for (int i = V; i < Vp; i++) {
                probs_bt[i] = 0.0f;
            }
        }
    }
}

int main() {
    printf("Testing softmax_forward (C reference)...\n");

    // Test dimensions: B=1, T=2, V=4, Vp=8 (padded vocab)
    int B = 1, T = 2, V = 4, Vp = 8;

    // Allocate arrays
    float* probs = (float*)malloc(B * T * Vp * sizeof(float));
    float* logits = (float*)malloc(B * T * Vp * sizeof(float));

    // Initialize logits
    // First position: [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]
    // Second position: [2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0] (all equal logits)
    logits[0] = 1.0f;
    logits[1] = 2.0f;
    logits[2] = 3.0f;
    logits[3] = 4.0f;
    for (int i = 4; i < 8; i++) {
        logits[i] = 0.0f;
    }
    logits[8] = 2.0f;
    logits[9] = 2.0f;
    logits[10] = 2.0f;
    logits[11] = 2.0f;
    for (int i = 12; i < 16; i++) {
        logits[i] = 0.0f;
    }

    // Run softmax_forward
    softmax_forward(probs, logits, B, T, V, Vp);

    // Print outputs
    printf("First position probabilities:\n");
    printf("  probs[0] = %f\n", probs[0]);
    printf("  probs[1] = %f\n", probs[1]);
    printf("  probs[2] = %f\n", probs[2]);
    printf("  probs[3] = %f\n", probs[3]);

    // Verify sum
    float sum1 = probs[0] + probs[1] + probs[2] + probs[3];
    printf("  Sum = %f (should be 1.0)\n", sum1);

    printf("Second position probabilities (all equal logits):\n");
    printf("  probs[8] = %f (expected 0.25)\n", probs[8]);
    printf("  probs[9] = %f (expected 0.25)\n", probs[9]);
    printf("  probs[10] = %f (expected 0.25)\n", probs[10]);
    printf("  probs[11] = %f (expected 0.25)\n", probs[11]);

    // Verify sum
    float sum2 = probs[8] + probs[9] + probs[10] + probs[11];
    printf("  Sum = %f (should be 1.0)\n", sum2);

    // Clean up
    free(probs);
    free(logits);

    printf("softmax_forward test completed!\n");
    return 0;
}
