// Direct comparison test - just run forward pass
#include "train_gpt2.c"

int main() {
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
    
    // Our test input: [15496, 995, 318]
    int B = 1, T = 3;
    int x[] = {15496, 995, 318};
    int y[] = {0, 0, 0}; // dummy targets
    
    // Run forward pass
    gpt2_forward(&model, x, y, B, T);
    
    // Print logits for last position
    printf("Input tokens: [15496, 995, 318]\n");
    printf("Logits for next token (first 10):\n");
    
    int last_pos = (T-1) * model.config.padded_vocab_size;
    for (int i = 0; i < 10; i++) {
        printf("  token[%d]: logit = %.6f\n", i, model.acts.logits[last_pos + i]);
    }
    
    // Get probabilities
    float* probs = model.acts.probs;
    printf("\nProbabilities for next token (first 10):\n");
    for (int i = 0; i < 10; i++) {
        printf("  token[%d]: prob = %.6f\n", i, probs[last_pos + i]);
    }
    
    // Find argmax
    int max_idx = 0;
    float max_prob = 0.0f;
    for (int i = 0; i < model.config.vocab_size; i++) {
        if (probs[last_pos + i] > max_prob) {
            max_prob = probs[last_pos + i];
            max_idx = i;
        }
    }
    printf("\nPredicted next token: %d (prob = %.6f)\n", max_idx, max_prob);
    
    gpt2_free(&model);
    return 0;
}
