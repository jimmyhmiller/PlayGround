#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    // Simulate allocating activation tensors for B=1, T=103, C=768, L=12, NH=12, Vp=50304
    int B = 1, T = 103, C = 768, L = 12, NH = 12, Vp = 50304;
    
    clock_t start = clock();
    
    // Calculate sizes (same as in the Lisp code)
    int att_size = ((((L * B) * NH) * T) * T);  // This is HUGE
    float* att = malloc(att_size * sizeof(float));
    
    clock_t end = clock();
    printf("Allocated %d MB for attention\n", (att_size * 4) / (1024*1024));
    printf("Time: %.3f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);
    
    free(att);
    return 0;
}
