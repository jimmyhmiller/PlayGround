// The hand-written C twin of examples/float_bench.tal: malloc a 50M-element
// double buffer, fill with 1.0, sum, free, print. Build with -O2.
#include <stdio.h>
#include <stdlib.h>

#define N 50000000ll

int main(void) {
    double *a = malloc(N * sizeof(double));
    for (long long i = 0; i < N; i++) a[i] = 1.0;
    double acc = 0.0;
    for (long long i = 0; i < N; i++) acc += a[i];
    free(a);
    printf("%lld\n", (long long)acc);
    return 0;
}
