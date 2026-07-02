// The hand-written C twin of examples/arr_bench.tal: malloc a 10M-element
// array, fill it, overwrite one slot, sum, free, print. Build with -O2.
#include <stdio.h>
#include <stdlib.h>

#define N 10000000ll

int main(void) {
    long long *a = malloc(N * sizeof(long long));
    for (long long i = 0; i < N; i++) a[i] = 1;
    a[1234567] = 42;
    long long acc = 0;
    for (long long i = 0; i < N; i++) acc += a[i];
    free(a);
    printf("%lld\n", acc);
    return 0;
}
