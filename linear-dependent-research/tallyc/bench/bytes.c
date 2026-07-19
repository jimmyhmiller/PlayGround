// The hand-written C twin of examples/bytes_bench.tal: malloc a 100 MB byte
// buffer (`unsigned char`, one byte per element), fill it, overwrite one slot,
// sum every byte, free, print. Build with -O2.
#include <stdio.h>
#include <stdlib.h>

#define N 100000000ll

int main(void) {
    unsigned char *a = malloc(N);          // 100 MB — one byte per element
    for (long long i = 0; i < N; i++) a[i] = 1;
    a[1234567] = 42;
    long long acc = 0;
    for (long long i = 0; i < N; i++) acc += a[i];
    free(a);
    printf("%lld\n", acc);
    return 0;
}
