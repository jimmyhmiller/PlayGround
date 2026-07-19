// The hand-written C twin of examples/qsort.tal: LCG fill (same constants),
// in-place Lomuto quicksort, sorted-check + checksum. Build with -O2.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define N 1000000ll

static void swap_(uint64_t *a, long long i, long long j) {
    uint64_t t = a[i]; a[i] = a[j]; a[j] = t;
}
static long long part_(uint64_t *a, long long lo, long long hi) {
    uint64_t pv = a[hi];
    long long j = lo;
    for (long long i = lo; i < hi; i++)
        if (a[i] < pv) { swap_(a, i, j); j++; }
    swap_(a, j, hi);
    return j;
}
static void qsort_(uint64_t *a, long long lo, long long hi) {
    if (lo < hi) {
        long long p = part_(a, lo, hi);
        if (p > 0) qsort_(a, lo, p - 1);
        qsort_(a, p + 1, hi);
    }
}
int main(void) {
    uint64_t *a = malloc(N * sizeof(uint64_t));
    uint64_t x = 88172645463325252ull;
    for (long long i = 0; i < N; i++) {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        a[i] = x % 1000000;
    }
    qsort_(a, 0, N - 1);
    uint64_t ok = 1, acc = 0, prev = 0;
    for (long long i = 0; i < N; i++) { ok &= prev <= a[i]; acc += a[i]; prev = a[i]; }
    free(a);
    printf("%llu\n", (unsigned long long)(ok * acc));
    return 0;
}
