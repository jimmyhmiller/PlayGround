/* C helper linked into the struct-by-value ABI tests. Each function takes or
 * returns a struct by value, so a Coil program that interoperates with it must
 * lower the System V / AAPCS64 struct ABI correctly. */
#include <stdint.h>

/* <=16 bytes: passed/returned in registers (coerced). */
typedef struct {
    int32_t a;
    int32_t b;
} Small; /* 8 bytes */

/* >16 bytes: passed indirectly (byval/sret). */
typedef struct {
    int64_t a;
    int64_t b;
    int64_t c;
} Big; /* 24 bytes */

/* A homogeneous float aggregate (HFA on AArch64; two SSE on SysV). */
typedef struct {
    float x;
    float y;
} Hfa2;

Small make_small(int32_t a, int32_t b) {
    Small s = {a, b};
    return s;
}

int32_t sum_small(Small s) {
    return s.a + s.b;
}

/* Both take and return a small struct by value (round-trip in one call). */
Small scale_small(Small s, int32_t k) {
    Small r = {s.a * k, s.b * k};
    return r;
}

Big make_big(int64_t a, int64_t b, int64_t c) {
    Big g = {a, b, c};
    return g;
}

int64_t sum_big(Big g) {
    return g.a + g.b + g.c;
}

/* Take a big struct, return a big struct (sret + byval together). */
Big add_big(Big x, Big y) {
    Big r = {x.a + y.a, x.b + y.b, x.c + y.c};
    return r;
}

Hfa2 make_hfa2(float x, float y) {
    Hfa2 h = {x, y};
    return h;
}

float sum_hfa2(Hfa2 h) {
    return h.x + h.y;
}
