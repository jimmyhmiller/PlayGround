/* A C `main` that calls Coil functions which pass/return structs by value, to
 * prove the interop works in the *other* direction: a C caller relies on Coil's
 * emitted code honoring the platform struct ABI. */
#include <stdint.h>

typedef struct {
    int64_t x;
    int64_t y;
} Point; /* 16 bytes */

typedef struct {
    int64_t a;
    int64_t b;
    int64_t c;
} Triple; /* 24 bytes (indirect) */

/* Defined in Coil. */
Point coil_make_point(int64_t x, int64_t y);
Triple coil_make_triple(int64_t a, int64_t b, int64_t c);

int main(void) {
    Point p = coil_make_point(30, 12); /* {30,12} -> 42 */
    Triple t = coil_make_triple(10, 20, 12); /* {10,20,12} -> 42 */
    return (int)(p.x + p.y + (t.a + t.b + t.c) - 42); /* 42 + 0 = 42 */
}
