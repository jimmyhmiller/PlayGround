/* A C `main` that calls Coil functions exposed via `(export-c …)` — by their C
 * symbols (renamed with :as) — including one that takes a struct BY VALUE, which
 * exercises the C-ABI export thunk. Proves the export feature both directions:
 * C receives a struct from Coil, and C passes a struct to Coil. */
#include <stdint.h>

typedef struct { int64_t x, y; } Point;

Point   shapes_make_point(int64_t x, int64_t y);  /* returns a struct */
int64_t shapes_add(int64_t a, int64_t b);         /* scalars */
int64_t shapes_dist2(Point p);                    /* takes a struct BY VALUE (thunk) */

int main(void) {
    Point p = shapes_make_point(20, 22);          /* {20, 22} */
    int64_t s = shapes_add(40, 2);                /* 42 */
    int64_t d = shapes_dist2((Point){3, 4});      /* 9 + 16 = 25 */
    /* (20+22 - 42) + (25 - 25) = 0 on success */
    return (int)(((p.x + p.y) - s) + (d - 25));
}
