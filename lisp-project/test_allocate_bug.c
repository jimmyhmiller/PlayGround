#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "stdio.h"
#include "stdlib.h"

// Local type definitions
typedef struct Point Point;
struct Point {
    int32_t x;
    int32_t y;
};


typedef struct {
    int32_t (*test_allocate)();
} Namespace_test_allocate;

Namespace_test_allocate g_test_allocate;

static int32_t test_allocate();

void init_namespace_test_allocate(Namespace_test_allocate* ns) {
    ns->test_allocate = &test_allocate;
}

static int32_t test_allocate() {
    return ({ Point p = (Point){42, 100}; ({ Point* p_ptr = (Point*)({ Point* __tmp_ptr = malloc(sizeof(Point)); *__tmp_ptr = p; __tmp_ptr; }); ({ int32_t x_val = p_ptr->x; printf("x value: %d (expected 42)\n", x_val); x_val; }); }); });
}
int main() {
    init_namespace_test_allocate(&g_test_allocate);
    // namespace test-allocate
    g_test_allocate.test_allocate();
    return 0;
}
