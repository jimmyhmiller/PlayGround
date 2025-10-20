#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "stdio.h"
#include "stdlib.h"

typedef struct {
    int32_t (*test_allocate_opaque)();
} Namespace_test_allocate_opaque;

Namespace_test_allocate_opaque g_test_allocate_opaque;

static int32_t test_allocate_opaque();

void init_namespace_test_allocate_opaque(Namespace_test_allocate_opaque* ns) {
    ns->test_allocate_opaque = &test_allocate_opaque;
}

static int32_t test_allocate_opaque() {
    return ({ OpaqueState state = create_opaque_state(); ({ OpaqueState* state_ptr = (OpaqueState*)({ OpaqueState* __tmp_ptr = malloc(sizeof(OpaqueState)); __tmp_ptr; }); printf("If this compiles without the copy, the bug is reproduced\n"); 0; }); });
}
int main() {
    init_namespace_test_allocate_opaque(&g_test_allocate_opaque);
    // namespace test-allocate-opaque
    g_test_allocate_opaque.test_allocate_opaque();
    return 0;
}
