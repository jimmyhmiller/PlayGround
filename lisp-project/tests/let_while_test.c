#include <stdio.h>
#include <stdint.h>

#include "stdio.h"

typedef struct {
    int32_t (*test_function_body)();
    int32_t (*test_let_body)();
    int32_t (*main_fn)();
} Namespace_user;

Namespace_user g_user;

static int32_t test_function_body();
static int32_t test_let_body();
static int32_t main_fn();

void init_namespace_user(Namespace_user* ns) {
    ns->test_function_body = &test_function_body;
    ns->test_let_body = &test_let_body;
    ns->main_fn = &main_fn;
}

static int32_t test_function_body() {
    return ({ int32_t x = 0; ({ while ((x < 3)) { x = (x + 1); } }); printf("After while: x = %d\n", x); 42; });
}
static int32_t test_let_body() {
    return ({ int32_t result = ({ int32_t x = 0; ({ while ((x < 3)) { x = (x + 1); } }); x; }); printf("Result: %d\n", result); 0; });
}
static int32_t main_fn() {
    printf("Test 1: Function body with while (should work)\n");
    g_user.test_function_body();
    printf("\nTest 2: Let body with while (should fail)\n");
    g_user.test_let_body();
    return 0;
}
int main() {
    init_namespace_user(&g_user);
    g_user.main_fn();
    return 0;
}
