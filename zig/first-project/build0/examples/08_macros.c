#include <stdio.h>
#include <stdint.h>

#include "stdio.h"

typedef struct {
    int32_t (*main_fn)();
} Namespace_user;

Namespace_user g_user;

static int32_t main_fn();

void init_namespace_user(Namespace_user* ns) {
    ns->main_fn = &main_fn;
}

static int32_t main_fn() {
    ({ int32_t x = (5 + 1); printf("add1(5) = %d\n", x); });
    ({ int32_t y = (7 * 7); printf("square(7) = %d\n", y); });
    ({ int32_t z = ((10 + 1) + 1); printf("double-add1(10) = %d\n", z); });
    ({ int32_t result = ((3 + 1) + (4 * 4)); printf("(add1 3) + (square 4) = %d\n", result); });
    return 0;
}
int main() {
    init_namespace_user(&g_user);
    g_user.main_fn();
    return 0;
}
