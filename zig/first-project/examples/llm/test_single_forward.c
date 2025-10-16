#include <stdio.h>
#include <stdint.h>

#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "time.h"

typedef struct {
    int32_t (*main_fn)();
} Namespace_user;

Namespace_user g_user;

static int32_t main_fn();

void init_namespace_user(Namespace_user* ns) {
    ns->main_fn = &main_fn;
}

static int32_t main_fn() {
    ({ int64_t start = clock(); printf("Simulating forward pass...\n"); ({ int64_t end = clock(); printf("Time: %ld clock ticks\n", (end - start)); }); });
    return 0;
}
int main() {
    init_namespace_user(&g_user);
    g_user.main_fn();
    return 0;
}
