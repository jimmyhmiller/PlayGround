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
    printf("Hello, World!\n");
    return 0;
}
int main() {
    init_namespace_user(&g_user);
    g_user.main_fn();
    return 0;
}
