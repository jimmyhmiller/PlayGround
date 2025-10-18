#include <stdio.h>
#include <stdint.h>

#include "stdio.h"

typedef struct {
    uint32_t (*fib)(uint32_t);
    int32_t (*main_fn)();
} Namespace_user;

Namespace_user g_user;

static uint32_t fib(uint32_t);
static int32_t main_fn();

void init_namespace_user(Namespace_user* ns) {
    ns->fib = &fib;
    ns->main_fn = &main_fn;
}

static uint32_t fib(uint32_t n) {
    return ((n <= 1) ? n : (g_user.fib((n - 1)) + g_user.fib((n - 2))));
}
static int32_t main_fn() {
    printf("fib(0) = %u\n", g_user.fib(0));
    printf("fib(1) = %u\n", g_user.fib(1));
    printf("fib(5) = %u\n", g_user.fib(5));
    printf("fib(10) = %u\n", g_user.fib(10));
    printf("fib(15) = %u\n", g_user.fib(15));
    return 0;
}
int main() {
    init_namespace_user(&g_user);
    g_user.main_fn();
    return 0;
}
