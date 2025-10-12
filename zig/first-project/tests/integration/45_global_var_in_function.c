#include <stdio.h>
#include <stdint.h>

typedef struct {
    int32_t counter;
    int32_t (*increment)();
    int32_t (*get_counter)();
    int32_t (*test)();
} Namespace_user;

Namespace_user g_user;

static int32_t increment();
static int32_t get_counter();
static int32_t test();

void init_namespace_user(Namespace_user* ns) {
    ns->counter = 0;
    ns->increment = &increment;
    ns->get_counter = &get_counter;
    ns->test = &test;
}

static int32_t increment() {
    g_user.counter = (g_user.counter + 1);
    return g_user.counter;
}
static int32_t get_counter() {
    return g_user.counter;
}
static int32_t test() {
    g_user.increment();
    g_user.increment();
    return g_user.get_counter();
}
int main() {
    init_namespace_user(&g_user);
    g_user.test();
    return 0;
}
