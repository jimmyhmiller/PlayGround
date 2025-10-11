#include <stdio.h>

typedef struct {
    long long counter;
} Namespace_user;

Namespace_user g_user;


void init_namespace_user(Namespace_user* ns) {
    ns->counter = 0;
}

int main() {
    init_namespace_user(&g_user);
    ({ for (long long i__0 = 0; (i__0 < 5); i__0 = (i__0 + 1)) { g_user.counter = (g_user.counter + 1); } });
    printf(((const char*)"%lld\n"), g_user.counter);
    return 0;
}
