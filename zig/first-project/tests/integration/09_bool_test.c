#include <stdio.h>
#include <stdbool.h>

typedef struct {
    bool x;
    bool y;
    long long result;
} Namespace_user;

Namespace_user g_user;


void init_namespace_user(Namespace_user* ns) {
    ns->x = 1;
    ns->y = 0;
    ns->result = (ns->x ? 1 : 0);
}

int main() {
    init_namespace_user(&g_user);
    printf(((const char*)"%lld\n"), g_user.result);
    return 0;
}
