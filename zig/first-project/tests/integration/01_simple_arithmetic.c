#include <stdio.h>

typedef struct {
    long long result;
} Namespace_user;

Namespace_user g_user;


void init_namespace_user(Namespace_user* ns) {
    ns->result = (2 + 3);
}

int main() {
    init_namespace_user(&g_user);
    printf(((const char*)"%lld\n"), g_user.result);
    return 0;
}
