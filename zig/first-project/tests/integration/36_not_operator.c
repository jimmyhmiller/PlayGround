#include <stdio.h>

typedef struct {
    long long result1;
    long long result2;
} Namespace_user;

Namespace_user g_user;


void init_namespace_user(Namespace_user* ns) {
    ns->result1 = ((!(1)) ? 1 : 0);
    ns->result2 = ((!(0)) ? 1 : 0);
}

int main() {
    init_namespace_user(&g_user);
    printf(((const char*)"%d %d\n"), g_user.result1, g_user.result2);
    return 0;
}
