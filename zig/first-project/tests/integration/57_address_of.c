#include <stdio.h>

typedef struct {
    long long x;
    long long* ptr;
    long long val;
} Namespace_user;

Namespace_user g_user;


void init_namespace_user(Namespace_user* ns) {
    ns->x = 100;
    ns->ptr = (&ns->x);
    ns->val = (*ns->ptr);
}

int main() {
    init_namespace_user(&g_user);
    printf("%lld\n", g_user.val);
    return 0;
}
