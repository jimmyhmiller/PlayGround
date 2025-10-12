#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

typedef struct {
    long long x;
    long long result;
} Namespace_user;

Namespace_user g_user;


void init_namespace_user(Namespace_user* ns) {
    ns->x = 100;
    ns->result = ({ long long x = 10; ({ long long x = 20; x; }); });
}

int main() {
    init_namespace_user(&g_user);
    printf(((const char*)"%lld\n"), g_user.result);
    return 0;
}
