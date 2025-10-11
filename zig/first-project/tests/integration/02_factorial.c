#include <stdio.h>

typedef struct {
    long long (*factorial)(long long);
    long long result;
} Namespace_user;

Namespace_user g_user;

static long long factorial(long long);

void init_namespace_user(Namespace_user* ns) {
    ns->factorial = &factorial;
    ns->result = ns->factorial(5);
}

static long long factorial(long long n) {
    return ((n == 0) ? 1 : (n * factorial((n - 1))));
}
int main() {
    init_namespace_user(&g_user);
    printf(((const char*)"%lld\n"), g_user.result);
    return 0;
}
