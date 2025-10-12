#include <stdio.h>

typedef struct {
    long long (*apply_twice)(long long (*)(long long), long long);
    long long (*add_one)(long long);
    long long result;
} Namespace_user;

Namespace_user g_user;

static long long apply_twice(long long (*)(long long), long long);
static long long add_one(long long);

void init_namespace_user(Namespace_user* ns) {
    ns->apply_twice = &apply_twice;
    ns->add_one = &add_one;
    ns->result = ns->apply_twice(ns->add_one, 10);
}

static long long apply_twice(long long (*f)(long long), long long x) {
    return f(f(x));
}
static long long add_one(long long n) {
    return (n + 1);
}
int main() {
    init_namespace_user(&g_user);
    printf(((const char*)"%lld\n"), g_user.result);
    return 0;
}
