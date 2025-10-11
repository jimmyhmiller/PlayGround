#include <stdio.h>

typedef struct {
    long long counter;
    long long sum;
} Namespace_user;

Namespace_user g_user;


void init_namespace_user(Namespace_user* ns) {
    ns->counter = 0;
    ns->sum = 0;
}

int main() {
    init_namespace_user(&g_user);
    ((g_user.counter < 5) ? /* unsupported: (while (< counter 5) (set! sum (+ sum counter)) (set! counter (+ counter 1))) */;
    printf("%lld\n", g_user.sum);
    return 0;
}
