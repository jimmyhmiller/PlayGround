#include <stdio.h>

typedef struct {
    long long x;
    long long y;
} Point;


typedef struct {
    Point p;
    long long x_val;
    long long y_val;
    long long result;
} Namespace_user;

Namespace_user g_user;


void init_namespace_user(Namespace_user* ns) {
}

int main() {
    init_namespace_user(&g_user);
    g_user.p = (Point){10, 20};
    g_user.x_val = g_user.p.x;
    g_user.y_val = g_user.p.y;
    g_user.result = (g_user.x_val + g_user.y_val);
    printf("%lld\n", g_user.result);
    return 0;
}
