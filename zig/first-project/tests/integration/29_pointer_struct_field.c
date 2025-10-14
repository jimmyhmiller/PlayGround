#include <stdio.h>
#include <stdlib.h>

typedef struct {
    long long x;
    long long y;
} Point;


typedef struct {
    Point p;
    Point* ptr;
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
    g_user.ptr = ({ Point* __tmp_ptr = malloc(sizeof(Point)); *__tmp_ptr = g_user.p; __tmp_ptr; });
    g_user.x_val = g_user.ptr->x;
    g_user.ptr->y = 99;
    g_user.y_val = g_user.ptr->y;
    /* unsupported */;
    g_user.result = (g_user.x_val + g_user.y_val);
    printf("%lld\n", g_user.result);
    return 0;
}
