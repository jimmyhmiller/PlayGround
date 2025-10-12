#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

typedef struct {
    long long x;
    long long y;
} Point;


typedef struct {
    Point (*make_point)(long long, long long);
    long long (*distance_squared)(Point);
    Point p;
    long long result;
} Namespace_user;

Namespace_user g_user;

static Point make_point(long long, long long);
static long long distance_squared(Point);

void init_namespace_user(Namespace_user* ns) {
}

static Point make_point(long long x, long long y) {
    return (Point){x, y};
}
static long long distance_squared(Point p) {
    return ({ long long px = p.x; ({ long long py = p.y; ((px * px) + (py * py)); }); });
}
int main() {
    init_namespace_user(&g_user);
    g_user.make_point = make_point;
    g_user.distance_squared = distance_squared;
    g_user.p = g_user.make_point(3, 4);
    g_user.result = g_user.distance_squared(g_user.p);
    printf(((const char*)"%lld\n"), g_user.result);
    return 0;
}
