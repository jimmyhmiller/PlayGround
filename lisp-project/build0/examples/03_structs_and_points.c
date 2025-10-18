#include <stdio.h>
#include <stdint.h>

#include "stdio.h"
typedef struct {
    int32_t x;
    int32_t y;
} Point;

typedef struct {
    Point start;
    Point end;
} Line;


typedef struct {
    int32_t (*manhattan_distance)(Point, Point);
    int32_t (*main_fn)();
} Namespace_user;

Namespace_user g_user;

static int32_t manhattan_distance(Point, Point);
static int32_t main_fn();

void init_namespace_user(Namespace_user* ns) {
    ns->manhattan_distance = &manhattan_distance;
    ns->main_fn = &main_fn;
}

static int32_t manhattan_distance(Point p1, Point p2) {
    return ({ int32_t dx = (p1.x - p2.x); ({ int32_t dy = (p1.y - p2.y); (((dx < 0) ? (0 - dx) : dx) + ((dy < 0) ? (0 - dy) : dy)); }); });
}
static int32_t main_fn() {
    return ({ Point origin = (Point){0, 0}; ({ Point p1 = (Point){3, 4}; ({ Line line = (Line){origin, p1}; printf("Origin: (%d, %d)\n", origin.x, origin.y); printf("Point 1: (%d, %d)\n", p1.x, p1.y); printf("Line start: (%d, %d)\n", line.start.x, line.start.y); printf("Line end: (%d, %d)\n", line.end.x, line.end.y); printf("Manhattan distance: %d\n", g_user.manhattan_distance(origin, p1)); 0; }); }); });
}
int main() {
    init_namespace_user(&g_user);
    g_user.main_fn();
    return 0;
}
