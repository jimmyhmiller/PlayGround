#include <stdio.h>

typedef struct {
    long long x;
    long long y;
} Point;

typedef enum {
    Color_Red,
    Color_Green,
    Color_Blue,
} Color;


typedef struct {
    long long value;
    long long (*add_one)(long long);
    long long (*add)(long long, long long);
    Point (*make_point)(long long, long long);
    long long (*get_red_value)();
    long long magic_number;
} Namespace_math_utils;

Namespace_math_utils g_math_utils;

static long long add_one(long long);
static long long add(long long, long long);
static Point make_point(long long, long long);
static long long get_red_value();

void init_namespace_math_utils(Namespace_math_utils* ns) {
    ns->value = 42;
    ns->add_one = &add_one;
    ns->add = &add;
    ns->make_point = &make_point;
    ns->get_red_value = &get_red_value;
    ns->magic_number = 7;
}

static long long add_one(long long x) {
    return (x + 1);
}
static long long add(long long a, long long b) {
    return (a + b);
}
static Point make_point(long long x, long long y) {
    return (Point){x, y};
}
static long long get_red_value() {
    return 255;
}
void lisp_main(void) {
    init_namespace_math_utils(&g_math_utils);
}
