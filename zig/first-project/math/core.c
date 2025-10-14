#include <stdio.h>

// Required namespace: math.utils
typedef enum {
    Color_Red,
    Color_Green,
    Color_Blue,
} Color;

typedef struct {
    long long x;
    long long y;
} Point;

typedef struct {
    Point (*make_point)(long long, long long);
    Color Color_Blue;
    long long (*add_one)(long long);
    long long value;
    Color Color_Red;
    long long (*add)(long long, long long);
    Color Color_Green;
    long long (*get_red_value)();
    long long magic_number;
} Namespace_math_utils;

extern Namespace_math_utils g_math_utils;
void init_namespace_math_utils(Namespace_math_utils* ns);


typedef struct {
    long long double_value;
    long long (*add_two)(long long);
} Namespace_math_core;

Namespace_math_core g_math_core;

static long long add_two(long long);

void init_namespace_math_core(Namespace_math_core* ns) {
    init_namespace_math_utils(&g_math_utils);
}

static long long add_two(long long x) {
    return g_math_utils.add_one(g_math_utils.add_one(x));
}
void lisp_main(void) {
    init_namespace_math_utils(&g_math_utils);
    init_namespace_math_core(&g_math_core);
}
