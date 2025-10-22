#include <stdio.h>

// Required namespace: math.utils
typedef struct Point Point;
typedef enum {
    Color_Red,
    Color_Green,
    Color_Blue,
} Color;

struct Point {
    long long x;
    long long y;
};

typedef struct {
    long long value;
    long long (*add_one)(long long);
    long long (*add)(long long, long long);
    Point (*make_point)(long long, long long);
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
    ns->double_value = (g_math_utils.value * 2);
    ns->add_two = &add_two;
}

static long long add_two(long long x) {
    return g_math_utils.add_one(g_math_utils.add_one(x));
}

// Built-in argc/argv globals
int lisp_argc;
char** lisp_argv;

void lisp_main(void) {
    init_namespace_math_utils(&g_math_utils);
    init_namespace_math_core(&g_math_core);
}
