#include <stdio.h>
#include <stdint.h>

#include "stdio.h"
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
    long long imported_value;
    long long incremented;
    long long sum;
    long long composed;
    long long red_value;
    long long magic;
    int32_t (*main_fn)();
} Namespace_example_modules;

Namespace_example_modules g_example_modules;

static int32_t main_fn();

void init_namespace_example_modules(Namespace_example_modules* ns) {
    ns->imported_value = g_math_utils.value;
    ns->incremented = g_math_utils.add_one(10);
    ns->sum = g_math_utils.add(5, 7);
    ns->composed = g_math_utils.add_one(g_math_utils.add(ns->imported_value, ns->incremented));
    ns->red_value = g_math_utils.get_red_value();
    ns->magic = g_math_utils.magic_number;
    ns->main_fn = &main_fn;
}

static int32_t main_fn() {
    printf("=== Module System Demo ===\n");
    printf("Imported value: %lld\n", g_example_modules.imported_value);
    printf("Function result: %lld\n", g_example_modules.incremented);
    printf("Sum: %lld\n", g_example_modules.sum);
    printf("Composed: %lld\n", g_example_modules.composed);
    printf("Red value: %lld\n", g_example_modules.red_value);
    printf("Magic number: %lld\n", g_example_modules.magic);
    return 0;
}
int main() {
    init_namespace_math_utils(&g_math_utils);
    init_namespace_example_modules(&g_example_modules);
    // namespace example.modules
    // require [math.utils :as mu]
    g_example_modules.main_fn();
    return 0;
}
