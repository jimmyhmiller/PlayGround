#include <stdio.h>

typedef enum {
    Color_Red,
    Color_Green,
    Color_Blue,
} Color;


typedef struct {
    long long (*get_color_value)(Color);
    long long result1;
    long long result2;
    long long result3;
} Namespace_test;

Namespace_test g_test;

static long long get_color_value(Color);

void init_namespace_test(Namespace_test* ns) {
}

static long long get_color_value(Color c) {
    return ((c == Color_Red) ? 1 : ((c == Color_Green) ? 2 : 3));
}
int main() {
    init_namespace_test(&g_test);
    // namespace test
    g_test.get_color_value = get_color_value;
    g_test.result1 = g_test.get_color_value(Color_Red);
    g_test.result2 = g_test.get_color_value(Color_Green);
    g_test.result3 = g_test.get_color_value(Color_Blue);
    printf(((const char*)"%lld %lld %lld\n"), g_test.result1, g_test.result2, g_test.result3);
    return 0;
}
