#include <stdio.h>

typedef enum {
    Color_Red,
    Color_Green,
    Color_Blue,
} Color;

typedef struct {
    long long x;
    long long y;
    Color color;
} ColoredPoint;


typedef struct {
    ColoredPoint p;
    long long px;
    long long py;
    Color pc;
    long long color_val;
    long long result;
} Namespace_test;

Namespace_test g_test;


void init_namespace_test(Namespace_test* ns) {
    ns->p = (ColoredPoint){10, 20, Color_Blue};
    ns->px = ns->p.x;
    ns->py = ns->p.y;
    ns->pc = ns->p.color;
    ns->color_val = ((ns->pc == Color_Blue) ? 3 : 0);
    ns->result = (ns->px + (ns->py + ns->color_val));
}

int main() {
    init_namespace_test(&g_test);
    // namespace test
    printf(((const char*)"%lld\n"), g_test.result);
    return 0;
}
