#include "namespace.h"
#include <stdio.h>
#include <stdlib.h>

// Example reloadable function
int add_numbers(int a, int b) {
    printf("Version 1: Adding %d + %d\n", a, b);
    return a + b;
}

// Example struct
typedef struct Point {
    int x;
    int y;
} Point;

Point* create_point(int x, int y) {
    Point *p = malloc(sizeof(Point));
    p->x = x;
    p->y = y;
    return p;
}

void print_point(Point *p) {
    printf("Point(%d, %d)\n", p->x, p->y);
}

// Bundle initialization - registers definitions
void bundle_init(Namespace *ns) {
    Definition *def_add = namespace_lookup(ns, "add_numbers");
    if (def_add) {
        definition_update(def_add, add_numbers);
    } else {
        namespace_define(ns, "add_numbers", DEF_FUNCTION, add_numbers);
    }

    Definition *def_create = namespace_lookup(ns, "create_point");
    if (def_create) {
        definition_update(def_create, create_point);
    } else {
        namespace_define(ns, "create_point", DEF_FUNCTION, create_point);
    }

    Definition *def_print = namespace_lookup(ns, "print_point");
    if (def_print) {
        definition_update(def_print, print_point);
    } else {
        namespace_define(ns, "print_point", DEF_FUNCTION, print_point);
    }
}
