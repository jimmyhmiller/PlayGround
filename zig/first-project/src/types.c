#include <stdio.h>

// Local type definitions
typedef struct my_type my_type;
struct my_type {
    long long x;
    long long y;
};


typedef struct {
    long long foo;
} Namespace_types;

Namespace_types g_types;


void init_namespace_types(Namespace_types* ns) {
    ns->foo = 123;
}

void lisp_main(void) {
    init_namespace_types(&g_types);
}
