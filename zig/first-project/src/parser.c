#include <stdio.h>

// Required namespace: types
typedef struct my_type my_type;
struct my_type {
    long long x;
    long long y;
};

typedef struct {
    long long foo;
} Namespace_types;

extern Namespace_types g_types;
void init_namespace_types(Namespace_types* ns);


typedef struct {
    long long (*parse_fn)();
} Namespace_src_parser;

Namespace_src_parser g_src_parser;

static long long parse_fn();

void init_namespace_src_parser(Namespace_src_parser* ns) {
    init_namespace_types(&g_types);
    ns->parse_fn = &parse_fn;
}

static long long parse_fn() {
    return 42;
}
void lisp_main(void) {
    init_namespace_types(&g_types);
    init_namespace_src_parser(&g_src_parser);
}
