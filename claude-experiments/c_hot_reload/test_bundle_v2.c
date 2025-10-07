#include "namespace.h"

// Version 2: changed implementations
int test_add(int a, int b) {
    return a + b + 100;  // Changed!
}

int test_multiply(int a, int b) {
    return a * b * 2;  // Changed!
}

void bundle_init(Namespace *ns) {
    Definition *def_add = namespace_lookup(ns, "test_add");
    if (def_add) {
        definition_update(def_add, test_add);
    } else {
        namespace_define(ns, "test_add", DEF_FUNCTION, test_add);
    }

    Definition *def_mul = namespace_lookup(ns, "test_multiply");
    if (def_mul) {
        definition_update(def_mul, test_multiply);
    } else {
        namespace_define(ns, "test_multiply", DEF_FUNCTION, test_multiply);
    }
}
