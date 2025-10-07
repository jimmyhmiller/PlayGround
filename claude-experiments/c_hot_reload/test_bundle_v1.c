#include "namespace.h"

int test_add(int a, int b) {
    return a + b;
}

int test_multiply(int a, int b) {
    return a * b;
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
