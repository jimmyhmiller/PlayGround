#include "namespace.h"
#include <stdio.h>
#include <stdlib.h>

// Utility function that other bundles can call
int multiply(int a, int b) {
    printf("Utility: Multiplying %d * %d\n", a, b);
    return a * b;
}

void bundle_init(Namespace *ns) {
    Definition *def = namespace_lookup(ns, "multiply");
    if (def) {
        definition_update(def, multiply);
    } else {
        namespace_define(ns, "multiply", DEF_FUNCTION, multiply);
    }
}
