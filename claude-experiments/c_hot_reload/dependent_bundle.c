#include "namespace.h"
#include <stdio.h>

// This bundle uses functions from other namespaces
typedef int (*MultiplyFn)(int, int);

int calculate_area(Namespace *ns, int width, int height) {
    // Look up multiply from the namespace
    Definition *multiply_def = namespace_lookup(ns, "multiply");
    if (!multiply_def) {
        printf("Error: multiply not found in namespace\n");
        return -1;
    }

    // Get the function pointer and call it
    MultiplyFn multiply = (MultiplyFn)definition_get(multiply_def);
    return multiply(width, height);
}

// Bundle needs namespace passed to use cross-bundle calls
static Namespace *g_namespace = NULL;

int area_wrapper(int w, int h) {
    return calculate_area(g_namespace, w, h);
}

void bundle_init(Namespace *ns) {
    g_namespace = ns;  // Store namespace for later use

    Definition *def = namespace_lookup(ns, "calculate_area");
    if (def) {
        definition_update(def, area_wrapper);
    } else {
        namespace_define(ns, "calculate_area", DEF_FUNCTION, area_wrapper);
    }
}
