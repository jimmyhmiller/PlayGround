#include "namespace.h"
#include "bundle.h"
#include <stdio.h>

typedef int (*CalculateAreaFn)(int, int);

int main() {
    Namespace *ns = namespace_create("main");

    // Load utility bundle first
    Bundle *util = bundle_load("./util_bundle.so", ns);
    if (!util) {
        fprintf(stderr, "Failed to load util bundle\n");
        return 1;
    }

    // Load dependent bundle
    Bundle *dependent = bundle_load("./dependent_bundle.so", ns);
    if (!dependent) {
        fprintf(stderr, "Failed to load dependent bundle\n");
        return 1;
    }

    printf("=== Cross-bundle call demo ===\n");

    // Call calculate_area which internally calls multiply from util bundle
    Definition *area_def = namespace_lookup(ns, "calculate_area");
    if (area_def) {
        CalculateAreaFn calc = (CalculateAreaFn)definition_get(area_def);
        int result = calc(5, 3);
        printf("Area result: %d\n", result);
    }

    // Cleanup
    bundle_unload(dependent);
    bundle_unload(util);
    namespace_destroy(ns);

    return 0;
}
