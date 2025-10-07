#include "namespace.h"
#include "bundle.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// Type-safe function pointer getter
typedef int (*AddFn)(int, int);

int main() {
    // Create namespace
    Namespace *ns = namespace_create("main");

    // Load initial bundle
    Bundle *bundle = bundle_load("./example_bundle.so", ns);
    if (!bundle) {
        fprintf(stderr, "Failed to load bundle\n");
        return 1;
    }

    printf("=== Initial run ===\n");

    // Look up and call function through indirection
    Definition *add_def = namespace_lookup(ns, "add_numbers");
    if (add_def) {
        AddFn add = (AddFn)definition_get(add_def);
        int result = add(5, 3);
        printf("Result: %d\n\n", result);
    }

    printf("Watching for bundle changes. Modify example_bundle.c and rebuild.\n");
    printf("Press Ctrl+C to exit.\n\n");

    // Simulate hot reload loop
    int counter = 0;
    while (1) {
        sleep(2);
        counter++;

        // Try to reload bundle
        if (bundle_reload(bundle) == 0) {
            printf("=== After reload #%d ===\n", counter);

            // Call function again - gets latest version
            Definition *add_def = namespace_lookup(ns, "add_numbers");
            if (add_def) {
                AddFn add = (AddFn)definition_get(add_def);
                int result = add(10, 7);
                printf("Result: %d\n\n", result);
            }
        }
    }

    // Cleanup
    bundle_unload(bundle);
    namespace_destroy(ns);

    return 0;
}
