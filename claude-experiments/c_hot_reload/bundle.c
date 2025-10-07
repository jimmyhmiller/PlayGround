#define _POSIX_C_SOURCE 200809L
#include "bundle.h"
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

Bundle* bundle_load(const char *path, Namespace *ns) {
    Bundle *bundle = malloc(sizeof(Bundle));
    bundle->path = strdup(path);
    bundle->ns = ns;
    bundle->handle = NULL;

    if (bundle_reload(bundle) != 0) {
        free(bundle->path);
        free(bundle);
        return NULL;
    }

    return bundle;
}

int bundle_reload(Bundle *bundle) {
    // Close old handle first to allow new version to be loaded
    if (bundle->handle) {
        dlclose(bundle->handle);
        bundle->handle = NULL;
    }

    void *new_handle = dlopen(bundle->path, RTLD_NOW | RTLD_LOCAL);
    if (!new_handle) {
        fprintf(stderr, "Failed to load bundle: %s\n", dlerror());
        return -1;
    }

    // Look for init function
    BundleInitFn init = (BundleInitFn)dlsym(new_handle, "bundle_init");
    if (!init) {
        fprintf(stderr, "No bundle_init found: %s\n", dlerror());
        dlclose(new_handle);
        return -1;
    }

    // Call init to register/update definitions
    init(bundle->ns);

    bundle->handle = new_handle;
    printf("Bundle loaded/reloaded: %s\n", bundle->path);
    return 0;
}

void bundle_unload(Bundle *bundle) {
    if (bundle->handle) {
        dlclose(bundle->handle);
    }
    free(bundle->path);
    free(bundle);
}
