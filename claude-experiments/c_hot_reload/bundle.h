#ifndef BUNDLE_H
#define BUNDLE_H

#include "namespace.h"

typedef struct Bundle {
    void *handle;  // dlopen handle
    char *path;
    Namespace *ns;
} Bundle;

// Bundle API
Bundle* bundle_load(const char *path, Namespace *ns);
int bundle_reload(Bundle *bundle);
void bundle_unload(Bundle *bundle);

// Bundle initialization function type
typedef void (*BundleInitFn)(Namespace *ns);

#endif
