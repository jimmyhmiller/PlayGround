#include "namespace.h"

int change(int current) {
    return current + 1;  // Increment by 1
}

void bundle_init(Namespace *ns) {
    Definition *def = namespace_lookup(ns, "change");
    if (def) {
        definition_update(def, change);
    } else {
        namespace_define(ns, "change", DEF_FUNCTION, change);
    }
}
