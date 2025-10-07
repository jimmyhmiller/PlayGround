#include "namespace.h"
#include <stdio.h>

const char* greeting() {
    return "Yo! What's up?";
}

void bundle_init(Namespace *ns) {
    Definition *def = namespace_lookup(ns, "greeting");
    if (def) {
        definition_update(def, greeting);
    } else {
        namespace_define(ns, "greeting", DEF_FUNCTION, greeting);
    }
}
