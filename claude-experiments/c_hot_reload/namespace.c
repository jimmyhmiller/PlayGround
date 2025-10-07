#define _POSIX_C_SOURCE 200809L
#include "namespace.h"
#include <stdlib.h>
#include <string.h>

Namespace* namespace_create(const char *name) {
    Namespace *ns = malloc(sizeof(Namespace));
    ns->name = strdup(name);
    ns->definitions = NULL;
    return ns;
}

void namespace_destroy(Namespace *ns) {
    Definition *def = ns->definitions;
    while (def) {
        Definition *next = def->next;
        free(def->name);
        free(def);
        def = next;
    }
    free(ns->name);
    free(ns);
}

Definition* namespace_define(Namespace *ns, const char *name, DefType type, void *initial_ptr) {
    Definition *def = malloc(sizeof(Definition));
    def->name = strdup(name);
    def->type = type;
    atomic_init(&def->ptr, initial_ptr);
    def->next = ns->definitions;
    ns->definitions = def;
    return def;
}

Definition* namespace_lookup(Namespace *ns, const char *name) {
    Definition *def = ns->definitions;
    while (def) {
        if (strcmp(def->name, name) == 0) {
            return def;
        }
        def = def->next;
    }
    return NULL;
}

void definition_update(Definition *def, void *new_ptr) {
    atomic_store(&def->ptr, new_ptr);
}

void* definition_get(Definition *def) {
    return atomic_load(&def->ptr);
}
