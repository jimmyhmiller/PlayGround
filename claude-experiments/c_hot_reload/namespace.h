#ifndef NAMESPACE_H
#define NAMESPACE_H

#include <stdatomic.h>
#include <stddef.h>

// Definition types
typedef enum {
    DEF_FUNCTION,
    DEF_STRUCT,
    DEF_ENUM,
    DEF_VAR
} DefType;

// Generic definition container
typedef struct Definition {
    char *name;
    DefType type;
    _Atomic(void*) ptr;  // Atomic pointer to actual definition
    struct Definition *next;
} Definition;

// Namespace structure
typedef struct Namespace {
    char *name;
    Definition *definitions;
} Namespace;

// Namespace API
Namespace* namespace_create(const char *name);
void namespace_destroy(Namespace *ns);

// Definition management
Definition* namespace_define(Namespace *ns, const char *name, DefType type, void *initial_ptr);
Definition* namespace_lookup(Namespace *ns, const char *name);
void definition_update(Definition *def, void *new_ptr);
void* definition_get(Definition *def);

#endif
