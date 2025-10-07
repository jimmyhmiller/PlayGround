#include "test.h"
#include "namespace.h"
#include <stdlib.h>

// Test namespace creation
int test_namespace_create() {
    Namespace *ns = namespace_create("test");
    TEST_ASSERT_NOT_NULL(ns, "namespace should be created");
    TEST_ASSERT_STR_EQ(ns->name, "test", "namespace name should match");
    TEST_ASSERT_NULL(ns->definitions, "definitions should be empty initially");
    namespace_destroy(ns);
    return 0;
}

// Test defining a function
int test_namespace_define() {
    Namespace *ns = namespace_create("test");

    void *dummy_fn = (void*)0x1234;
    Definition *def = namespace_define(ns, "test_func", DEF_FUNCTION, dummy_fn);

    TEST_ASSERT_NOT_NULL(def, "definition should be created");
    TEST_ASSERT_STR_EQ(def->name, "test_func", "definition name should match");
    TEST_ASSERT_EQ(def->type, DEF_FUNCTION, "definition type should be FUNCTION");
    TEST_ASSERT_EQ(definition_get(def), dummy_fn, "definition pointer should match");

    namespace_destroy(ns);
    return 0;
}

// Test lookup
int test_namespace_lookup() {
    Namespace *ns = namespace_create("test");

    void *dummy_fn = (void*)0x5678;
    namespace_define(ns, "my_func", DEF_FUNCTION, dummy_fn);

    Definition *found = namespace_lookup(ns, "my_func");
    TEST_ASSERT_NOT_NULL(found, "definition should be found");
    TEST_ASSERT_STR_EQ(found->name, "my_func", "found definition name should match");

    Definition *not_found = namespace_lookup(ns, "nonexistent");
    TEST_ASSERT_NULL(not_found, "nonexistent definition should return NULL");

    namespace_destroy(ns);
    return 0;
}

// Test atomic update
int test_definition_update() {
    Namespace *ns = namespace_create("test");

    void *old_ptr = (void*)0x1111;
    void *new_ptr = (void*)0x2222;

    Definition *def = namespace_define(ns, "test_func", DEF_FUNCTION, old_ptr);
    TEST_ASSERT_EQ(definition_get(def), old_ptr, "initial pointer should match");

    definition_update(def, new_ptr);
    TEST_ASSERT_EQ(definition_get(def), new_ptr, "updated pointer should match");

    namespace_destroy(ns);
    return 0;
}

// Test multiple definitions
int test_multiple_definitions() {
    Namespace *ns = namespace_create("test");

    namespace_define(ns, "func1", DEF_FUNCTION, (void*)0x1);
    namespace_define(ns, "func2", DEF_FUNCTION, (void*)0x2);
    namespace_define(ns, "func3", DEF_FUNCTION, (void*)0x3);

    Definition *def1 = namespace_lookup(ns, "func1");
    Definition *def2 = namespace_lookup(ns, "func2");
    Definition *def3 = namespace_lookup(ns, "func3");

    TEST_ASSERT_NOT_NULL(def1, "func1 should be found");
    TEST_ASSERT_NOT_NULL(def2, "func2 should be found");
    TEST_ASSERT_NOT_NULL(def3, "func3 should be found");

    TEST_ASSERT_EQ(definition_get(def1), (void*)0x1, "func1 pointer should match");
    TEST_ASSERT_EQ(definition_get(def2), (void*)0x2, "func2 pointer should match");
    TEST_ASSERT_EQ(definition_get(def3), (void*)0x3, "func3 pointer should match");

    namespace_destroy(ns);
    return 0;
}

int main() {
    int tests_total = 0;
    int tests_passed = 0;
    int tests_failed = 0;

    printf("=== Namespace Tests ===\n");
    RUN_TEST(test_namespace_create);
    RUN_TEST(test_namespace_define);
    RUN_TEST(test_namespace_lookup);
    RUN_TEST(test_definition_update);
    RUN_TEST(test_multiple_definitions);

    printf("\n=== Results ===\n");
    printf("Total: %d, Passed: %d, Failed: %d\n", tests_total, tests_passed, tests_failed);

    return tests_failed == 0 ? 0 : 1;
}
