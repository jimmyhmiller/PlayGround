#define _DEFAULT_SOURCE
#include "test.h"
#include "namespace.h"
#include "bundle.h"
#include <stdlib.h>
#include <unistd.h>

typedef int (*BinaryOpFn)(int, int);

int test_bundle_load() {
    Namespace *ns = namespace_create("test");

    Bundle *bundle = bundle_load("./test_bundle_v1.so", ns);
    TEST_ASSERT_NOT_NULL(bundle, "bundle should load");

    Definition *def = namespace_lookup(ns, "test_add");
    TEST_ASSERT_NOT_NULL(def, "test_add should be defined after bundle load");

    bundle_unload(bundle);
    namespace_destroy(ns);
    return 0;
}

int test_bundle_function_call() {
    Namespace *ns = namespace_create("test");
    Bundle *bundle = bundle_load("./test_bundle_v1.so", ns);
    TEST_ASSERT_NOT_NULL(bundle, "bundle should load");

    Definition *def = namespace_lookup(ns, "test_add");
    TEST_ASSERT_NOT_NULL(def, "test_add should be defined");

    BinaryOpFn add = (BinaryOpFn)definition_get(def);
    int result = add(5, 3);
    TEST_ASSERT_EQ(result, 8, "test_add(5, 3) should return 8");

    bundle_unload(bundle);
    namespace_destroy(ns);
    return 0;
}

int test_bundle_reload() {
    Namespace *ns = namespace_create("test");

    // Copy v1 to test location
    system("cp test_bundle_v1.c test_bundle_reload.c");
    system("gcc -Wall -Wextra -std=c11 -g -fPIC -shared test_bundle_reload.c -o test_bundle_reload.so 2>/dev/null");

    Bundle *bundle = bundle_load("./test_bundle_reload.so", ns);
    TEST_ASSERT_NOT_NULL(bundle, "bundle should load");

    Definition *def = namespace_lookup(ns, "test_add");
    BinaryOpFn add = (BinaryOpFn)definition_get(def);
    int result_v1 = add(5, 3);
    TEST_ASSERT_EQ(result_v1, 8, "v1: test_add(5, 3) should return 8");

    // Copy v2 and rebuild
    system("cp test_bundle_v2.c test_bundle_reload.c");
    system("gcc -Wall -Wextra -std=c11 -g -fPIC -shared test_bundle_reload.c -o test_bundle_reload.so 2>/dev/null");
    usleep(100000); // Wait for file system

    // Reload
    int reload_result = bundle_reload(bundle);
    TEST_ASSERT_EQ(reload_result, 0, "bundle_reload should succeed");

    // Call again - should get new version
    add = (BinaryOpFn)definition_get(def);
    int result_v2 = add(5, 3);
    TEST_ASSERT_EQ(result_v2, 108, "v2: test_add(5, 3) should return 108");

    bundle_unload(bundle);
    namespace_destroy(ns);
    system("rm -f test_bundle_reload.c test_bundle_reload.so");
    return 0;
}

int test_multiple_functions() {
    Namespace *ns = namespace_create("test");
    Bundle *bundle = bundle_load("./test_bundle_v1.so", ns);

    Definition *add_def = namespace_lookup(ns, "test_add");
    Definition *mul_def = namespace_lookup(ns, "test_multiply");

    TEST_ASSERT_NOT_NULL(add_def, "test_add should exist");
    TEST_ASSERT_NOT_NULL(mul_def, "test_multiply should exist");

    BinaryOpFn add = (BinaryOpFn)definition_get(add_def);
    BinaryOpFn mul = (BinaryOpFn)definition_get(mul_def);

    TEST_ASSERT_EQ(add(10, 5), 15, "add(10, 5) should be 15");
    TEST_ASSERT_EQ(mul(10, 5), 50, "mul(10, 5) should be 50");

    bundle_unload(bundle);
    namespace_destroy(ns);
    return 0;
}

int main() {
    int tests_total = 0;
    int tests_passed = 0;
    int tests_failed = 0;

    printf("=== Bundle Tests ===\n");
    RUN_TEST(test_bundle_load);
    RUN_TEST(test_bundle_function_call);
    RUN_TEST(test_bundle_reload);
    RUN_TEST(test_multiple_functions);

    printf("\n=== Results ===\n");
    printf("Total: %d, Passed: %d, Failed: %d\n", tests_total, tests_passed, tests_failed);

    return tests_failed == 0 ? 0 : 1;
}
