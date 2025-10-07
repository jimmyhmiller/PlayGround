#define _DEFAULT_SOURCE
#include "test.h"
#include "namespace.h"
#include "bundle.h"

typedef int (*CalcFn)(int, int);

int test_cross_bundle_calls() {
    Namespace *ns = namespace_create("test");

    // Load util bundle first
    Bundle *util = bundle_load("./util_bundle.so", ns);
    TEST_ASSERT_NOT_NULL(util, "util bundle should load");

    // Load dependent bundle
    Bundle *dependent = bundle_load("./dependent_bundle.so", ns);
    TEST_ASSERT_NOT_NULL(dependent, "dependent bundle should load");

    // Call calculate_area which uses multiply internally
    Definition *area_def = namespace_lookup(ns, "calculate_area");
    TEST_ASSERT_NOT_NULL(area_def, "calculate_area should be defined");

    CalcFn calc = (CalcFn)definition_get(area_def);
    int result = calc(5, 4);
    TEST_ASSERT_EQ(result, 20, "calculate_area(5, 4) should return 20");

    bundle_unload(dependent);
    bundle_unload(util);
    namespace_destroy(ns);
    return 0;
}

int test_bundle_order() {
    Namespace *ns = namespace_create("test");

    // Load dependent first (before util)
    Bundle *dependent = bundle_load("./dependent_bundle.so", ns);
    TEST_ASSERT_NOT_NULL(dependent, "dependent bundle should load even without util");

    // Now load util
    Bundle *util = bundle_load("./util_bundle.so", ns);
    TEST_ASSERT_NOT_NULL(util, "util bundle should load");

    // Now calling should work
    Definition *area_def = namespace_lookup(ns, "calculate_area");
    CalcFn calc = (CalcFn)definition_get(area_def);
    int result = calc(3, 7);
    TEST_ASSERT_EQ(result, 21, "calculate_area(3, 7) should return 21");

    bundle_unload(dependent);
    bundle_unload(util);
    namespace_destroy(ns);
    return 0;
}

int main() {
    int tests_total = 0;
    int tests_passed = 0;
    int tests_failed = 0;

    printf("=== Cross-Bundle Tests ===\n");
    RUN_TEST(test_cross_bundle_calls);
    RUN_TEST(test_bundle_order);

    printf("\n=== Results ===\n");
    printf("Total: %d, Passed: %d, Failed: %d\n", tests_total, tests_passed, tests_failed);

    return tests_failed == 0 ? 0 : 1;
}
