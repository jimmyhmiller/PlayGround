#ifndef TEST_H
#define TEST_H

#include <stdio.h>
#include <string.h>

#define TEST_ASSERT(condition, message) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL: %s:%d - %s\n", __FILE__, __LINE__, message); \
        return -1; \
    } \
} while(0)

#define TEST_ASSERT_EQ(a, b, message) do { \
    if ((a) != (b)) { \
        fprintf(stderr, "FAIL: %s:%d - %s (expected %ld, got %ld)\n", __FILE__, __LINE__, message, (long)(b), (long)(a)); \
        return -1; \
    } \
} while(0)

#define TEST_ASSERT_STR_EQ(a, b, message) do { \
    if (strcmp((a), (b)) != 0) { \
        fprintf(stderr, "FAIL: %s:%d - %s (expected '%s', got '%s')\n", __FILE__, __LINE__, message, (b), (a)); \
        return -1; \
    } \
} while(0)

#define TEST_ASSERT_NULL(ptr, message) do { \
    if ((ptr) != NULL) { \
        fprintf(stderr, "FAIL: %s:%d - %s\n", __FILE__, __LINE__, message); \
        return -1; \
    } \
} while(0)

#define TEST_ASSERT_NOT_NULL(ptr, message) do { \
    if ((ptr) == NULL) { \
        fprintf(stderr, "FAIL: %s:%d - %s\n", __FILE__, __LINE__, message); \
        return -1; \
    } \
} while(0)

#define RUN_TEST(test_func) do { \
    printf("Running %s... ", #test_func); \
    fflush(stdout); \
    if (test_func() == 0) { \
        printf("PASS\n"); \
        tests_passed++; \
    } else { \
        tests_failed++; \
    } \
    tests_total++; \
} while(0)

#endif
