#include <stdio.h>
#include <stdint.h>

#include "stdio.h"

typedef struct {
    int32_t (*test_file_ops)();
    int32_t (*main_fn)();
} Namespace_user;

Namespace_user g_user;

static int32_t test_file_ops();
static int32_t main_fn();

void init_namespace_user(Namespace_user* ns) {
    ns->test_file_ops = &test_file_ops;
    ns->main_fn = &main_fn;
}

static int32_t test_file_ops() {
    printf("=== Testing File I/O ===\n");
    ({ void* f_out = (void*)fopen("/tmp/test_llm.txt", "w"); ((f_out == NULL) ? printf("ERROR: Failed to open file for writing\n") : ({ fprintf(f_out, "Hello from Lisp!\n"); fprintf(f_out, "File I/O works!\n"); fclose(f_out); printf("Successfully wrote to /tmp/test_llm.txt\n"); })); 0; });
    ({ void* f_in = (void*)fopen("/tmp/test_llm.txt", "r"); ((f_in == NULL) ? printf("ERROR: Failed to open file for reading\n") : ({ fclose(f_in); printf("Successfully opened file for reading\n"); })); 0; });
    printf("File I/O test completed!\n");
    printf("Check /tmp/test_llm.txt to see the output\n");
    return 0;
}
static int32_t main_fn() {
    return g_user.test_file_ops();
}
int main() {
    init_namespace_user(&g_user);
    g_user.main_fn();
    return 0;
}
