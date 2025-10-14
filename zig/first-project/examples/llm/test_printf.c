#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "stdio.h"

typedef struct {
    int32_t (*main_fn)();
} Namespace_user;

Namespace_user g_user;

static int32_t main_fn();

void init_namespace_user(Namespace_user* ns) {
    ns->main_fn = &main_fn;
}

static int32_t main_fn() {
    printf("Testing printf with format specifiers:\n");
    ({ int32_t x = 42; printf("Integer: %d\n", x); 0; });
    ({ float y = 3.14159; printf("Float: %f\n", y); 0; });
    ({ int32_t a = 10; float b = 2.5; printf("Int=%d, Float=%f\n", a, b); 0; });
    ({ float* arr = (float*)(float*)malloc(3 * sizeof(float)); (arr[0] = 1.1); (arr[1] = 2.2); (arr[2] = 3.3); printf("Array: [%f, %f, %f]\n", arr[0], arr[1], arr[2]); free(arr); 0; });
    return 0;
}
int main() {
    init_namespace_user(&g_user);
    g_user.main_fn();
    return 0;
}
