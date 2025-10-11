#include <stdio.h>
#include <stdlib.h>

typedef struct {
    long long (*sum_array_recursive)(long long*, long long, long long);
    long long* arr;
    long long result;
} Namespace_user;

Namespace_user g_user;

static long long sum_array_recursive(long long*, long long, long long);

void init_namespace_user(Namespace_user* ns) {
    ns->sum_array_recursive = &sum_array_recursive;
    ns->arr = ({ long long* __arr = (long long*)malloc(5 * sizeof(long long)); for (size_t __i = 0; __i < 5; __i++) __arr[__i] = 0; __arr; });
}

static long long sum_array_recursive(long long* arr, long long idx, long long len) {
    return ((idx == len) ? 0 : (arr[idx] + sum_array_recursive(arr, (idx + 1), len)));
}
int main() {
    init_namespace_user(&g_user);
    (g_user.arr[0] = 1);
    (g_user.arr[1] = 2);
    (g_user.arr[2] = 3);
    (g_user.arr[3] = 4);
    (g_user.arr[4] = 5);
    g_user.result = g_user.sum_array_recursive(g_user.arr, 0, 5);
    free(g_user.arr);
    printf(((const char*)"%lld\n"), g_user.result);
    return 0;
}
