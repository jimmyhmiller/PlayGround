#include <stdio.h>
#include <stdint.h>
#include <stddef.h>

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
    return ({ int32_t arr[5]; for (size_t __i_arr = 0; __i_arr < 5; __i_arr++) { arr[__i_arr] = 0; } (arr[0] = 10); (arr[1] = 20); (arr[2] = 30); (arr[3] = 40); (arr[4] = 50); printf("Array length: %d\n", 5); printf("Array elements: "); ({ for (int32_t i = 0; (i < 5); i = (i + 1)) { printf("%d ", arr[i]); } }); printf("\n"); ({ int32_t sum = 0; ({ for (int32_t i = 0; (i < 5); i = (i + 1)) { sum = (sum + arr[i]); } }); printf("Sum: %d\n", sum); }); ({ int32_t squares[10]; for (size_t __i_squares = 0; __i_squares < 10; __i_squares++) { squares[__i_squares] = 0; } ({ for (int32_t i = 0; (i < 10); i = (i + 1)) { (squares[i] = (i * i)); } }); printf("Squares (0-9): "); ({ for (int32_t i = 0; (i < 10); i = (i + 1)) { printf("%d ", squares[i]); } }); printf("\n"); 0; }); });
}
int main() {
    init_namespace_user(&g_user);
    g_user.main_fn();
    return 0;
}
