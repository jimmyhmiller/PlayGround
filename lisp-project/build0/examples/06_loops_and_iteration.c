#include <stdio.h>
#include <stdint.h>

#include "stdio.h"

typedef struct {
    int32_t (*factorial_while)(int32_t);
    int32_t (*factorial_for)(int32_t);
    int32_t (*countdown)(int32_t);
    int32_t (*main_fn)();
} Namespace_user;

Namespace_user g_user;

static int32_t factorial_while(int32_t);
static int32_t factorial_for(int32_t);
static int32_t countdown(int32_t);
static int32_t main_fn();

void init_namespace_user(Namespace_user* ns) {
    ns->factorial_while = &factorial_while;
    ns->factorial_for = &factorial_for;
    ns->countdown = &countdown;
    ns->main_fn = &main_fn;
}

static int32_t factorial_while(int32_t n) {
    return ({ int32_t result = 1; ({ int32_t i = n; ({ while ((i > 1)) { result = (result * i); i = (i - 1); } }); result; }); });
}
static int32_t factorial_for(int32_t n) {
    return ({ int32_t result = 1; ({ for (int32_t i = 2; (i <= n); i = (i + 1)) { result = (result * i); } }); result; });
}
static int32_t countdown(int32_t n) {
    ({ while ((n > 0)) { printf("%d...\n", n); n = (n - 1); } });
    return printf("Blast off!\n");
}
static int32_t main_fn() {
    printf("Factorial (while): 5! = %d\n", g_user.factorial_while(5));
    printf("Factorial (for): 7! = %d\n", g_user.factorial_for(7));
    printf("\nCountdown:\n");
    g_user.countdown(5);
    return 0;
}
int main() {
    init_namespace_user(&g_user);
    g_user.main_fn();
    return 0;
}
