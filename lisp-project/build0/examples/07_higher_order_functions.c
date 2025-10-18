#include <stdio.h>
#include <stdint.h>

#include "stdio.h"

typedef struct {
    int32_t (*apply_twice)(int32_t (*)(int32_t), int32_t);
    int32_t (*apply_n_times)(int32_t (*)(int32_t), int32_t, int32_t);
    int32_t (*add1)(int32_t);
    int32_t (*_double)(int32_t);
    int32_t (*square)(int32_t);
    int32_t (*compose)(int32_t (*)(int32_t), int32_t (*)(int32_t), int32_t);
    int32_t (*main_fn)();
} Namespace_user;

Namespace_user g_user;

static int32_t apply_twice(int32_t (*)(int32_t), int32_t);
static int32_t apply_n_times(int32_t (*)(int32_t), int32_t, int32_t);
static int32_t add1(int32_t);
static int32_t _double(int32_t);
static int32_t square(int32_t);
static int32_t compose(int32_t (*)(int32_t), int32_t (*)(int32_t), int32_t);
static int32_t main_fn();

void init_namespace_user(Namespace_user* ns) {
    ns->apply_twice = &apply_twice;
    ns->apply_n_times = &apply_n_times;
    ns->add1 = &add1;
    ns->_double = &_double;
    ns->square = &square;
    ns->compose = &compose;
    ns->main_fn = &main_fn;
}

static int32_t apply_twice(int32_t (*f)(int32_t), int32_t x) {
    return f(f(x));
}
static int32_t apply_n_times(int32_t (*f)(int32_t), int32_t n, int32_t x) {
    return ((n <= 0) ? x : g_user.apply_n_times(f, (n - 1), f(x)));
}
static int32_t add1(int32_t x) {
    return (x + 1);
}
static int32_t _double(int32_t x) {
    return (x * 2);
}
static int32_t square(int32_t x) {
    return (x * x);
}
static int32_t compose(int32_t (*f)(int32_t), int32_t (*g)(int32_t), int32_t x) {
    return f(g(x));
}
static int32_t main_fn() {
    printf("apply-twice(add1, 5) = %d\n", g_user.apply_twice(g_user.add1, 5));
    printf("apply-twice(double, 3) = %d\n", g_user.apply_twice(g_user._double, 3));
    printf("apply-n-times(add1, 10, 0) = %d\n", g_user.apply_n_times(g_user.add1, 10, 0));
    printf("apply-n-times(double, 3, 1) = %d\n", g_user.apply_n_times(g_user._double, 3, 1));
    printf("compose(square, double, 3) = %d\n", g_user.compose(g_user.square, g_user._double, 3));
    printf("compose(double, square, 3) = %d\n", g_user.compose(g_user._double, g_user.square, 3));
    return 0;
}
int main() {
    init_namespace_user(&g_user);
    g_user.main_fn();
    return 0;
}
