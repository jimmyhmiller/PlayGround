#include <stdio.h>
#include <stdint.h>

#include "stdio.h"
typedef struct {
    int32_t value;
} SimpleToken;


typedef struct {
    SimpleToken (*make_simple_token)(int32_t);
    SimpleToken (*problematic_pattern)();
    int32_t (*main_fn)();
} Namespace_user;

Namespace_user g_user;

static SimpleToken make_simple_token(int32_t);
static SimpleToken problematic_pattern();
static int32_t main_fn();

void init_namespace_user(Namespace_user* ns) {
    ns->make_simple_token = &make_simple_token;
    ns->problematic_pattern = &problematic_pattern;
    ns->main_fn = &main_fn;
}

static SimpleToken make_simple_token(int32_t v) {
    return (SimpleToken){v};
}
static SimpleToken problematic_pattern() {
    return ({ int32_t start = 10; ({ int32_t input = 5; ({ int32_t len = 0; ({ while ((len < 3)) { len = (len + 1); } }); g_user.make_simple_token(len); }); }); });
}
static int32_t main_fn() {
    printf("Testing the exact pattern from tokenizer...\n");
    ({ SimpleToken tok = g_user.problematic_pattern(); printf("Token value: %d\n", tok.value); });
    return 0;
}
int main() {
    init_namespace_user(&g_user);
    g_user.main_fn();
    return 0;
}
