#include <stdio.h>
#include <stdlib.h>

typedef struct {
    long long x;
    long long* ptr;
    long long val;
} Namespace_user;

Namespace_user g_user;


void init_namespace_user(Namespace_user* ns) {
    ns->x = 10;
    ns->ptr = ({ long long* __tmp_ptr = malloc(sizeof(long long)); *__tmp_ptr = ns->x; __tmp_ptr; });
    ns->val = (*ns->ptr);
}

int main() {
    init_namespace_user(&g_user);
    (*g_user.ptr = 42);
    free(g_user.ptr);
    printf(((const char*)"%lld\n"), g_user.val);
    return 0;
}
