#include <stdio.h>
#include <string.h>

typedef struct {
    long long matrix[2][2];
    long long row0[2];
    long long row1[2];
    long long r0[2];
    long long r1[2];
    long long a;
    long long b;
    long long c;
    long long d;
    long long result;
} Namespace_user;

Namespace_user g_user;


void init_namespace_user(Namespace_user* ns) {
    for (size_t __i_row0 = 0; __i_row0 < 2; __i_row0++) {
        ns->row0[__i_row0] = 0;
    }
    for (size_t __i_row1 = 0; __i_row1 < 2; __i_row1++) {
        ns->row1[__i_row1] = 0;
    }
}

int main() {
    init_namespace_user(&g_user);
    (g_user.row0[0] = 1);
    (g_user.row0[1] = 2);
    (g_user.row1[0] = 3);
    (g_user.row1[1] = 4);
    memcpy(g_user.matrix[0], g_user.row0, sizeof(g_user.row0));
    memcpy(g_user.matrix[1], g_user.row1, sizeof(g_user.row1));
    memcpy(g_user.r0, g_user.matrix[0], sizeof(g_user.r0));
    memcpy(g_user.r1, g_user.matrix[1], sizeof(g_user.r1));
    g_user.a = g_user.r0[0];
    g_user.b = g_user.r0[1];
    g_user.c = g_user.r1[0];
    g_user.d = g_user.r1[1];
    g_user.result = ((g_user.a + g_user.b) + (g_user.c + g_user.d));
    printf(((const char*)"%lld\n"), g_user.result);
    return 0;
}
