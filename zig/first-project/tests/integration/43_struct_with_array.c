#include <stdio.h>
#include <string.h>

typedef struct {
    long long data[3];
} Vec3;


typedef struct {
    long long arr[3];
    Vec3 v;
    long long v_data[3];
    long long x;
    long long y;
    long long z;
    long long result;
} Namespace_user;

Namespace_user g_user;


void init_namespace_user(Namespace_user* ns) {
}

int main() {
    init_namespace_user(&g_user);
    memcpy(g_user.arr, ({ long long __tmp_arr[3]; for (size_t __i = 0; __i < 3; __i++) __tmp_arr[__i] = 0; __tmp_arr; }), sizeof(g_user.arr));
    (g_user.arr[0] = 1);
    (g_user.arr[1] = 2);
    (g_user.arr[2] = 3);
    g_user.v = ({ Vec3 __tmp_struct; memcpy(__tmp_struct.data, g_user.arr, sizeof(__tmp_struct.data)); __tmp_struct; });
    memcpy(g_user.v_data, g_user.v.data, sizeof(g_user.v_data));
    g_user.x = g_user.v_data[0];
    g_user.y = g_user.v_data[1];
    g_user.z = g_user.v_data[2];
    g_user.result = (g_user.x + (g_user.y + g_user.z));
    printf(((const char*)"%lld\n"), g_user.result);
    return 0;
}
