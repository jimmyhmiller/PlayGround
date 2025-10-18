#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "stdio.h"
typedef struct {
    int32_t x;
    int32_t y;
} Point;


typedef struct {
    int32_t (*main_fn)();
} Namespace_user;

Namespace_user g_user;

static int32_t main_fn();

void init_namespace_user(Namespace_user* ns) {
    ns->main_fn = &main_fn;
}

static int32_t main_fn() {
    return ({ int32_t* ptr = (int32_t*)({ int32_t* __tmp_ptr = malloc(sizeof(int32_t)); *__tmp_ptr = 42; __tmp_ptr; }); ({ int32_t _ = printf("Initial value: %d\n", (*ptr)); ({ (*ptr = 100); ({ int32_t _ = printf("After write: %d\n", (*ptr)); ({ Point* point_ptr = (Point*)({ Point* __tmp_ptr = malloc(sizeof(Point)); *__tmp_ptr = (Point){10, 20}; __tmp_ptr; }); ({ int32_t _ = printf("Point: (%d, %d)\n", point_ptr->x, point_ptr->y); ({ point_ptr->x = 30; ({ point_ptr->y = 40; printf("Modified Point: (%d, %d)\n", point_ptr->x, point_ptr->y); 0; }); }); }); }); }); }); }); });
}
int main() {
    init_namespace_user(&g_user);
    g_user.main_fn();
    return 0;
}
