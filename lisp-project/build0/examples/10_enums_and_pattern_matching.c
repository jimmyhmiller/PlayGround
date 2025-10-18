#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

#include "stdio.h"
typedef enum {
    Color_Red,
    Color_Green,
    Color_Blue,
    Color_Yellow,
} Color;

typedef enum {
    Direction_North,
    Direction_South,
    Direction_East,
    Direction_West,
} Direction;

typedef struct {
    int32_t x;
    int32_t y;
} Coord;


typedef struct {
    uint8_t* (*color_to_string)(Color);
    bool (*is_primary_color)(Color);
    Coord (*direction_to_coords)(Direction);
    Direction (*opposite_direction)(Direction);
    int32_t (*main_fn)();
} Namespace_user;

Namespace_user g_user;

static uint8_t* color_to_string(Color);
static bool is_primary_color(Color);
static Coord direction_to_coords(Direction);
static Direction opposite_direction(Direction);
static int32_t main_fn();

void init_namespace_user(Namespace_user* ns) {
    ns->color_to_string = &color_to_string;
    ns->is_primary_color = &is_primary_color;
    ns->direction_to_coords = &direction_to_coords;
    ns->opposite_direction = &opposite_direction;
    ns->main_fn = &main_fn;
}

static uint8_t* color_to_string(Color c) {
    return ((c == Color_Red) ? "Red" : ((c == Color_Green) ? "Green" : ((c == Color_Blue) ? "Blue" : "Yellow")));
}
static bool is_primary_color(Color c) {
    return ((c == Color_Red) || ((c == Color_Green) || (c == Color_Blue)));
}
static Coord direction_to_coords(Direction dir) {
    return ((dir == Direction_North) ? (Coord){0, 1} : ((dir == Direction_South) ? (Coord){0, -1} : ((dir == Direction_East) ? (Coord){1, 0} : (Coord){-1, 0})));
}
static Direction opposite_direction(Direction dir) {
    return ((dir == Direction_North) ? Direction_South : ((dir == Direction_South) ? Direction_North : ((dir == Direction_East) ? Direction_West : Direction_East)));
}
static int32_t main_fn() {
    printf("Colors:\n");
    printf("  Red: %s (primary: %d)\n", g_user.color_to_string(Color_Red), g_user.is_primary_color(Color_Red));
    printf("  Yellow: %s (primary: %d)\n", g_user.color_to_string(Color_Yellow), g_user.is_primary_color(Color_Yellow));
    printf("\nDirections:\n");
    ({ Coord north_coords = g_user.direction_to_coords(Direction_North); printf("  North coords: (%d, %d)\n", north_coords.x, north_coords.y); });
    ({ Coord east_coords = g_user.direction_to_coords(Direction_East); printf("  East coords: (%d, %d)\n", east_coords.x, east_coords.y); });
    printf("\nOpposite of North is %d (South=%d)\n", g_user.opposite_direction(Direction_North), Direction_South);
    return 0;
}
int main() {
    init_namespace_user(&g_user);
    g_user.main_fn();
    return 0;
}
