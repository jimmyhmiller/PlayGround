// Thin C shim: scalar-only wrappers around the raylib calls that take Color
// BY VALUE (gc-rust passes value structs by pointer, an ABI mismatch) or return
// bool. gc-rust calls these via extern "C" with pure scalars.
#include "raylib.h"

void  rs_clear(int r, int g, int b)            { ClearBackground((Color){r,g,b,255}); }
void  rs_circle(double x, double y, double rad, int r, int g, int b, int a) {
    DrawCircle((int)x, (int)y, (float)rad, (Color){r,g,b,a});
}
void  rs_rect(double x, double y, double w, double h, int r, int g, int b, int a) {
    DrawRectangle((int)x, (int)y, (int)w, (int)h, (Color){r,g,b,a});
}
void  rs_text(const char* s, int x, int y, int size, int r, int g, int b) {
    DrawText(s, x, y, size, (Color){r,g,b,255});
}
int   rs_should_close(void)        { return WindowShouldClose() ? 1 : 0; }
int   rs_key_down(int key)         { return IsKeyDown(key) ? 1 : 0; }
double rs_frame_time(void)         { return (double)GetFrameTime(); }
void  rs_screenshot(const char* path) { TakeScreenshot(path); }
