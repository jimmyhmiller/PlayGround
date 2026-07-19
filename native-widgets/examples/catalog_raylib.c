/*
 * Copyright 2026 Jimmy Miller
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0. See LICENSE at the
 * repository root, and THIRD_PARTY_NOTICES.md for the upstream work this
 * derives from.
 */

/*
 * The raylib driver for the catalog: a window, fonts, and input.
 *
 * All the drawing lives in catalog_pages.c, which never mentions raylib.
 */

#include "catalog_pages.h"
#include "nw_raylib.h"

#include <raylib.h>
#include <stdio.h>
#include <string.h>

static Font mono_font;
static Font sans_font;

static void load_fonts(catalog_state *state) {
  mono_font = LoadFontEx("assets/fonts/IBMPlexMono-Regular.ttf", 42, 0, 0);
  sans_font = LoadFontEx("assets/fonts/IBMPlexSans-Regular.ttf", 42, 0, 0);
  SetTextureFilter(mono_font.texture, TEXTURE_FILTER_BILINEAR);
  SetTextureFilter(sans_font.texture, TEXTURE_FILTER_BILINEAR);
  state->mono = &mono_font;
  state->sans = &sans_font;
}

static void open_window(const char *title) {
  SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT |
                 FLAG_WINDOW_HIGHDPI);
  InitWindow((int)CATALOG_WIDTH, (int)CATALOG_HEIGHT, title);
}

static catalog_input capture_input(void) {
  catalog_input in;
  memset(&in, 0, sizeof in);
  Vector2 mouse = GetMousePosition();
  in.pointer_x = mouse.x;
  in.pointer_y = mouse.y;
  in.pointer_pressed = IsMouseButtonPressed(MOUSE_BUTTON_LEFT);
  in.pointer_down = IsMouseButtonDown(MOUSE_BUTTON_LEFT);
  in.tab_pressed = IsKeyPressed(KEY_TAB);
  in.space_pressed = IsKeyPressed(KEY_SPACE);
  in.left_pressed = IsKeyPressed(KEY_LEFT);
  in.right_pressed = IsKeyPressed(KEY_RIGHT);
  return in;
}

static int run_snapshot(const char *prefix) {
  open_window("native widgets - snapshot");

  catalog_state state = catalog_initial_state();
  load_fonts(&state);

  nw_backend backend = nw_raylib_backend();
  catalog_input in;
  memset(&in, 0, sizeof in);

  for (int64_t page = 0; page < CATALOG_PAGE_COUNT; page++) {
    state.page = page;
    /* Shapes reach the framebuffer only when EndDrawing flushes the batch, so
     * capture after a completed frame rather than inside one. */
    for (int warm = 0; warm < 3; warm++) {
      state.time = GetTime();
      BeginDrawing();
      catalog_draw(&backend, &state, in);
      EndDrawing();
    }
    char path[256];
    snprintf(path, sizeof path, "%s%s.png", prefix, catalog_page_names[page]);
    TakeScreenshot(path);
  }

  UnloadFont(mono_font);
  UnloadFont(sans_font);
  CloseWindow();
  return 0;
}

static int run_interactive(void) {
  open_window("native widgets - catalog");
  SetTargetFPS(60);

  catalog_state state = catalog_initial_state();
  load_fonts(&state);
  nw_backend backend = nw_raylib_backend();

  while (!WindowShouldClose()) {
    catalog_input in = capture_input();
    catalog_apply_input(&state, in);
    state.time = GetTime();
    BeginDrawing();
    catalog_draw(&backend, &state, in);
    EndDrawing();
  }

  UnloadFont(mono_font);
  UnloadFont(sans_font);
  CloseWindow();
  return 0;
}

int main(int argc, char **argv) {
  if (argc >= 2 && strcmp(argv[1], "--check") == 0) {
    int ok = catalog_interaction_check();
    printf("%s\n", ok ? "interaction check passed" : "interaction check FAILED");
    return ok ? 0 : 1;
  }
  if (argc >= 3 && strcmp(argv[1], "--snapshot") == 0) {
    return run_snapshot(argv[2]);
  }
  if (argc >= 2 && strcmp(argv[1], "--snapshot") == 0) {
    return run_snapshot("catalog-");
  }
  return run_interactive();
}
