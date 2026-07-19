/*
 * Copyright 2026 Jimmy Miller
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0. See LICENSE at the
 * repository root, and THIRD_PARTY_NOTICES.md for the upstream work this
 * derives from.
 */

#include "nw_raylib.h"

#include <raylib.h>

/* raylib's rounded-rectangle roundness is a fraction of the shorter side, so
 * pixel radii convert here rather than in the widgets. */
#define NW_CORNER_SEGMENTS 8
#define NW_RING_SEGMENTS 24

static Rectangle to_rect(nw_frame f) {
  Rectangle r = {(float)f.left, (float)f.top, (float)f.width, (float)f.height};
  return r;
}

static Color to_color(nw_color c) {
  Color out = {c.r, c.g, c.b, c.a};
  return out;
}

static float to_roundness(nw_frame f, double radius) {
  double shorter = f.width < f.height ? f.width : f.height;
  if (shorter <= 0.0) {
    return 0.0f;
  }
  return (float)(radius / shorter);
}

static Font resolve_font(nw_font font) {
  return font ? *(Font *)font : GetFontDefault();
}

static int64_t rl_clear(void *ctx, nw_color tint) {
  (void)ctx;
  ClearBackground(to_color(tint));
  return 0;
}

static int64_t rl_fill_rounded_rect(void *ctx, nw_frame bounds, double radius,
                                    nw_color tint) {
  (void)ctx;
  DrawRectangleRounded(to_rect(bounds), to_roundness(bounds, radius),
                       NW_CORNER_SEGMENTS, to_color(tint));
  return 0;
}

static int64_t rl_stroke_rounded_rect(void *ctx, nw_frame bounds, double radius,
                                      double thickness, nw_color tint) {
  (void)ctx;
  DrawRectangleRoundedLinesEx(to_rect(bounds), to_roundness(bounds, radius),
                              NW_CORNER_SEGMENTS, (float)thickness,
                              to_color(tint));
  return 0;
}

static int64_t rl_fill_rect(void *ctx, nw_frame bounds, nw_color tint) {
  (void)ctx;
  DrawRectangleRec(to_rect(bounds), to_color(tint));
  return 0;
}

static int64_t rl_line(void *ctx, double from_x, double from_y, double to_x,
                       double to_y, double thickness, nw_color tint) {
  (void)ctx;
  Vector2 from = {(float)from_x, (float)from_y};
  Vector2 to = {(float)to_x, (float)to_y};
  DrawLineEx(from, to, (float)thickness, to_color(tint));
  return 0;
}

static int64_t rl_fill_circle(void *ctx, double center_x, double center_y,
                              double radius, nw_color tint) {
  (void)ctx;
  DrawCircle((int)center_x, (int)center_y, (float)radius, to_color(tint));
  return 0;
}

static int64_t rl_stroke_circle(void *ctx, double center_x, double center_y,
                                double radius, double thickness,
                                nw_color tint) {
  (void)ctx;
  /* raylib's circle outline is always hairline, so anything thicker has to be
   * drawn as a ring band. */
  if (thickness <= 1.0) {
    DrawCircleLines((int)center_x, (int)center_y, (float)radius,
                    to_color(tint));
    return 0;
  }
  Vector2 center = {(float)center_x, (float)center_y};
  DrawRing(center, (float)(radius - thickness / 2.0),
           (float)(radius + thickness / 2.0), 0.0f, 360.0f, 36,
           to_color(tint));
  return 0;
}

static int64_t rl_ring(void *ctx, double center_x, double center_y,
                       double inner_radius, double outer_radius,
                       double start_degrees, double end_degrees,
                       nw_color tint) {
  (void)ctx;
  Vector2 center = {(float)center_x, (float)center_y};
  DrawRing(center, (float)inner_radius, (float)outer_radius,
           (float)start_degrees, (float)end_degrees, NW_RING_SEGMENTS,
           to_color(tint));
  return 0;
}

static int64_t rl_draw_text(void *ctx, nw_font font, const char *text,
                            double left, double top, double size,
                            nw_color tint) {
  (void)ctx;
  Vector2 position = {(float)left, (float)top};
  DrawTextEx(resolve_font(font), text, position, (float)size, 0.0f,
             to_color(tint));
  return 0;
}

static double rl_measure_text(void *ctx, nw_font font, const char *text,
                              double size) {
  (void)ctx;
  return (double)MeasureTextEx(resolve_font(font), text, (float)size, 0.0f).x;
}

nw_backend nw_raylib_backend(void) {
  nw_backend backend;
  backend.ctx = 0;
  backend.clear = rl_clear;
  backend.fill_rounded_rect = rl_fill_rounded_rect;
  backend.stroke_rounded_rect = rl_stroke_rounded_rect;
  backend.fill_rect = rl_fill_rect;
  backend.line = rl_line;
  backend.fill_circle = rl_fill_circle;
  backend.stroke_circle = rl_stroke_circle;
  backend.ring = rl_ring;
  backend.draw_text = rl_draw_text;
  backend.measure_text = rl_measure_text;
  return backend;
}
