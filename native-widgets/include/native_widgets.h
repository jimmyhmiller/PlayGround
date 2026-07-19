/*
 * Copyright 2026 Jimmy Miller
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0. See LICENSE at the
 * repository root, and THIRD_PARTY_NOTICES.md for the upstream work this
 * derives from.
 */

/*
 * native_widgets.h — a render-agnostic widget library.
 *
 * The component designs are adapted from the Vercel Labs Native SDK canvas
 * widgets (https://github.com/vercel-labs/native), provided under the Apache
 * License 2.0, and have been modified. See THIRD_PARTY_NOTICES.md and
 * licenses/native-sdk-Apache-2.0.txt.
 *
 * The library draws nothing itself. You give it an nw_backend of function
 * pointers and it calls back into your renderer, so the same widgets run on
 * raylib, a software rasterizer, a PDF writer, or a recorder in a test.
 *
 * Coordinates are pixels in a top-left origin space, as doubles. Corner radii
 * are pixels too: converting to whatever your renderer wants is the backend's
 * job.
 *
 * Strings are NUL-terminated UTF-8 and are only borrowed for the duration of
 * the call. The library allocates nothing and owns nothing.
 */

#ifndef NATIVE_WIDGETS_H
#define NATIVE_WIDGETS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Types
 * ------------------------------------------------------------------------- */

typedef struct {
  double left, top, width, height;
} nw_frame;

typedef struct {
  uint8_t r, g, b, a;
} nw_color;

/* Widget appearance is a pure function of these four flags: the library keeps
 * no state of its own between calls. */
typedef struct {
  int64_t hovered, pressed, focused, disabled;
} nw_interaction;

/* Whatever your renderer's font handle is. The library only hands it back to
 * draw_text and measure_text. */
typedef void *nw_font;

/*
 * The renderer you supply. Every entry receives your ctx first. Drawing
 * entries return 0; the return exists only to keep the table uniform.
 *
 * Every field must be non-NULL: the library calls through them without
 * checking, and a widget will only invoke what its shape needs.
 */
typedef struct nw_backend {
  void *ctx;

  int64_t (*clear)(void *ctx, nw_color tint);

  int64_t (*fill_rounded_rect)(void *ctx, nw_frame bounds, double radius,
                               nw_color tint);
  int64_t (*stroke_rounded_rect)(void *ctx, nw_frame bounds, double radius,
                                 double thickness, nw_color tint);
  int64_t (*fill_rect)(void *ctx, nw_frame bounds, nw_color tint);

  int64_t (*line)(void *ctx, double from_x, double from_y, double to_x,
                  double to_y, double thickness, nw_color tint);
  int64_t (*fill_circle)(void *ctx, double center_x, double center_y,
                         double radius, nw_color tint);
  int64_t (*stroke_circle)(void *ctx, double center_x, double center_y,
                           double radius, double thickness, nw_color tint);
  /*
   * Angles are degrees, increasing clockwise, with zero at three o'clock —
   * the same sense as the y-down coordinate space. The band runs between
   * inner_radius and outer_radius.
   */
  int64_t (*ring)(void *ctx, double center_x, double center_y,
                  double inner_radius, double outer_radius,
                  double start_degrees, double end_degrees, nw_color tint);

  /*
   * `top` is the top of the line box, not the baseline: a backend whose text
   * API positions by baseline must add the ascent back. `size` is the em size
   * in pixels, and glyphs are laid out with no extra letter spacing.
   */
  int64_t (*draw_text)(void *ctx, nw_font font, const char *text, double left,
                       double top, double size, nw_color tint);
  /*
   * Advance width of `text` in pixels at `size`. Widgets use this to centre
   * labels, so centring is exactly as good as this measurement: a backend with
   * no font metrics can estimate, and its labels will be off by a pixel or two
   * rather than wrong.
   */
  double (*measure_text)(void *ctx, nw_font font, const char *text,
                         double size);
} nw_backend;

/* -------------------------------------------------------------------------
 * Constructors, geometry, hit testing
 * ------------------------------------------------------------------------- */

nw_frame nw_frame_make(double left, double top, double width, double height);
nw_color nw_color_make(int64_t red, int64_t green, int64_t blue);
nw_color nw_color_rgba(int64_t red, int64_t green, int64_t blue, int64_t alpha);
nw_interaction nw_interaction_make(int64_t hovered, int64_t pressed,
                                   int64_t focused, int64_t disabled);

int64_t nw_frame_contains(nw_frame bounds, double horizontal, double vertical);

/* Derives hover and press from a pointer; focus and disabled stay yours. */
nw_interaction nw_pointer_interaction(nw_frame bounds, double horizontal,
                                      double vertical, int64_t pointer_down,
                                      int64_t focused, int64_t disabled);

nw_frame nw_dialog_frame(double screen_width, double screen_height,
                         double width, double height);
nw_frame nw_pagination_cell_frame(nw_frame bounds, int64_t item);

/* -------------------------------------------------------------------------
 * Theme tokens — house dark
 * ------------------------------------------------------------------------- */

nw_color nw_color_background(void);
nw_color nw_color_surface(void);
nw_color nw_color_surface_subtle(void);
nw_color nw_color_surface_pressed(void);
nw_color nw_color_text(void);
nw_color nw_color_text_muted(void);
nw_color nw_color_border(void);
nw_color nw_color_accent(void);
nw_color nw_color_accent_text(void);
nw_color nw_color_focus_ring(void);
nw_color nw_color_danger(void);

/* -------------------------------------------------------------------------
 * Primitives — the same calls the widgets make, for your own chrome
 * ------------------------------------------------------------------------- */

int64_t nw_clear(const nw_backend *backend, nw_color tint);
int64_t nw_fill_rounded_rect(const nw_backend *backend, nw_frame bounds,
                             double radius, nw_color tint);
int64_t nw_stroke_rounded_rect(const nw_backend *backend, nw_frame bounds,
                               double radius, double thickness, nw_color tint);
int64_t nw_fill_rect(const nw_backend *backend, nw_frame bounds, nw_color tint);
int64_t nw_draw_line(const nw_backend *backend, double from_x, double from_y,
                     double to_x, double to_y, double thickness, nw_color tint);
int64_t nw_draw_text(const nw_backend *backend, nw_font font, const char *text,
                     double left, double top, double size, nw_color tint);
double nw_measure_text(const nw_backend *backend, nw_font font,
                       const char *text, double size);
int64_t nw_draw_centered_text(const nw_backend *backend, nw_font font,
                              const char *text, nw_frame bounds, double size,
                              nw_color tint);
int64_t nw_draw_focus_ring(const nw_backend *backend, nw_frame bounds,
                           double radius, int64_t focused);

/* -------------------------------------------------------------------------
 * Controls
 * ------------------------------------------------------------------------- */

int64_t nw_button(const nw_backend *backend, nw_font font, const char *label,
                  nw_frame bounds, nw_interaction state);
int64_t nw_input(const nw_backend *backend, nw_font font, const char *value,
                 const char *placeholder, nw_frame bounds,
                 nw_interaction state);
int64_t nw_textarea(const nw_backend *backend, nw_font font, const char *value,
                    const char *placeholder, nw_frame bounds,
                    nw_interaction state);
int64_t nw_checkbox(const nw_backend *backend, nw_font font, const char *label,
                    nw_frame bounds, int64_t checked, nw_interaction state);
int64_t nw_switch(const nw_backend *backend, nw_font font, const char *label,
                  nw_frame bounds, int64_t checked, nw_interaction state);
/* `value` is 0..1 and is clamped. */
int64_t nw_slider(const nw_backend *backend, nw_frame bounds, double value,
                  nw_interaction state);
int64_t nw_toggle(const nw_backend *backend, nw_font font, const char *label,
                  nw_frame bounds, int64_t checked, nw_interaction state);
int64_t nw_radio(const nw_backend *backend, nw_font font, const char *label,
                 nw_frame bounds, int64_t selected, nw_interaction state);

/* -------------------------------------------------------------------------
 * Surfaces
 * ------------------------------------------------------------------------- */

int64_t nw_alert(const nw_backend *backend, nw_font font, const char *title,
                 const char *description, nw_frame bounds,
                 int64_t destructive);
int64_t nw_avatar(const nw_backend *backend, nw_font font,
                  const char *initials, nw_frame bounds);
int64_t nw_badge(const nw_backend *backend, nw_font font, const char *label,
                 nw_frame bounds, int64_t strong);
int64_t nw_breadcrumb(const nw_backend *backend, nw_font font, double left,
                      double top, const char *active);
int64_t nw_bubble(const nw_backend *backend, nw_font font, const char *message,
                  nw_frame bounds, int64_t agent);
int64_t nw_card(const nw_backend *backend, nw_font font, const char *title,
                const char *description, nw_frame bounds);
int64_t nw_separator(const nw_backend *backend, nw_frame bounds);
int64_t nw_skeleton(const nw_backend *backend, nw_frame bounds);
/* `value` is 0..1 and is clamped. */
int64_t nw_progress(const nw_backend *backend, nw_frame bounds, double value);
/* `time` is seconds; the arc position is derived from it. */
int64_t nw_spinner(const nw_backend *backend, nw_frame bounds, double time);

/*
 * Two-column table. `cells` is row-major, 2 * row_count strings, row 0 being
 * the header. `column_split` is the divider as a fraction of the width.
 */
int64_t nw_table(const nw_backend *backend, nw_font font, nw_frame bounds,
                 const char **cells, int64_t row_count, double column_split);

/* -------------------------------------------------------------------------
 * Navigation
 * ------------------------------------------------------------------------- */

int64_t nw_accordion(const nw_backend *backend, nw_font font,
                     const char *title, const char *body, nw_frame bounds,
                     int64_t open, nw_interaction state);
int64_t nw_tabs(const nw_backend *backend, nw_font font, nw_frame bounds,
                const char *first, const char *second, const char *third,
                int64_t selected);
int64_t nw_button_group(const nw_backend *backend, nw_font font,
                        nw_frame bounds, const char *first, const char *second,
                        const char *third, int64_t selected);
int64_t nw_toggle_group(const nw_backend *backend, nw_font font,
                        nw_frame bounds, const char *first, const char *second,
                        const char *third, int64_t selected);
int64_t nw_pagination(const nw_backend *backend, nw_font font, nw_frame bounds,
                      const char **labels, int64_t label_count, int64_t page);

/* The closed control. Draw the popup separately so you control overlay order. */
int64_t nw_select(const nw_backend *backend, nw_font font, const char *label,
                  const char *placeholder, nw_frame bounds, int64_t open,
                  nw_interaction state);
/* The popup panel, positioned below `bounds` (the control's own frame). */
int64_t nw_select_menu(const nw_backend *backend, nw_frame bounds,
                       double height);
/*
 * One option inside that panel. `top` is absolute. `emphasis` is 0 for muted,
 * 1 for active, 2 for active with a highlight backing.
 */
int64_t nw_select_option(const nw_backend *backend, nw_font font,
                         const char *label, nw_frame bounds, double top,
                         int64_t emphasis);
int64_t nw_combobox(const nw_backend *backend, nw_font font, const char *query,
                    const char *placeholder, nw_frame bounds,
                    nw_interaction state);

int64_t nw_menu_surface(const nw_backend *backend, nw_frame bounds);
int64_t nw_menu_item(const nw_backend *backend, nw_font font,
                     const char *label, double left, double top,
                     int64_t destructive);
int64_t nw_tooltip(const nw_backend *backend, nw_font font, const char *label,
                   nw_frame bounds);

/* -------------------------------------------------------------------------
 * Overlays
 * ------------------------------------------------------------------------- */

/* Full-screen dimmer. `opacity` is 0..255. */
int64_t nw_scrim(const nw_backend *backend, double screen_width,
                 double screen_height, int64_t opacity);
int64_t nw_dialog(const nw_backend *backend, nw_font font, const char *title,
                  const char *description, const char *confirm,
                  double screen_width, double screen_height);
int64_t nw_drawer(const nw_backend *backend, nw_font font, const char *title,
                  const char *description, double screen_width,
                  double screen_height);
int64_t nw_sheet(const nw_backend *backend, nw_font font, const char *title,
                 const char *description, double screen_width,
                 double screen_height);
/* `split` is the divider as a fraction of the width, 0..1. */
int64_t nw_resizable(const nw_backend *backend, nw_font font, nw_frame bounds,
                     const char *leading_label, const char *trailing_label,
                     double split);

#ifdef __cplusplus
}
#endif

#endif /* NATIVE_WIDGETS_H */
