/*
 * Copyright 2026 Jimmy Miller
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0. See LICENSE at the
 * repository root, and THIRD_PARTY_NOTICES.md for the upstream work this
 * derives from.
 */

/*
 * The component catalog's state, layout, and drawing — with no renderer.
 *
 * This file and its .c include only native_widgets.h. The raylib driver and
 * the SVG driver both link against it unchanged, which is the point: if a
 * renderer assumption had leaked into the widgets, it would have to surface
 * here.
 */

#ifndef CATALOG_PAGES_H
#define CATALOG_PAGES_H

#include "native_widgets.h"

#define CATALOG_WIDTH 960.0
#define CATALOG_HEIGHT 700.0
#define CATALOG_PAGE_COUNT 4

enum {
  PAGE_CONTROLS = 0,
  PAGE_SURFACES,
  PAGE_NAVIGATION,
  PAGE_OVERLAYS
};

/* Focusable controls on the controls page, in tab order. */
enum {
  CONTROL_BUTTON = 0,
  CONTROL_INPUT,
  CONTROL_TEXTAREA,
  CONTROL_CHECKBOX,
  CONTROL_SWITCH,
  CONTROL_SLIDER,
  CONTROL_TOGGLE,
  CONTROL_RADIO_FIRST,
  CONTROL_RADIO_SECOND,
  CONTROL_COUNT
};

typedef struct {
  int64_t page;
  int64_t focused_control;
  int64_t button_count;
  int64_t checkbox_checked;
  int64_t switch_checked;
  double slider_value;
  int64_t slider_dragging;
  int64_t toggle_checked;
  int64_t radio_value;
  int64_t accordion_open;
  int64_t tab_value;
  int64_t group_value;
  int64_t pagination_value;
  int64_t select_open;
  int64_t combobox_open;
  int64_t overlay;
  double resizable_split;
  /* Opaque to this file; whatever the driver's backend understands. */
  nw_font mono;
  nw_font sans;
  /* Seconds, supplied by the driver, so rendering stays a pure function. */
  double time;
} catalog_state;

typedef struct {
  double pointer_x;
  double pointer_y;
  int64_t pointer_pressed;
  int64_t pointer_down;
  int64_t tab_pressed;
  int64_t space_pressed;
  int64_t left_pressed;
  int64_t right_pressed;
} catalog_input;

extern const char *const catalog_page_names[CATALOG_PAGE_COUNT];

catalog_state catalog_initial_state(void);
void catalog_apply_input(catalog_state *state, catalog_input in);
void catalog_draw(const nw_backend *backend, const catalog_state *state,
                  catalog_input in);

/* Runs the scripted pointer sequence; returns 1 when the state matches. */
int catalog_interaction_check(void);

#endif /* CATALOG_PAGES_H */
