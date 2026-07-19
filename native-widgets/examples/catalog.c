/*
 * The component catalog, drawn entirely through the public C API.
 *
 * This is both the demo and the parity fixture: `catalog --snapshot <prefix>`
 * renders each page to a PNG so the extracted library can be compared against
 * the renders of the code it was extracted from.
 */

#include "native_widgets.h"
#include "nw_raylib.h"

#include <raylib.h>
#include <stdio.h>
#include <string.h>

#define SCREEN_WIDTH 960.0
#define SCREEN_HEIGHT 700.0

#define PAGE_CONTROLS 0
#define PAGE_SURFACES 1
#define PAGE_NAVIGATION 2
#define PAGE_OVERLAYS 3

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
  Font mono;
  Font sans;
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

static catalog_state initial_state(void) {
  catalog_state state;
  memset(&state, 0, sizeof state);
  state.slider_value = 0.42;
  state.accordion_open = 1;
  state.group_value = 1;
  state.pagination_value = 2;
  state.select_open = 1;
  state.combobox_open = 1;
  state.resizable_split = 0.58;
  return state;
}

/* ---- layout ------------------------------------------------------------ */

static nw_frame button_frame(void) { return nw_frame_make(100.0, 166.0, 220.0, 32.0); }
static nw_frame input_frame(void) { return nw_frame_make(100.0, 264.0, 320.0, 32.0); }
static nw_frame textarea_frame(void) { return nw_frame_make(100.0, 364.0, 320.0, 92.0); }
static nw_frame checkbox_frame(void) { return nw_frame_make(560.0, 166.0, 260.0, 28.0); }
static nw_frame switch_frame(void) { return nw_frame_make(560.0, 250.0, 260.0, 28.0); }
static nw_frame slider_frame(void) { return nw_frame_make(560.0, 348.0, 280.0, 28.0); }
static nw_frame toggle_frame(void) { return nw_frame_make(560.0, 442.0, 112.0, 32.0); }
static nw_frame radio_first_frame(void) { return nw_frame_make(560.0, 548.0, 180.0, 28.0); }
static nw_frame radio_second_frame(void) { return nw_frame_make(560.0, 584.0, 180.0, 28.0); }
static nw_frame accordion_frame(int64_t open) {
  return nw_frame_make(64.0, 142.0, 400.0, open ? 100.0 : 48.0);
}
static nw_frame tabs_frame(void) { return nw_frame_make(64.0, 300.0, 360.0, 40.0); }
static nw_frame group_frame(void) { return nw_frame_make(64.0, 404.0, 360.0, 36.0); }
static nw_frame select_frame(void) { return nw_frame_make(540.0, 142.0, 330.0, 32.0); }
static nw_frame combobox_frame(void) { return nw_frame_make(540.0, 310.0, 330.0, 32.0); }
static nw_frame nav_tab_frame(int64_t index) {
  return nw_frame_make(64.0 + 160.0 * (double)index, 650.0, 150.0, 32.0);
}
static nw_frame overlay_button_frame(int64_t index) {
  return nw_frame_make(64.0 + 174.0 * (double)index, 142.0, 160.0, 34.0);
}

/* ---- input ------------------------------------------------------------- */

static void activate_focused(catalog_state *state) {
  switch (state->focused_control) {
  case CONTROL_BUTTON: state->button_count++; break;
  case CONTROL_CHECKBOX: state->checkbox_checked = 1 - state->checkbox_checked; break;
  case CONTROL_SWITCH: state->switch_checked = 1 - state->switch_checked; break;
  case CONTROL_TOGGLE: state->toggle_checked = 1 - state->toggle_checked; break;
  case CONTROL_RADIO_FIRST: state->radio_value = 0; break;
  case CONTROL_RADIO_SECOND: state->radio_value = 1; break;
  default: break;
  }
}

static double clamp01(double value) {
  if (value < 0.0) return 0.0;
  if (value > 1.0) return 1.0;
  return value;
}

/* Which of three equal segments a coordinate falls in, clamped to 0..2. */
static int64_t segment_at(double coordinate, double origin, double width) {
  double index = (coordinate - origin) / width;
  if (index < 0.0) index = 0.0;
  if (index > 2.0) index = 2.0;
  return (int64_t)index;
}

static void apply_input(catalog_state *state, catalog_input in) {
  double x = in.pointer_x;
  double y = in.pointer_y;
  int pressed = in.pointer_pressed == 1;

  if (pressed) {
    for (int64_t tab = 0; tab < 4; tab++) {
      if (nw_frame_contains(nav_tab_frame(tab), x, y)) {
        state->page = tab;
      }
    }
  }

  if (in.tab_pressed) {
    state->focused_control = (state->focused_control + 1) % CONTROL_COUNT;
  }
  if (in.space_pressed) {
    activate_focused(state);
  }
  if (state->focused_control == CONTROL_SLIDER && in.left_pressed) {
    state->slider_value = clamp01(state->slider_value - 0.05);
  }
  if (state->focused_control == CONTROL_SLIDER && in.right_pressed) {
    state->slider_value = clamp01(state->slider_value + 0.05);
  }

  if (pressed && state->page == PAGE_CONTROLS) {
    if (nw_frame_contains(button_frame(), x, y)) {
      state->focused_control = CONTROL_BUTTON;
      state->button_count++;
    } else if (nw_frame_contains(input_frame(), x, y)) {
      state->focused_control = CONTROL_INPUT;
    } else if (nw_frame_contains(textarea_frame(), x, y)) {
      state->focused_control = CONTROL_TEXTAREA;
    } else if (nw_frame_contains(checkbox_frame(), x, y)) {
      state->focused_control = CONTROL_CHECKBOX;
      state->checkbox_checked = 1 - state->checkbox_checked;
    } else if (nw_frame_contains(switch_frame(), x, y)) {
      state->focused_control = CONTROL_SWITCH;
      state->switch_checked = 1 - state->switch_checked;
    } else if (nw_frame_contains(slider_frame(), x, y)) {
      state->focused_control = CONTROL_SLIDER;
      state->slider_dragging = 1;
    } else if (nw_frame_contains(toggle_frame(), x, y)) {
      state->focused_control = CONTROL_TOGGLE;
      state->toggle_checked = 1 - state->toggle_checked;
    } else if (nw_frame_contains(radio_first_frame(), x, y)) {
      state->focused_control = CONTROL_RADIO_FIRST;
      state->radio_value = 0;
    } else if (nw_frame_contains(radio_second_frame(), x, y)) {
      state->focused_control = CONTROL_RADIO_SECOND;
      state->radio_value = 1;
    }
  }

  if (pressed && state->page == PAGE_NAVIGATION) {
    if (nw_frame_contains(accordion_frame(state->accordion_open), x, y)) {
      state->accordion_open = 1 - state->accordion_open;
    } else if (nw_frame_contains(tabs_frame(), x, y)) {
      state->tab_value = segment_at(x, 64.0, 120.0);
    } else if (nw_frame_contains(group_frame(), x, y)) {
      state->group_value = segment_at(x, 64.0, 120.0);
    } else if (nw_frame_contains(select_frame(), x, y)) {
      state->select_open = 1 - state->select_open;
    } else if (nw_frame_contains(combobox_frame(), x, y)) {
      state->combobox_open = 1 - state->combobox_open;
    }
  }

  if (pressed && state->page == PAGE_OVERLAYS) {
    int opened = 0;
    for (int64_t index = 0; index < 3; index++) {
      if (nw_frame_contains(overlay_button_frame(index), x, y)) {
        state->overlay = index + 1;
        opened = 1;
      }
    }
    if (!opened && state->overlay != 0) {
      state->overlay = 0;
    }
  }

  if (in.pointer_down && state->slider_dragging) {
    state->slider_value = clamp01((x - 560.0) / 280.0);
  }
  if (!in.pointer_down) {
    state->slider_dragging = 0;
  }
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

static nw_interaction control_state(const catalog_state *state,
                                    catalog_input in, int64_t identifier,
                                    nw_frame bounds) {
  return nw_pointer_interaction(bounds, in.pointer_x, in.pointer_y,
                                in.pointer_down,
                                state->focused_control == identifier ? 1 : 0, 0);
}

/* ---- pages ------------------------------------------------------------- */

static void draw_controls(const nw_backend *b, const catalog_state *s,
                          catalog_input in) {
  nw_font mono = (nw_font)&s->mono;
  nw_font sans = (nw_font)&s->sans;
  nw_color muted = nw_color_text_muted();

  nw_draw_text(b, mono, "BUTTON", 100.0, 132.0, 11.0, muted);
  nw_button(b, sans, s->button_count == 0 ? "Create workflow" : "Created",
            button_frame(), control_state(s, in, CONTROL_BUTTON, button_frame()));

  nw_draw_text(b, mono, "INPUT", 100.0, 230.0, 11.0, muted);
  nw_input(b, sans, "", "Describe an outcome...", input_frame(),
           control_state(s, in, CONTROL_INPUT, input_frame()));

  nw_draw_text(b, mono, "TEXTAREA", 100.0, 330.0, 11.0, muted);
  nw_textarea(b, sans, "", "Add context for the workflow agent...",
              textarea_frame(),
              control_state(s, in, CONTROL_TEXTAREA, textarea_frame()));

  nw_draw_text(b, mono, "CHECKBOX", 560.0, 132.0, 11.0, muted);
  nw_checkbox(b, sans, "Require human approval", checkbox_frame(),
              s->checkbox_checked,
              control_state(s, in, CONTROL_CHECKBOX, checkbox_frame()));

  nw_draw_text(b, mono, "SWITCH", 560.0, 216.0, 11.0, muted);
  nw_switch(b, sans, "Run automatically", switch_frame(), s->switch_checked,
            control_state(s, in, CONTROL_SWITCH, switch_frame()));

  nw_draw_text(b, mono, "SLIDER", 560.0, 314.0, 11.0, muted);
  nw_slider(b, slider_frame(), s->slider_value,
            control_state(s, in, CONTROL_SLIDER, slider_frame()));

  nw_draw_text(b, mono, "TOGGLE", 560.0, 408.0, 11.0, muted);
  nw_toggle(b, sans, "Inspect", toggle_frame(), s->toggle_checked,
            control_state(s, in, CONTROL_TOGGLE, toggle_frame()));

  nw_draw_text(b, mono, "RADIO GROUP", 560.0, 514.0, 11.0, muted);
  nw_radio(b, sans, "Ask before running", radio_first_frame(),
           s->radio_value == 0 ? 1 : 0,
           control_state(s, in, CONTROL_RADIO_FIRST, radio_first_frame()));
  nw_radio(b, sans, "Run immediately", radio_second_frame(),
           s->radio_value == 1 ? 1 : 0,
           control_state(s, in, CONTROL_RADIO_SECOND, radio_second_frame()));
}

static void draw_surfaces(const nw_backend *b, const catalog_state *s) {
  nw_font mono = (nw_font)&s->mono;
  nw_font sans = (nw_font)&s->sans;
  nw_color muted = nw_color_text_muted();

  nw_alert(b, sans, "Workflow paused",
           "Human approval is required before the next step.",
           nw_frame_make(64.0, 138.0, 400.0, 82.0), 0);
  nw_card(b, sans, "Release readiness", "7 steps / approval required",
          nw_frame_make(64.0, 252.0, 400.0, 112.0));
  nw_bubble(b, sans, "Inspect the repository and recommend a plan.",
            nw_frame_make(64.0, 398.0, 400.0, 46.0), 0);
  nw_breadcrumb(b, sans, 64.0, 486.0, "Release readiness");

  nw_draw_text(b, mono, "AVATAR + BADGE", 64.0, 538.0, 11.0, muted);
  nw_avatar(b, sans, "AG", nw_frame_make(64.0, 566.0, 40.0, 40.0));
  nw_badge(b, sans, "Running", nw_frame_make(118.0, 572.0, 82.0, 26.0), 1);
  nw_badge(b, sans, "Approval", nw_frame_make(212.0, 572.0, 88.0, 26.0), 0);

  nw_draw_text(b, mono, "TABLE", 520.0, 104.0, 11.0, muted);
  const char *cells[] = {
      "Workflow",       "State",  "Release readiness", "Ready",
      "Support triage", "Paused", "Contract review",   "Draft",
  };
  nw_table(b, sans, nw_frame_make(520.0, 138.0, 372.0, 164.0), cells, 4, 0.58);

  nw_draw_text(b, mono, "PROGRESS", 520.0, 336.0, 11.0, muted);
  nw_progress(b, nw_frame_make(520.0, 368.0, 260.0, 8.0), 0.62);

  nw_draw_text(b, mono, "SKELETON + SPINNER", 520.0, 420.0, 11.0, muted);
  nw_skeleton(b, nw_frame_make(520.0, 454.0, 280.0, 16.0));
  nw_skeleton(b, nw_frame_make(520.0, 480.0, 220.0, 16.0));
  nw_spinner(b, nw_frame_make(824.0, 458.0, 28.0, 28.0), GetTime());

  nw_separator(b, nw_frame_make(520.0, 536.0, 372.0, 1.0));
  nw_alert(b, sans, "Creation failed", "The workflow file could not be parsed.",
           nw_frame_make(520.0, 562.0, 372.0, 64.0), 1);
}

static void draw_navigation(const nw_backend *b, const catalog_state *s) {
  nw_font mono = (nw_font)&s->mono;
  nw_font sans = (nw_font)&s->sans;
  nw_color muted = nw_color_text_muted();
  nw_interaction idle = nw_interaction_make(0, 0, 0, 0);

  nw_draw_text(b, mono, "ACCORDION", 64.0, 108.0, 11.0, muted);
  nw_accordion(b, sans, "What will the agent inspect?",
               "Repository structure, tooling, and existing workflows.",
               accordion_frame(s->accordion_open), s->accordion_open, idle);

  nw_draw_text(b, mono, "TABS", 64.0, 268.0, 11.0, muted);
  nw_tabs(b, sans, tabs_frame(), "Design", "Runs", "Settings", s->tab_value);

  nw_draw_text(b, mono, "BUTTON GROUP", 64.0, 372.0, 11.0, muted);
  nw_button_group(b, sans, group_frame(), "Back", "Pause", "Next",
                  s->group_value);

  nw_draw_text(b, mono, "TOGGLE GROUP", 64.0, 472.0, 11.0, muted);
  nw_toggle_group(b, sans, nw_frame_make(64.0, 504.0, 360.0, 36.0), "Back",
                  "Pause", "Next", 0);

  nw_draw_text(b, mono, "PAGINATION", 64.0, 570.0, 11.0, muted);
  const char *pages[] = {"<", "1", "2", "3", ">"};
  nw_pagination(b, sans, nw_frame_make(64.0, 600.0, 180.0, 32.0), pages, 5,
                s->pagination_value);

  nw_draw_text(b, mono, "SELECT", 540.0, 108.0, 11.0, muted);
  nw_frame select = select_frame();
  nw_select(b, sans, "Ask before running", "Select...", select, s->select_open,
            idle);
  if (s->select_open) {
    nw_select_menu(b, select, 92.0);
    nw_select_option(b, sans, "Ask before running", select, select.top + 49.0, 1);
    nw_select_option(b, sans, "Run immediately", select, select.top + 77.0, 0);
    nw_select_option(b, sans, "Save as draft", select, select.top + 105.0, 0);
  }

  nw_draw_text(b, mono, "COMBOBOX", 540.0, 276.0, 11.0, muted);
  nw_frame combobox = combobox_frame();
  nw_combobox(b, sans, "Release", "Search workflows...", combobox,
              nw_interaction_make(0, 0, 1, 0));
  if (s->combobox_open) {
    nw_select_menu(b, combobox, 70.0);
    nw_select_option(b, sans, "Release readiness", combobox,
                     combobox.top + 49.0, 2);
    nw_select_option(b, sans, "Support triage", combobox, combobox.top + 78.0, 0);
  }

  nw_draw_text(b, mono, "DROPDOWN MENU", 540.0, 430.0, 11.0, muted);
  nw_frame menu = nw_frame_make(540.0, 462.0, 220.0, 112.0);
  nw_menu_surface(b, menu);
  nw_menu_item(b, sans, "Run workflow", menu.left + 12.0, menu.top + 10.0, 0);
  nw_menu_item(b, sans, "Duplicate", menu.left + 12.0, menu.top + 38.0, 0);
  nw_separator(b, nw_frame_make(menu.left + 8.0, menu.top + 66.0,
                                menu.width - 16.0, 1.0));
  nw_menu_item(b, sans, "Delete", menu.left + 12.0, menu.top + 78.0, 1);

  nw_tooltip(b, sans, "Open workflow actions",
             nw_frame_make(678.0, 592.0, 174.0, 30.0));
}

static void draw_overlays(const nw_backend *b, const catalog_state *s) {
  nw_font mono = (nw_font)&s->mono;
  nw_font sans = (nw_font)&s->sans;
  nw_color muted = nw_color_text_muted();
  nw_interaction idle = nw_interaction_make(0, 0, 0, 0);

  nw_draw_text(b, mono, "OVERLAY SURFACES", 64.0, 106.0, 11.0, muted);
  nw_button(b, sans, "Open dialog", overlay_button_frame(0), idle);
  nw_button(b, sans, "Open drawer", overlay_button_frame(1), idle);
  nw_button(b, sans, "Open sheet", overlay_button_frame(2), idle);

  nw_draw_text(b, mono, "RESIZABLE", 64.0, 228.0, 11.0, muted);
  nw_resizable(b, sans, nw_frame_make(64.0, 262.0, 828.0, 280.0),
               "Workflow graph", "Inspector", s->resizable_split);

  nw_draw_text(b, sans, "Click outside an open overlay to close it.", 64.0,
               584.0, 13.0, muted);

  switch (s->overlay) {
  case 1:
    nw_dialog(b, sans, "Build this workflow?",
              "The creator agent will write an executable workflow.",
              "Build workflow", SCREEN_WIDTH, SCREEN_HEIGHT);
    break;
  case 2:
    nw_drawer(b, sans, "Run history",
              "Inspect events and agent output from the latest run.",
              SCREEN_WIDTH, SCREEN_HEIGHT);
    break;
  case 3:
    nw_sheet(b, sans, "Workflow settings", "Configure approvals and execution.",
             SCREEN_WIDTH, SCREEN_HEIGHT);
    break;
  default:
    break;
  }
}

static void draw_page_bar(const nw_backend *b, const catalog_state *s) {
  nw_font sans = (nw_font)&s->sans;
  const char *names[] = {"Controls", "Surfaces", "Navigation", "Overlays"};
  nw_interaction idle = nw_interaction_make(0, 0, 0, 0);
  for (int64_t index = 0; index < 4; index++) {
    nw_toggle(b, sans, names[index], nav_tab_frame(index),
              s->page == index ? 1 : 0, idle);
  }
}

static void draw_catalog(const nw_backend *b, const catalog_state *s,
                         catalog_input in) {
  const char *titles[] = {"Native controls", "Native surfaces",
                          "Native navigation", "Native overlays"};

  nw_clear(b, nw_color_background());
  nw_draw_text(b, (nw_font)&s->sans, titles[s->page], 64.0, 38.0, 28.0,
               nw_color_text());
  nw_draw_text(b, (nw_font)&s->mono,
               "HOUSE DARK / APACHE-2.0 ADAPTATION / POINTER + KEYBOARD", 64.0,
               78.0, 11.0, nw_color_text_muted());

  switch (s->page) {
  case PAGE_CONTROLS: draw_controls(b, s, in); break;
  case PAGE_SURFACES: draw_surfaces(b, s); break;
  case PAGE_NAVIGATION: draw_navigation(b, s); break;
  case PAGE_OVERLAYS: draw_overlays(b, s); break;
  default: break;
  }

  if (s->overlay == 0) {
    draw_page_bar(b, s);
  }
}

/* ---- entry ------------------------------------------------------------- */

static void load_fonts(catalog_state *state) {
  state->mono = LoadFontEx("assets/fonts/IBMPlexMono-Regular.ttf", 42, 0, 0);
  state->sans = LoadFontEx("assets/fonts/IBMPlexSans-Regular.ttf", 42, 0, 0);
  SetTextureFilter(state->mono.texture, TEXTURE_FILTER_BILINEAR);
  SetTextureFilter(state->sans.texture, TEXTURE_FILTER_BILINEAR);
}

static int run_snapshot(const char *prefix) {
  const char *pages[] = {"controls", "surfaces", "navigation", "overlays"};

  SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT |
                 FLAG_WINDOW_HIGHDPI);
  InitWindow((int)SCREEN_WIDTH, (int)SCREEN_HEIGHT, "native widgets - snapshot");

  catalog_state state = initial_state();
  load_fonts(&state);

  nw_backend backend = nw_raylib_backend();
  catalog_input in;
  memset(&in, 0, sizeof in);

  for (int64_t page = 0; page < 4; page++) {
    state.page = page;
    /* Shapes reach the framebuffer only when EndDrawing flushes the batch, so
     * capture after a completed frame rather than inside one. */
    for (int warm = 0; warm < 3; warm++) {
      BeginDrawing();
      draw_catalog(&backend, &state, in);
      EndDrawing();
    }
    char path[256];
    snprintf(path, sizeof path, "%s%s.png", prefix, pages[page]);
    TakeScreenshot(path);
  }

  UnloadFont(state.mono);
  UnloadFont(state.sans);
  CloseWindow();
  return 0;
}

static int run_interactive(void) {
  SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT |
                 FLAG_WINDOW_HIGHDPI);
  InitWindow((int)SCREEN_WIDTH, (int)SCREEN_HEIGHT, "native widgets - catalog");
  SetTargetFPS(60);

  catalog_state state = initial_state();
  load_fonts(&state);
  nw_backend backend = nw_raylib_backend();

  while (!WindowShouldClose()) {
    catalog_input in = capture_input();
    apply_input(&state, in);
    BeginDrawing();
    draw_catalog(&backend, &state, in);
    EndDrawing();
  }

  UnloadFont(state.mono);
  UnloadFont(state.sans);
  CloseWindow();
  return 0;
}

/* Headless interaction check: the same sequence the original catalog asserted. */
static int run_check(void) {
  catalog_state state = initial_state();
  catalog_input in;
  memset(&in, 0, sizeof in);

  struct { double x, y; int64_t pressed, down; } steps[] = {
      {200.0, 182.0, 1, 0}, {600.0, 180.0, 1, 0}, {600.0, 264.0, 1, 0},
      {700.0, 360.0, 1, 1}, {700.0, 360.0, 0, 0}, {600.0, 456.0, 1, 0},
      {600.0, 598.0, 1, 0},
  };

  for (size_t step = 0; step < sizeof steps / sizeof steps[0]; step++) {
    memset(&in, 0, sizeof in);
    in.pointer_x = steps[step].x;
    in.pointer_y = steps[step].y;
    in.pointer_pressed = steps[step].pressed;
    in.pointer_down = steps[step].down;
    apply_input(&state, in);
  }

  int ok = state.button_count == 1 && state.checkbox_checked == 1 &&
           state.switch_checked == 1 && state.slider_value > 0.49 &&
           state.toggle_checked == 1 && state.radio_value == 1;
  printf("%s\n", ok ? "interaction check passed" : "interaction check FAILED");
  return ok ? 0 : 1;
}

int main(int argc, char **argv) {
  if (argc >= 2 && strcmp(argv[1], "--check") == 0) {
    return run_check();
  }
  if (argc >= 3 && strcmp(argv[1], "--snapshot") == 0) {
    return run_snapshot(argv[2]);
  }
  if (argc >= 2 && strcmp(argv[1], "--snapshot") == 0) {
    return run_snapshot("catalog-");
  }
  return run_interactive();
}
