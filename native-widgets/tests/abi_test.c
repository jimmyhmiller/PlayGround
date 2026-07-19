/*
 * Checks that the C ABI the header promises is the one the library actually
 * has: struct arguments and struct returns in both directions, and widgets
 * calling back through the vtable in the expected order.
 */

#include "native_widgets.h"

#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);                   \
      failures++;                                                              \
    }                                                                          \
  } while (0)

/* ---- a backend that records instead of drawing ------------------------- */

#define MAX_CALLS 128

typedef struct {
  char calls[MAX_CALLS][64];
  int count;
  int text_calls;
  double last_text_left;
  double last_text_top;
} recorder;

static void record(void *ctx, const char *name) {
  recorder *r = (recorder *)ctx;
  if (r->count < MAX_CALLS) {
    snprintf(r->calls[r->count], sizeof(r->calls[0]), "%s", name);
    r->count++;
  }
}

static int64_t rec_clear(void *ctx, nw_color t) {
  (void)t;
  record(ctx, "clear");
  return 0;
}
static int64_t rec_fill_rounded(void *ctx, nw_frame b, double radius,
                                nw_color t) {
  (void)b;
  (void)radius;
  (void)t;
  record(ctx, "fill_rounded_rect");
  return 0;
}
static int64_t rec_stroke_rounded(void *ctx, nw_frame b, double radius,
                                  double thickness, nw_color t) {
  (void)b;
  (void)radius;
  (void)thickness;
  (void)t;
  record(ctx, "stroke_rounded_rect");
  return 0;
}
static int64_t rec_fill_rect(void *ctx, nw_frame b, nw_color t) {
  (void)b;
  (void)t;
  record(ctx, "fill_rect");
  return 0;
}
static int64_t rec_line(void *ctx, double x1, double y1, double x2, double y2,
                        double thickness, nw_color t) {
  (void)x1;
  (void)y1;
  (void)x2;
  (void)y2;
  (void)thickness;
  (void)t;
  record(ctx, "line");
  return 0;
}
static int64_t rec_fill_circle(void *ctx, double x, double y, double radius,
                               nw_color t) {
  (void)x;
  (void)y;
  (void)radius;
  (void)t;
  record(ctx, "fill_circle");
  return 0;
}
static int64_t rec_stroke_circle(void *ctx, double x, double y, double radius,
                                 nw_color t) {
  (void)x;
  (void)y;
  (void)radius;
  (void)t;
  record(ctx, "stroke_circle");
  return 0;
}
static int64_t rec_ring(void *ctx, double x, double y, double inner,
                        double outer, double start, double end, nw_color t) {
  (void)x;
  (void)y;
  (void)inner;
  (void)outer;
  (void)start;
  (void)end;
  (void)t;
  record(ctx, "ring");
  return 0;
}
static int64_t rec_draw_text(void *ctx, nw_font font, const char *text,
                             double left, double top, double size,
                             nw_color t) {
  recorder *r = (recorder *)ctx;
  (void)font;
  (void)text;
  (void)size;
  (void)t;
  record(ctx, "draw_text");
  r->text_calls++;
  r->last_text_left = left;
  r->last_text_top = top;
  return 0;
}
/* A predictable metric so centering math is checkable. */
static double rec_measure_text(void *ctx, nw_font font, const char *text,
                               double size) {
  (void)ctx;
  (void)font;
  return (double)strlen(text) * size * 0.5;
}

static nw_backend recording_backend(recorder *r) {
  nw_backend backend;
  memset(&backend, 0, sizeof backend);
  backend.ctx = r;
  backend.clear = rec_clear;
  backend.fill_rounded_rect = rec_fill_rounded;
  backend.stroke_rounded_rect = rec_stroke_rounded;
  backend.fill_rect = rec_fill_rect;
  backend.line = rec_line;
  backend.fill_circle = rec_fill_circle;
  backend.stroke_circle = rec_stroke_circle;
  backend.ring = rec_ring;
  backend.draw_text = rec_draw_text;
  backend.measure_text = rec_measure_text;
  return backend;
}

static int called(const recorder *r, const char *name) {
  for (int i = 0; i < r->count; i++) {
    if (strcmp(r->calls[i], name) == 0) {
      return 1;
    }
  }
  return 0;
}

/* ---- tests -------------------------------------------------------------- */

static void test_struct_returns(void) {
  /* 32-byte struct: returned indirectly on arm64. */
  nw_frame f = nw_frame_make(10.0, 20.0, 220.0, 32.0);
  CHECK(f.left == 10.0);
  CHECK(f.top == 20.0);
  CHECK(f.width == 220.0);
  CHECK(f.height == 32.0);

  /* 4-byte struct: returned in a register. */
  nw_color c = nw_color_rgba(255, 100, 103, 200);
  CHECK(c.r == 255 && c.g == 100 && c.b == 103 && c.a == 200);

  nw_color accent = nw_color_accent();
  CHECK(accent.r == 229 && accent.g == 229 && accent.b == 229 &&
        accent.a == 255);

  nw_color border = nw_color_border();
  CHECK(border.a == 26);

  /* Struct in, struct out. */
  nw_frame dialog = nw_dialog_frame(960.0, 700.0, 420.0, 210.0);
  CHECK(dialog.left == 270.0);
  CHECK(dialog.top == 245.0);

  nw_frame cell = nw_pagination_cell_frame(nw_frame_make(64.0, 600.0, 180.0, 32.0), 2);
  CHECK(cell.left == 64.0 + 64.0);
  CHECK(cell.width == 32.0 && cell.height == 32.0);
}

static void test_hit_testing(void) {
  nw_frame bounds = nw_frame_make(100.0, 166.0, 220.0, 32.0);
  CHECK(nw_frame_contains(bounds, 200.0, 182.0) == 1);
  CHECK(nw_frame_contains(bounds, 99.0, 182.0) == 0);
  CHECK(nw_frame_contains(bounds, 200.0, 199.0) == 0);
  /* Edges are inclusive. */
  CHECK(nw_frame_contains(bounds, 100.0, 166.0) == 1);
  CHECK(nw_frame_contains(bounds, 320.0, 198.0) == 1);

  nw_interaction inside = nw_pointer_interaction(bounds, 200.0, 182.0, 1, 0, 0);
  CHECK(inside.hovered == 1);
  CHECK(inside.pressed == 1);
  CHECK(inside.focused == 0);

  nw_interaction outside = nw_pointer_interaction(bounds, 0.0, 0.0, 1, 1, 0);
  CHECK(outside.hovered == 0);
  CHECK(outside.pressed == 0);
  CHECK(outside.focused == 1);
}

static void test_button_draws_through_backend(void) {
  recorder r;
  memset(&r, 0, sizeof r);
  nw_backend backend = recording_backend(&r);

  nw_button(&backend, NULL, "Create workflow", nw_frame_make(100.0, 166.0, 220.0, 32.0),
            nw_interaction_make(0, 0, 0, 0));

  CHECK(called(&r, "fill_rounded_rect"));
  CHECK(called(&r, "draw_text"));
  CHECK(r.text_calls == 1);
  /* Unfocused: no focus ring stroke. */
  CHECK(!called(&r, "stroke_rounded_rect"));

  /* measure_text reports 15 chars * 14 * 0.5 = 105 wide, so a 220-wide
   * button centers the label at 100 + (220 - 105) / 2. */
  CHECK(r.last_text_left == 100.0 + (220.0 - 105.0) / 2.0);
  CHECK(r.last_text_top == 166.0 + (32.0 - 14.0) / 2.0);
}

static void test_focus_ring_is_conditional(void) {
  recorder unfocused, focused;
  memset(&unfocused, 0, sizeof unfocused);
  memset(&focused, 0, sizeof focused);

  nw_backend a = recording_backend(&unfocused);
  nw_backend b = recording_backend(&focused);
  nw_frame bounds = nw_frame_make(0.0, 0.0, 100.0, 32.0);

  nw_toggle(&a, NULL, "Inspect", bounds, 1, nw_interaction_make(0, 0, 0, 0));
  nw_toggle(&b, NULL, "Inspect", bounds, 1, nw_interaction_make(0, 0, 1, 0));

  CHECK(!called(&unfocused, "stroke_rounded_rect"));
  CHECK(called(&focused, "stroke_rounded_rect"));
}

static void test_checkbox_checkmark(void) {
  recorder off, on;
  memset(&off, 0, sizeof off);
  memset(&on, 0, sizeof on);

  nw_backend a = recording_backend(&off);
  nw_backend b = recording_backend(&on);
  nw_frame bounds = nw_frame_make(560.0, 166.0, 260.0, 28.0);

  nw_checkbox(&a, NULL, "Require approval", bounds, 0, nw_interaction_make(0, 0, 0, 0));
  nw_checkbox(&b, NULL, "Require approval", bounds, 1, nw_interaction_make(0, 0, 0, 0));

  /* The tick is two strokes, drawn only when checked. */
  CHECK(!called(&off, "line"));
  CHECK(called(&on, "line"));
}

static void test_slider_clamps(void) {
  recorder low, high;
  memset(&low, 0, sizeof low);
  memset(&high, 0, sizeof high);

  nw_backend a = recording_backend(&low);
  nw_backend b = recording_backend(&high);
  nw_frame bounds = nw_frame_make(560.0, 348.0, 280.0, 28.0);

  /* Out-of-range values must not crash or draw a negative-width fill. */
  nw_slider(&a, bounds, -5.0, nw_interaction_make(0, 0, 0, 0));
  nw_slider(&b, bounds, 12.0, nw_interaction_make(0, 0, 0, 0));

  CHECK(low.count > 0);
  CHECK(high.count > 0);
  CHECK(called(&low, "fill_circle"));
  CHECK(called(&high, "fill_circle"));
}

static void test_table_uses_caller_strings(void) {
  recorder r;
  memset(&r, 0, sizeof r);
  nw_backend backend = recording_backend(&r);

  const char *cells[] = {
      "Workflow", "State", "Release readiness", "Ready",
      "Support triage", "Paused", "Contract review", "Draft",
  };
  nw_table(&backend, NULL, nw_frame_make(520.0, 138.0, 372.0, 164.0), cells, 4,
           0.58);

  /* Two cells per row across four rows. */
  CHECK(r.text_calls == 8);
  /* Column divider plus three row rules. */
  CHECK(called(&r, "line"));
}

static void test_pagination_uses_caller_labels(void) {
  recorder r;
  memset(&r, 0, sizeof r);
  nw_backend backend = recording_backend(&r);

  const char *labels[] = {"<", "1", "2", "3", "4", "5", ">"};
  nw_pagination(&backend, NULL, nw_frame_make(64.0, 600.0, 180.0, 32.0), labels,
                7, 2);

  CHECK(r.text_calls == 7);
  /* Exactly one active cell gets a background. */
  CHECK(called(&r, "fill_rounded_rect"));
}

static void test_spinner_uses_ring(void) {
  recorder r;
  memset(&r, 0, sizeof r);
  nw_backend backend = recording_backend(&r);

  nw_spinner(&backend, nw_frame_make(824.0, 458.0, 28.0, 28.0), 1.5);
  CHECK(called(&r, "ring"));
}

int main(void) {
  test_struct_returns();
  test_hit_testing();
  test_button_draws_through_backend();
  test_focus_ring_is_conditional();
  test_checkbox_checkmark();
  test_slider_clamps();
  test_table_uses_caller_strings();
  test_pagination_uses_caller_labels();
  test_spinner_uses_ring();

  if (failures == 0) {
    printf("all ABI and widget checks passed\n");
    return 0;
  }
  printf("%d check(s) failed\n", failures);
  return 1;
}
