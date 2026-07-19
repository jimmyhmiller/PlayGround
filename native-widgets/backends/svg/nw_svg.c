/*
 * Copyright 2026 Jimmy Miller
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0. See LICENSE at the
 * repository root, and THIRD_PARTY_NOTICES.md for the upstream work this
 * derives from.
 */

#include "nw_svg.h"

#include <math.h>
#include <string.h>

static nw_svg_context *ctx_of(void *ctx) { return (nw_svg_context *)ctx; }

/* SVG has no alpha channel on a colour, so opacity rides alongside. */
static void paint(FILE *out, const char *attribute, nw_color c) {
  fprintf(out, "%s=\"rgb(%d,%d,%d)\"", attribute, c.r, c.g, c.b);
  if (c.a != 255) {
    fprintf(out, " %s-opacity=\"%.4f\"", attribute, c.a / 255.0);
  }
}

static void write_escaped(FILE *out, const char *text) {
  for (const char *p = text; *p; p++) {
    switch (*p) {
    case '&': fputs("&amp;", out); break;
    case '<': fputs("&lt;", out); break;
    case '>': fputs("&gt;", out); break;
    case '"': fputs("&quot;", out); break;
    default: fputc(*p, out); break;
    }
  }
}

static int64_t svg_clear(void *ctx, nw_color tint) {
  FILE *out = ctx_of(ctx)->out;
  fputs("  <rect x=\"0\" y=\"0\" width=\"100%\" height=\"100%\" ", out);
  paint(out, "fill", tint);
  fputs("/>\n", out);
  return 0;
}

static int64_t svg_fill_rounded_rect(void *ctx, nw_frame b, double radius,
                                     nw_color tint) {
  FILE *out = ctx_of(ctx)->out;
  fprintf(out,
          "  <rect x=\"%.3f\" y=\"%.3f\" width=\"%.3f\" height=\"%.3f\" "
          "rx=\"%.3f\" ",
          b.left, b.top, b.width, b.height, radius);
  paint(out, "fill", tint);
  fputs("/>\n", out);
  return 0;
}

static int64_t svg_stroke_rounded_rect(void *ctx, nw_frame b, double radius,
                                       double thickness, nw_color tint) {
  FILE *out = ctx_of(ctx)->out;
  fprintf(out,
          "  <rect x=\"%.3f\" y=\"%.3f\" width=\"%.3f\" height=\"%.3f\" "
          "rx=\"%.3f\" fill=\"none\" stroke-width=\"%.3f\" ",
          b.left, b.top, b.width, b.height, radius, thickness);
  paint(out, "stroke", tint);
  fputs("/>\n", out);
  return 0;
}

static int64_t svg_fill_rect(void *ctx, nw_frame b, nw_color tint) {
  FILE *out = ctx_of(ctx)->out;
  fprintf(out, "  <rect x=\"%.3f\" y=\"%.3f\" width=\"%.3f\" height=\"%.3f\" ",
          b.left, b.top, b.width, b.height);
  paint(out, "fill", tint);
  fputs("/>\n", out);
  return 0;
}

static int64_t svg_line(void *ctx, double x1, double y1, double x2, double y2,
                        double thickness, nw_color tint) {
  FILE *out = ctx_of(ctx)->out;
  fprintf(out,
          "  <line x1=\"%.3f\" y1=\"%.3f\" x2=\"%.3f\" y2=\"%.3f\" "
          "stroke-width=\"%.3f\" ",
          x1, y1, x2, y2, thickness);
  paint(out, "stroke", tint);
  fputs("/>\n", out);
  return 0;
}

static int64_t svg_fill_circle(void *ctx, double cx, double cy, double radius,
                               nw_color tint) {
  FILE *out = ctx_of(ctx)->out;
  fprintf(out, "  <circle cx=\"%.3f\" cy=\"%.3f\" r=\"%.3f\" ", cx, cy, radius);
  paint(out, "fill", tint);
  fputs("/>\n", out);
  return 0;
}

static int64_t svg_stroke_circle(void *ctx, double cx, double cy, double radius,
                                 double thickness, nw_color tint) {
  FILE *out = ctx_of(ctx)->out;
  fprintf(out,
          "  <circle cx=\"%.3f\" cy=\"%.3f\" r=\"%.3f\" fill=\"none\" "
          "stroke-width=\"%.3f\" ",
          cx, cy, radius, thickness);
  paint(out, "stroke", tint);
  fputs("/>\n", out);
  return 0;
}

/* Degrees, clockwise, zero at three o'clock — matching the widget's spinner. */
static int64_t svg_ring(void *ctx, double cx, double cy, double inner,
                        double outer, double start, double end, nw_color tint) {
  FILE *out = ctx_of(ctx)->out;
  double mid = (inner + outer) / 2.0;
  double sweep = end - start;
  double a0 = start * M_PI / 180.0;
  double a1 = end * M_PI / 180.0;
  int large = fabs(sweep) > 180.0 ? 1 : 0;

  fprintf(out,
          "  <path d=\"M %.3f %.3f A %.3f %.3f 0 %d 1 %.3f %.3f\" "
          "fill=\"none\" stroke-width=\"%.3f\" stroke-linecap=\"round\" ",
          cx + mid * cos(a0), cy + mid * sin(a0), mid, mid, large,
          cx + mid * cos(a1), cy + mid * sin(a1), outer - inner);
  paint(out, "stroke", tint);
  fputs("/>\n", out);
  return 0;
}

static int64_t svg_draw_text(void *ctx, nw_font font, const char *text,
                             double left, double top, double size,
                             nw_color tint) {
  nw_svg_context *c = ctx_of(ctx);
  /*
   * The library positions text by the top of the line box. SVG's y is the
   * baseline, so the ascent has to be added back.
   */
  double baseline = top + size * c->ascent_ratio;
  fprintf(c->out,
          "  <text x=\"%.3f\" y=\"%.3f\" font-size=\"%.3f\" "
          "font-family=\"%s\" ",
          left, baseline, size,
          font && strcmp((const char *)font, "mono") == 0
              ? "IBM Plex Mono, monospace"
              : "IBM Plex Sans, sans-serif");
  paint(c->out, "fill", tint);
  fputc('>', c->out);
  write_escaped(c->out, text);
  fputs("</text>\n", c->out);
  return 0;
}

/*
 * With no font engine there is nothing to measure, so this estimates. Centred
 * labels land within a pixel or two rather than exactly, which is a property of
 * this backend, not of the widgets.
 */
static double svg_measure_text(void *ctx, nw_font font, const char *text,
                               double size) {
  (void)font;
  return (double)strlen(text) * size * ctx_of(ctx)->advance_ratio;
}

nw_backend nw_svg_backend(nw_svg_context *context, double width,
                          double height) {
  fprintf(context->out,
          "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%.0f\" "
          "height=\"%.0f\" viewBox=\"0 0 %.0f %.0f\">\n",
          width, height, width, height);

  if (context->advance_ratio == 0.0) {
    context->advance_ratio = 0.5;
  }
  if (context->ascent_ratio == 0.0) {
    context->ascent_ratio = 0.78;
  }

  nw_backend backend;
  backend.ctx = context;
  backend.clear = svg_clear;
  backend.fill_rounded_rect = svg_fill_rounded_rect;
  backend.stroke_rounded_rect = svg_stroke_rounded_rect;
  backend.fill_rect = svg_fill_rect;
  backend.line = svg_line;
  backend.fill_circle = svg_fill_circle;
  backend.stroke_circle = svg_stroke_circle;
  backend.ring = svg_ring;
  backend.draw_text = svg_draw_text;
  backend.measure_text = svg_measure_text;
  return backend;
}

void nw_svg_finish(nw_svg_context *context) {
  fputs("</svg>\n", context->out);
}
