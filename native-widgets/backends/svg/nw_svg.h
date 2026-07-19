/*
 * Copyright 2026 Jimmy Miller
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0. See LICENSE at the
 * repository root, and THIRD_PARTY_NOTICES.md for the upstream work this
 * derives from.
 */

/*
 * An nw_backend that writes SVG.
 *
 * It exists to keep the library honest. It shares nothing with raylib: no
 * window, no GPU, no rasterizer, no font engine, and its output is retained
 * vector geometry rather than pixels. If a raylib assumption were baked into
 * the widgets rather than into raylib's adapter, writing this would have been
 * impossible or would have required changing the library.
 */

#ifndef NW_SVG_H
#define NW_SVG_H

#include "native_widgets.h"

#include <stdio.h>

typedef struct {
  FILE *out;
  /*
   * Two things a font engine would otherwise answer. The library asks the
   * backend to measure text and positions labels from the top of the line box,
   * so a backend with no metrics has to approximate both.
   */
  double advance_ratio; /* mean glyph advance as a fraction of font size */
  double ascent_ratio;  /* baseline offset from the top of the line box */
} nw_svg_context;

/* Writes the SVG header. */
nw_backend nw_svg_backend(nw_svg_context *context, double width, double height);

/* Writes the closing tag. */
void nw_svg_finish(nw_svg_context *context);

#endif /* NW_SVG_H */
