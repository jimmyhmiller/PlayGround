/*
 * Copyright 2026 Jimmy Miller
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0. See LICENSE at the
 * repository root, and THIRD_PARTY_NOTICES.md for the upstream work this
 * derives from.
 */

/*
 * The same catalog, rendered to SVG.
 *
 * It links catalog_pages.c unchanged — the identical drawing code the raylib
 * driver uses — with a backend that has no window, no raster, and no font
 * metrics. Nothing in the library needed to change to make this work.
 */

#include "catalog_pages.h"
#include "nw_svg.h"

#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
  const char *prefix = argc >= 2 ? argv[1] : "catalog-";

  for (int64_t page = 0; page < CATALOG_PAGE_COUNT; page++) {
    char path[256];
    snprintf(path, sizeof path, "%s%s.svg", prefix, catalog_page_names[page]);

    FILE *out = fopen(path, "w");
    if (!out) {
      fprintf(stderr, "cannot write %s\n", path);
      return 1;
    }

    nw_svg_context context;
    memset(&context, 0, sizeof context);
    context.out = out;

    nw_backend backend =
        nw_svg_backend(&context, CATALOG_WIDTH, CATALOG_HEIGHT);

    catalog_state state = catalog_initial_state();
    state.page = page;
    /* Font handles here are just tags the SVG backend maps to a family. */
    state.mono = (nw_font) "mono";
    state.sans = (nw_font) "sans";
    /* Fixed, so the spinner is reproducible. */
    state.time = 1.5;

    catalog_input in;
    memset(&in, 0, sizeof in);

    catalog_draw(&backend, &state, in);

    nw_svg_finish(&context);
    fclose(out);
    printf("wrote %s\n", path);
  }

  return 0;
}
