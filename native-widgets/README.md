# native-widgets

A widget library that draws nothing.

Thirty-odd components — button, input, checkbox, switch, slider, table, tabs,
select, dialog, drawer, sheet — with all their composition, geometry, and theme
logic, and no dependency on any graphics library. You hand it a struct of
function pointers; it calls back into your renderer. It ships as a C library:
a static archive and one header.

Extracted from the flowline agent runner, where these widgets were written
directly against raylib.

## The idea

A widget needs perhaps ten drawing primitives. Everything else — how a focus
ring insets by three pixels, how a switch knob lands 22px along its track, what
a disabled button's fill is — is arithmetic and taste that has nothing to do
with the renderer underneath.

So that's the seam. `nw_backend` is ten function pointers:

```c
typedef struct nw_backend {
  void *ctx;
  int64_t (*clear)(void *ctx, nw_color tint);
  int64_t (*fill_rounded_rect)(void *ctx, nw_frame bounds, double radius, nw_color tint);
  int64_t (*stroke_rounded_rect)(void *ctx, nw_frame bounds, double radius, double thickness, nw_color tint);
  int64_t (*fill_rect)(void *ctx, nw_frame bounds, nw_color tint);
  int64_t (*line)(void *ctx, double from_x, double from_y, double to_x, double to_y, double thickness, nw_color tint);
  int64_t (*fill_circle)(void *ctx, double center_x, double center_y, double radius, nw_color tint);
  int64_t (*stroke_circle)(void *ctx, double center_x, double center_y, double radius, double thickness, nw_color tint);
  int64_t (*ring)(void *ctx, double center_x, double center_y, double inner_radius, double outer_radius, double start_degrees, double end_degrees, nw_color tint);
  int64_t (*draw_text)(void *ctx, nw_font font, const char *text, double left, double top, double size, nw_color tint);
  double  (*measure_text)(void *ctx, nw_font font, const char *text, double size);
} nw_backend;
```

Fill that in and every widget works. The raylib backend in
`backends/raylib/nw_raylib.c` is the whole adapter and it is about ninety lines.

Corner radii are pixels, not raylib's fraction-of-the-shorter-side — converting
is the backend's job, since it is a fact about raylib rather than about buttons.

### Is it actually renderer-agnostic?

Two things settle it rather than assert it.

`libnativewidgets.a` has **zero undefined symbols** — it asks nothing of the
outside world, not even libc — and the public header includes only `<stdint.h>`.

More to the point, there are two backends that share nothing. `backends/svg/`
writes SVG: no window, no GPU, no rasterizer, no font engine, and retained
vector output instead of pixels. Both drivers link the *same* `catalog_pages.c`
— the identical widget-drawing code — and the library did not change to make the
second one work:

```sh
./catalog        # raylib: a window you can click
./catalog_svg    # the same catalog, as four .svg files, no graphics library
```

Writing that second backend is also what exposed the places raylib's shape
*had* leaked into the vtable, all since fixed: `stroke_circle` had no thickness
because raylib's `DrawCircleLines` has none, and the text-origin and ring-angle
conventions were raylib's, inherited silently and undocumented. They are now
specified in the header, and the raylib backend emulates thick circle outlines
with a ring band rather than the API pretending they do not exist.

What a backend without font metrics cannot do is measure text, so its centred
labels land a pixel or two off. That is inherent — measuring text is the
renderer's job — and it is documented on `measure_text` rather than papered over.

## Using it

```c
#include "native_widgets.h"

nw_backend backend = my_backend();

nw_button(&backend, font, "Create workflow",
          nw_frame_make(100, 166, 220, 32),
          nw_pointer_interaction(nw_frame_make(100, 166, 220, 32),
                                 mouse_x, mouse_y, mouse_down,
                                 /* focused */ 1, /* disabled */ 0));
```

The library holds no state. A widget's appearance is a pure function of the
bounds and the four flags in `nw_interaction`, so focus, selection, and what is
open live in your application, where they belong. `nw_frame_contains` and
`nw_pointer_interaction` are there for hit testing; nothing above that — no tab
order, no event loop — is the library's business.

It allocates nothing and owns nothing. Strings are borrowed for the duration of
a call.

## Building

```sh
./build.sh              # library, tests, and both drivers
./catalog               # the interactive component catalog
./catalog --check       # headless interaction check
./catalog --snapshot    # render each page to a PNG
./catalog_svg           # render each page to an SVG, no graphics library
```

Requires `coil`, and raylib 5 for the raylib driver only. Consumers of the
library need neither: just `libnativewidgets.a` and `include/native_widgets.h`.

The demo loads fonts from `assets/fonts`, which it borrows from the flowline
repo, so run it from there or point it at your own.

## Layout

- `src/widgets.coil` — the library. Every widget, every token, no renderer.
- `include/native_widgets.h` — the public C API.
- `backends/raylib/` and `backends/svg/` — two backends with nothing in common.
- `examples/catalog_pages.c` — the catalog's layout and drawing, including only
  `native_widgets.h`. Both drivers share it unchanged.
- `examples/catalog_raylib.c` — window, fonts, input. `examples/catalog_svg.c` —
  writes SVG files.
- `tests/abi_test.c` — ABI and widget-behavior checks against a recording
  backend, no window required.
- `tests/parity/` — renders the catalog from flowline's original widgets and
  from this library, and compares pixels.

## Status

Working. The library exports 64 C symbols and passes the ABI and interaction
tests, and `tests/parity/run.sh` reports all four catalog pages
**pixel-identical** to what flowline's original raylib-coupled widgets render.

Extracting this surfaced two real Coil compiler bugs, since a `(fnptr c …)`
vtable is exactly the shape that had never been exercised: `call-ptr` applied no
C-ABI lowering at all, and both sides of the ABI spilled aggregates into an
alloca sized to the struct rather than to whole 8-byte slots, reading and
writing past the end of any struct that was not a multiple of 8. Both are fixed;
the write-up and its repro are in `coil/docs/BUG_CALLPTR_C_ABI.md`.

The one region the parity gate excludes is the spinner, whose arc comes from the
wall clock — the original differs from its own second run there by a comparable
number of pixels. The exclusion is printed on every run along with the pixel
count inside it, so a regression cannot hide behind it.

## Not done yet

- Theme tokens are fixed functions rather than a swappable `nw_theme`. The
  house dark palette is the only one.
- `nw_tabs`, `nw_button_group`, and `nw_toggle_group` take exactly three
  segments, matching the original design rather than generalizing past it.
- flowline still carries its own copy of the widgets; pointing it at this
  library is the next step.

## License and attribution

The component designs are adapted from the **Vercel Labs Native SDK** canvas
widgets (<https://github.com/vercel-labs/native>), provided under the **Apache
License 2.0**, and have been modified.

- `licenses/native-sdk-Apache-2.0.txt` — a copy of that license, as Apache 2.0
  section 4(a) requires when redistributing.
- `THIRD_PARTY_NOTICES.md` — what was taken and, per section 4(b), what was
  changed. `src/widgets.coil` and the public header carry the same notice.

Upstream ships no `NOTICE` file, so section 4(d) does not apply here.

The library ships no fonts. The example drivers load IBM Plex at runtime from
flowline's `assets/fonts`; those are under the SIL Open Font License, and
vendoring them here would mean carrying that license too.

**Not yet decided:** this repository has no `LICENSE` of its own for the parts
that are original work. Apache 2.0 is the natural choice given the derivation,
but that is a call to make rather than assume.
