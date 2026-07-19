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
  int64_t (*stroke_circle)(void *ctx, double center_x, double center_y, double radius, nw_color tint);
  int64_t (*ring)(void *ctx, double center_x, double center_y, double inner_radius, double outer_radius, double start_degrees, double end_degrees, nw_color tint);
  int64_t (*draw_text)(void *ctx, nw_font font, const char *text, double left, double top, double size, nw_color tint);
  double  (*measure_text)(void *ctx, nw_font font, const char *text, double size);
} nw_backend;
```

Fill that in and every widget works. The raylib backend in
`backends/raylib/nw_raylib.c` is the whole adapter and it is about ninety lines.

Corner radii are pixels, not raylib's fraction-of-the-shorter-side — converting
is the backend's job, since it is a fact about raylib rather than about buttons.

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
./build.sh              # library, header check, tests, and the raylib demo
./catalog               # the interactive component catalog
./catalog --check       # headless interaction check
./catalog --snapshot    # render each page to a PNG
```

Requires `coil` and, for the demo only, raylib 5. Consumers of the library need
neither: just `libnativewidgets.a` and `include/native_widgets.h`.

The demo loads fonts from `assets/fonts`, which it borrows from the flowline
repo, so run it from there or point it at your own.

## Layout

- `src/widgets.coil` — the library. Every widget, every token, no renderer.
- `include/native_widgets.h` — the public C API.
- `backends/raylib/` — the reference backend.
- `examples/catalog.c` — the component catalog, driven entirely through the
  public header. Doubles as the parity fixture.
- `tests/abi_test.c` — ABI and widget-behavior checks against a recording
  backend, no window required.
- `tests/parity/` — renders the catalog from flowline's original widgets and
  from this library, and compares pixels.

## Status

The library builds, exports 64 C symbols, and passes the ABI and interaction
tests. Geometry is pixel-exact against the original.

**Colors are wrong, because of a compiler bug, not a bug here.** Coil's LLVM
backend does not apply C-ABI lowering to indirect calls, so any struct smaller
than 16 bytes passed by value through a `(fnptr c …)` — which is exactly what
`nw_color` is — arrives corrupted. Written up with a minimal repro in
`coil/docs/BUG_CALLPTR_C_ABI.md`. The parity gate in `tests/parity/` currently
reports all four pages differing and should go green once that lands; it is the
check to re-run afterwards.

## Not done yet

- Theme tokens are fixed functions rather than a swappable `nw_theme`. The
  house dark palette is the only one.
- `nw_tabs`, `nw_button_group`, and `nw_toggle_group` take exactly three
  segments, matching the original design rather than generalizing past it.
- flowline still carries its own copy of the widgets; pointing it at this
  library is the next step.

## License

The component designs are an adaptation of the Native SDK canvas widgets. See
the flowline repo's `THIRD_PARTY_NOTICES.md` and
`licenses/native-sdk-Apache-2.0.txt`.
