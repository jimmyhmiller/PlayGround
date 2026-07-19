# Third-party notices

## Vercel Labs Native SDK

The component designs in this library — their composition, geometry, states, and
theme tokens — are adapted from the Native SDK canvas widgets:

- <https://github.com/vercel-labs/native>

Specifically, the components here follow these upstream sources:

- `src/primitives/canvas/widgets.zig`
- `src/primitives/canvas/tokens.zig`
- `src/primitives/canvas/events.zig`
- `src/primitives/canvas/widget_render_controls.zig`
- `src/primitives/canvas/widget_render_surfaces.zig`
- `docs/src/app/components/*` and `docs/public/components/*-dark.webp`, used as
  the visual reference for each component

The original work is provided by Vercel Labs under the Apache License 2.0. A
copy of that license is reproduced in `licenses/native-sdk-Apache-2.0.txt`.

### Modifications

This is not a copy of the upstream work. Per section 4(b) of the Apache License,
the changes are:

- Rewritten from Zig into Coil, and exposed as a C library rather than as part
  of an application.
- The renderer was removed entirely. Upstream draws to its own canvas; here every
  drawing operation goes through a caller-supplied `nw_backend` vtable, and the
  library links against no graphics library at all.
- Widget state ownership was inverted. Upstream widgets participate in a
  retained tree; these are stateless functions whose appearance is derived from
  the bounds and an `nw_interaction` the caller passes in.
- Application-specific content was replaced with caller-supplied data — table
  rows, tab and segment labels, pagination labels, select options, and dialog,
  drawer, sheet and resizable copy are all parameters rather than baked-in
  strings.
- Several components were split so callers control overlay ordering:
  `nw_select` / `nw_select_menu` / `nw_select_option`, and
  `nw_menu_surface` / `nw_menu_item`.
- Corner radii are expressed in pixels, and angles in degrees clockwise from
  three o'clock, rather than in any renderer's native convention.

The work passed through an intermediate stage in the flowline agent runner,
where these components were first ported to Coil against Raylib; this library
was extracted from that port.

## Fonts

This repository ships no fonts. The example drivers load IBM Plex Mono and IBM
Plex Sans from the flowline repository's `assets/fonts` directory at runtime.
Those fonts are licensed under the SIL Open Font License; if you vendor them
here, include `licenses/IBM-Plex-OFL.txt` alongside them.
