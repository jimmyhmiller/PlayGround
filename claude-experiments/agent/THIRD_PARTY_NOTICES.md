# Third-party notices

## Vercel Labs Native SDK

Two separate parts of flowline are adapted from the Native SDK.

### Widget catalog

The components in `src/native.coil` — button, input, textarea, checkbox, switch,
slider, toggle, radio, alert, avatar, badge, breadcrumb, bubble, card,
separator, skeleton, progress, spinner, table, accordion, tabs, button group,
toggle group, pagination, select, combobox, dropdown menu, tooltip, dialog,
drawer, sheet, and resizable — together with their theme tokens, are adapted
from the Native SDK canvas widgets:

- <https://github.com/vercel-labs/native/blob/main/src/primitives/canvas/widgets.zig>
- <https://github.com/vercel-labs/native/blob/main/src/primitives/canvas/tokens.zig>
- <https://github.com/vercel-labs/native/blob/main/src/primitives/canvas/widget_render_controls.zig>
- <https://github.com/vercel-labs/native/blob/main/src/primitives/canvas/widget_render_surfaces.zig>
- <https://github.com/vercel-labs/native/tree/main/docs/src/app/components>

Flowline's version has been rewritten in Coil against Raylib, with the
components' composition and geometry followed from the upstream dark reference
renders. The port is tracked in `docs/native-widget-port.md`.

These widgets have since been extracted into a standalone, renderer-independent
C library, which carries its own copy of this attribution.

### Text interaction

Flowline's text-entry editing model and interaction behavior in `src/main.coil`
are adapted from the Native SDK text interaction engine and AI chat composer:

- <https://github.com/vercel-labs/native/blob/main/src/primitives/canvas/text_interaction.zig>
- <https://github.com/vercel-labs/native/blob/main/src/primitives/canvas/events.zig>
- <https://github.com/vercel-labs/native/tree/main/examples/ai-chat-ts>

The original work is provided by Vercel Labs under the Apache License 2.0.
Flowline's version has been substantially modified and rewritten in Coil for a
Raylib application with centralized `AppState` transitions.

The applicable license is reproduced in
`licenses/native-sdk-Apache-2.0.txt`.
