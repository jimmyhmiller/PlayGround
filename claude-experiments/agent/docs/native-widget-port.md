# Native widget port

The port tracks Native SDK built-in components, not internal `WidgetKind`
variants. A component is complete only when all of these are true:

- Its composition and geometry follow the upstream component and house tokens.
- Pointer, keyboard, focus, disabled, and controlled-value behavior work.
- An end-to-end interaction check exercises the rendered component.
- Its reference render has been compared with the upstream dark reference.
- The application uses the shared component instead of separate chrome.

| Component | Composition | Interaction | Reference | Application |
| --- | --- | --- | --- | --- |
| Accordion | pending | pending | pending | pending |
| Alert | pending | not interactive | pending | pending |
| Avatar | pending | not interactive | pending | pending |
| Badge | pending | not interactive | pending | pending |
| Breadcrumb | pending | pending | pending | pending |
| Bubble | pending | pending | pending | pending |
| Button | pending | pending | pending | pending |
| Button group | pending | pending | pending | pending |
| Card | pending | pending | pending | pending |
| Checkbox | pending | pending | pending | pending |
| Combobox | pending | pending | pending | pending |
| Dialog | pending | pending | pending | pending |
| Drawer | pending | pending | pending | pending |
| Dropdown menu | pending | pending | pending | pending |
| Input | pending | pending | pending | pending |
| Pagination | pending | pending | pending | pending |
| Progress | pending | not interactive | pending | pending |
| Radio group | pending | pending | pending | pending |
| Resizable | pending | pending | pending | pending |
| Select | pending | pending | pending | pending |
| Separator | pending | not interactive | pending | pending |
| Sheet | pending | pending | pending | pending |
| Skeleton | pending | not interactive | pending | pending |
| Slider | pending | pending | pending | pending |
| Spinner | pending | not interactive | pending | pending |
| Switch | pending | pending | pending | pending |
| Table | pending | pending | pending | pending |
| Tabs | pending | pending | pending | pending |
| Textarea | pending | pending | pending | pending |
| Toggle | pending | pending | pending | pending |
| Toggle group | pending | pending | pending | pending |
| Tooltip | pending | pending | pending | pending |

Upstream source of truth:

- `src/primitives/canvas/widgets.zig`
- `src/primitives/canvas/tokens.zig`
- `src/primitives/canvas/events.zig`
- `src/primitives/canvas/widget_render_controls.zig`
- `src/primitives/canvas/widget_render_surfaces.zig`
- `docs/src/app/components/*`
- `docs/public/components/*-dark.webp`
