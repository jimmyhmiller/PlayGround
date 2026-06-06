# Glaze Components — expansion plan for the widget vocabulary

> Companion to `GLAZE.md` (the styling language) and `AUTHORING.md` (the handler
> model). This doc inventories the widget `Element` set against the major component
> libraries, identifies the gaps, and lays out a phased plan to close them **without
> baking any component's look into `render.rs`** — i.e. every new component must be
> first-class stylable through Glaze.

## The load-bearing constraint

The real work here is not "add 25 components" — it is **"add 25 components whose every
surface is reachable from Glaze, without baking any look into `render.rs`."** `GLAZE.md`
names the anti-pattern we are escaping:

> "hover/focus is hardcoded procedurally in `render.rs`."

The first instinct is to extend the existing styling seam. But that seam — **one
`protocol::Style` per `Element`** — is itself the wrong shape, and we should fix it
*before* building on it. The next section works out the right seam; everything after
depends on it.

### Why the current `Style` seam is wrong (not just narrow)

`glaze_style.rs::to_style()` maps `glaze::CompiledStyle` → `protocol::Style`. But
`CompiledStyle` is already richer than `Style`, and the conversion is **lossy**:

- **Glaze produces an ordered layer stack** — `CompiledStyle { box_, layers: Vec<Layer> }`
  (`eval.rs:77`). `Style` has one fixed slot each for background / border / shadow /
  shader, so `to_style` collapses the stack: the `Layer::Fill` arm is even commented
  *"last fill wins"*. A base-fill + gradient-overlay + inner-glow stack truncates to one
  fill.
- **States are structurally broken, not just hardcoded.** `CompiledShader` carries
  `used: Vec<Builtin>` — exactly which per-frame inputs (`hover`, `press`, `focus`,
  `selected`, `time`, `mouse`…) the generated WGSL reads. `to_style` **drops `used`**,
  and `GlazeUniforms` only feeds `size/resolution/radius`. So a Glaze `:hover` shader
  layer compiles fine and then **can never receive its `hover` input** at runtime. The
  whole point of Glaze's staging — "dynamic residue → WGSL with its free dynamic
  variables declared as uniforms" — dies at this boundary.
- **Composite components have many surfaces.** A slider has track / range / thumb; a
  select has trigger / menu / item with per-item hover & selected. One flat `Style` per
  element cannot address them, and a `Map<part, Style>` would only make the *same lossy
  struct* N-wide.

So `Style` is a **degraded shim Glaze is forced through**, not Glaze's natural output.
Building the component system on it would cap every component at the shim's ceiling.

---

## The right seam: a `StylePlan` (box + ordered layer stack)

Invert the dependency. Today `Style` is the seam and Glaze compiles *down* into it.
Instead, make the **layer-plan the renderer's seam**, and make `Style` (and Rhai, and
subprocess JSON) *front-ends that lower up into it* — the same target Glaze already
emits. This is exactly GLAZE.md's stated lowering target ("static residue → layer-plan +
uniform write-list; dynamic residue → WGSL fragments"); we just stop hiding it behind
`Style`.

### The datatype

```
struct StylePlan {
    box:    BoxStyle,        // taffy/box-model: padding, radius, flex_*, width/height…
    layers: Vec<Layer>,      // painted back-to-front, BELOW children unless `overlay`
}

enum Layer {
    Fill(Rgba),
    LinearGradient { stops: Vec<(f32, Rgba)>, angle: f32 },   // NEW — Style can't express
    Border { color: Rgba, width: f32, sides: Sides },         // per-side
    Shadow { color: Rgba, blur: f32, spread: f32, offset: Vec2, inset: bool },
    Image  { src: ImageRef, fit: Fit, tint: Option<Rgba> },
    Shader {
        wgsl:    String,           // compiled fragment body
        inputs:  Vec<Builtin>,     // the dropped `used` set — host feeds these as uniforms
        statics: Vec<(String, Value)>, // static-folded uniform write-list
        overlay: bool,             // composite over children vs. under
    },
}
```

This is essentially `glaze::CompiledStyle` promoted into the protocol as the canonical
type, with `Layer` widened (gradient, per-side border, spread/inset shadow, image,
shader-with-inputs). The renderer gets ONE entry point, `paint_plan(&StylePlan)`, that
walks `layers` in order.

### `Style` becomes sugar that lowers into a plan

`protocol::Style` stays — it is the **simple hand-authoring path** for Rhai/subprocess
widgets and we are not taking that away. It gains a single method, `Style::into_plan()`:

| `Style` field | lowers to |
|---|---|
| `background` | `Layer::Fill` |
| `background_image` | `Layer::Image` |
| `border` | `Layer::Border` |
| `shadow` | `Layer::Shadow` |
| `shader` | `Layer::Shader` (now carrying `inputs`) |
| box-model fields | `BoxStyle` |

`glaze_style.rs::to_style()` is **deleted**; Glaze's `CompiledStyle` *is* a `StylePlan`
(a near-identity `to_plan()` replaces it, losing nothing). Advanced producers may emit a
raw `StylePlan` directly if they want the full stack; everyone else keeps writing
`Style`. Net: three front-ends (`Style` sugar, Glaze, raw plan), **one renderer seam.**

### State styling: binding time decides (NOT "always a shader")

A `:hover`/`:press`/`:selected`/`:focus`/`:disabled` rule is handled by whichever of two
mechanisms GLAZE.md's **binding-time analysis** says it is — and the common case must be
the cheap one:

- **Discrete state rule** (the value depends only on the state, e.g.
  `item:hover { fill accent.hover }`) → compile to a small set of **precomputed per-state
  `StylePlan`s**, and the host **swaps the plan on the state-change event** (CPU). These
  are plain fills/borders that **batch normally** — no per-element material, no extra
  draw call. This is the default, because it is by far the most common rule.
- **Continuous-dynamic state rule** (the value also depends on a per-frame input —
  `time`, `mouse`, or a smoothed transition: `overlay shader { emit pulse(time)*hover }`)
  → lowers to a `Layer::Shader` whose `inputs` declare the builtins it reads; the host
  feeds a per-frame `StateInputs { hover, focus, press, selected, disabled }` (each a
  smoothable 0..1) into the uniform buffer.

This is what **fixes the dropped-`used` bug** (the shader path finally receives its
inputs) *and* keeps hover/focus out of `render.rs` *without* turning every hoverable row
into its own draw call. The earlier draft made GPU-uniform the default; that was wrong —
a 200-row table with a discrete hover fill would have become 200 un-batched materials.
Binding time is the correct discriminator, and it's already what the language computes.

**Who owns the transition.** Smooth fades (hover ramping 0→1 over ~120ms) need per-element
animation state. The host keeps a tiny `StateInputs` value per styled box and eases it
each frame; discrete CPU swaps read the eased value to pick/crossfade plans, shader layers
read it as a uniform. This animation bookkeeping is host-side infra Phase 1 must include —
it is not free and is not in `render.rs` today.

### Slots are named boxes, each with its own plan

A component is not "one quad with one plan." It is a small **tree of named boxes**, and
each box carries a `StylePlan`. A "slot" is just a box name a Glaze style can target:

```glaze
style slider {
  track { fill surface.sunken; radius 999px; height 4px }
  range { fill accent.solid }
  thumb {
    fill accent.solid; radius 999px; size 14px
    overlay shader { emit smoothstep(0,1, hover) * glow(accent.solid, 0.3) }
  }
}
```

compiles to a **typed per-component style struct** — `SliderStyle { track: StylePlan,
range: StylePlan, thumb: StylePlan }` — *not* a stringly-typed map. The component
definition owns the *structure* (a slider always has those three boxes in that
arrangement); Glaze owns the *look* of each. Each component declares its own
`<Name>Style` struct whose fields are exactly its slots; the element carries
`style: Option<SliderStyle>` (default = its own box). The `Style`-sugar and raw-plan
producers populate the same typed fields.

**Decision: typed fields, not a `Map<name, StylePlan>`.** The renderer reads
`style.track` directly — no lookup, no `Option` per slot to thread, no silent miss on a
typo. Glaze is string-keyed at parse time, so the type boundary lives in **one place**:
the Glaze→protocol adapter validates each `part {}` name against the component's known
slots and **rejects an unknown slot with a load-time diagnostic** (`slider has no slot
'thmub'`) surfaced to the author on `.glz` (hot-)reload — not a value that silently never
renders. (Glaze reloads from disk at runtime, so this is a Glaze-compile diagnostic, not
a Rust compile error.) Same for subprocess JSON — serde rejects unknown keys. The cost is
a `<Name>Style` struct + a small slot-name→field adapter per component; worth it for the
errors and discoverability.

### Slots are geometry, not only paint

A slot is a *box*, so it has a position and size as well as a `StylePlan` — and the doc
must not pretend styling is the whole story. `layout.rs` builds a Taffy tree from
`Element`s, but slot boxes are internal sub-structure, not `Element`s. So every
multi-slot component owns a **sub-layout step** that produces a rect per slot before
`paint_plan` runs on each. Two cases:

- **Flow slots** (e.g. a `Dialog`'s `title`/`body`/`footer`) → the component emits
  internal Taffy nodes and lets flexbox place them.
- **Value-driven slots** (e.g. a `Slider`'s `range` width and `thumb` x are functions of
  `value`; a `Bar`'s `fill`; a circular `Gauge`'s arc) → **not** flexbox; the render arm
  computes the rect arithmetically from the component's data.

This means a new component's render arm is "compute slot rects → `paint_plan` each slot,"
and the value-driven geometry is real per-component logic. Phase 1's retrofit widgets
(`Toggle` knob travel, `Bar` fill, `Tabs` indicator) are all value-driven, so they pin
this pattern down too.

### What this buys

- Glaze's full expressiveness (layer stacks, gradients, multiple shaders, per-state,
  per-slot) reaches the GPU **losslessly**.
- One renderer path instead of per-component painting; new components stop fighting a
  flat struct.
- Hand-authors keep the easy `Style` JSON; nothing they write breaks.
- The dropped-`used` state bug is fixed as a *consequence* of the design, not a patch.

### The cost (be honest)

- `render.rs` grows a layer-stack compositor + per-layer uniform binding (vs. today's
  fixed background/border/shadow/one-shader paint).
- The subprocess NDJSON protocol gains an optional `StylePlan` shape; we decide whether
  subprocess widgets may emit raw plans or only `Style` sugar (lean: sugar by default,
  raw plan opt-in).
- `GlazeUniforms` must generalize from `{size, resolution, radius}` to carry the folded
  `statics` write-list + the `StateInputs` block.
- **Entity / draw-call budget.** `paint_style_background` spawns a sprite per visual, so
  a plan of fill+gradient+border+shadow+shader is up to 5 entities per box, and a 5-slot
  component up to ~25 per instance — multiplied across a busy canvas. Phase 1 must set an
  explicit budget and a batching plan: collapse adjacent solid fills/borders into a single
  rounded-rect material where possible, share `GlazeShaderCache` handles (already keyed by
  body), and keep the discrete-state path on batchable sprites (see state styling above).
  This is the **single biggest practical risk** in the seam change.

## Current vocabulary (20 Elements + 3 Canvas items)

| Category | Elements |
|---|---|
| Layout | `Vstack`, `Hstack`, `Frame`, `Scroll`, `Spacer`, `Divider`, `Canvas` |
| Typography | `Text`, `Link`, `Badge` |
| Controls | `Button`, `Toggle`, `Input`, `TextArea`, `SwatchButton` |
| Data display | `Table`, `ListItem`, `Bar` (determinate progress) |
| Navigation | `Tabs` |
| Color | `Swatch`, `SwatchButton` |
| Canvas items | `Sprite`, `Rect`, `Text` |

Backing infrastructure already in place: the `Style` struct (box model + Taffy flex +
Glaze `ShaderSpec`), Taffy layout in `layout.rs`, and the handler model
(`on_click/toggle/tab_select/input_*/hover/drag/key/frame/bus/message`).

## Coverage vs. shadcn/ui, egui, SwiftUI

✅ have · 🟡 partial · ❌ missing

| Component | shadcn | egui | SwiftUI | Us |
|---|:-:|:-:|:-:|:-:|
| Stack / VStack / HStack | – | ✅ | ✅ | ✅ |
| Text / Label | ✅ | ✅ | ✅ | ✅ |
| Button | ✅ | ✅ | ✅ | ✅ |
| Link | – | ✅ | ✅ | ✅ |
| Badge / Tag | ✅ | – | – | ✅ |
| Switch / Toggle | ✅ | ✅ | ✅ | ✅ |
| Text input | ✅ | ✅ | ✅ | ✅ |
| Textarea | ✅ | ✅ | ✅ | ✅ |
| Table / DataTable | ✅ | ✅ | ✅ | ✅ |
| Tabs | ✅ | – | ✅ | ✅ |
| Progress (linear) | ✅ | ✅ | ✅ | ✅ `Bar` |
| Divider / Separator | ✅ | ✅ | ✅ | ✅ |
| **Checkbox** | ✅ | ✅ | – | ❌ |
| **Radio / RadioGroup** | ✅ | ✅ | ✅ | ❌ |
| **Slider** | ✅ | ✅ | ✅ | ❌ |
| **Stepper / NumberInput** | ✅ | ✅ | ✅ | ❌ |
| **Select / Combobox** | ✅ | ✅ | ✅ | ❌ |
| **Dropdown / Context Menu** | ✅ | ✅ | ✅ | ❌ |
| **Tooltip** | ✅ | ✅ | ✅ | ❌ |
| **Popover / HoverCard** | ✅ | – | ✅ | ❌ |
| **Dialog / Modal / Sheet / Alert** | ✅ | ✅ | ✅ | ❌ |
| **Toast / Notification** | ✅ | – | – | ❌ |
| **Accordion / Collapsible / Disclosure** | ✅ | ✅ | ✅ | ❌ |
| **Avatar** | ✅ | – | – | ❌ |
| **Image (inline flow)** | ✅ | ✅ | ✅ | 🟡 Canvas/bg only |
| **Spinner (indeterminate)** | – | ✅ | ✅ | ❌ |
| **Gauge / Ring progress** | ✅ | – | ✅ | ❌ |
| **Skeleton** | ✅ | – | ✅ | ❌ |
| **SegmentedControl / ToggleGroup** | ✅ | – | ✅ | 🟡 `Tabs` |
| **Rating** | ✅ | – | – | ❌ |
| **Card / GroupBox / Section** | ✅ | ✅ | ✅ | 🟡 `Frame` |
| **Breadcrumb** | ✅ | – | – | ❌ |
| **Pagination** | ✅ | – | – | ❌ |
| **Tree / Outline** | – | ✅ | ✅ | ❌ |
| **Command palette** | ✅ | – | – | ❌ |
| **Calendar / DatePicker** | ✅ | ✅ | ✅ | ❌ |
| **ColorPicker (full)** | – | ✅ | ✅ | 🟡 swatches |
| **Chart / Plot** | ✅ | ✅ | ✅ | 🟡 `Bar` |
| **Resizable / Splitter** | ✅ | ✅ | – | ❌ |
| **Sidebar / NavMenu** | ✅ | ✅ | ✅ | ❌ |
| **Markdown / RichText** | – | ✅ | ✅ | ❌ |

Two structural gaps dominate the list:

1. **No slot-aware styling** — composite components can't be styled per-part/per-state
   from Glaze.
2. **No floating overlay layer** — ~8 components (`Select`, `Menu`, `Tooltip`,
   `Popover`, `Dialog`, `Sheet`, `Toast`, `DatePicker`) need content that escapes pane
   flow-layout and pane bounds. Today the only "overlay" in the code is the unrelated
   pane-edit command UI (`lib.rs`).

---

## Phase 1 — Make `StylePlan` the seam (foundational, gates everything)

Goal: `render.rs` paints **only** by walking a `StylePlan`'s layer stack; `Style`,
Glaze, Rhai, and subprocess all lower into `StylePlan`; states are GPU uniforms; slots
are named boxes. Sequenced so each step is independently shippable and non-breaking.

### 1a. Introduce `StylePlan` + `paint_plan` (no behavior change)
- Add `StylePlan { box, layers: Vec<Layer> }` and the widened `Layer` enum to
  `protocol`.
- Add `Style::into_plan()`; route today's single-`Style` rendering through a new
  `paint_plan()` that supersedes `paint_style_background`. Existing widgets render
  identically — they just travel through the plan now.
- Replace `glaze_style.rs::to_style()` with `to_plan()` (near-identity from
  `CompiledStyle`); **stop flattening the layer stack**.
- **Fix the dropped-`used` bug**: `Layer::Shader` carries `inputs: Vec<Builtin>`;
  generalize `GlazeUniforms` to a `StateInputs` block + folded `statics` write-list, and
  have the host feed per-frame `hover/focus/press/selected/disabled`. After this step,
  Glaze `:hover`/`:press` shader layers actually receive their inputs for the first time.

### 1b. Widen the layer stack beyond what `Style` can say
- Add `Layer::LinearGradient`, per-side `Border`, `spread`/`inset` `Shadow`, and
  multi-shader stacking to the compositor. These are things the flat `Style` literally
  cannot express — landing them proves the seam is wider than the old shim.

### 1c. Slots — typed per-component style structs
- Each component declares a `<Name>Style` struct whose fields are exactly its slots
  (`SliderStyle { track, range, thumb }`), each field a `StylePlan`; the element carries
  `style: Option<<Name>Style>` (default field = the element's own box). Keep the existing
  `Style` sugar for single-surface widgets, so every current widget keeps working
  unchanged.
- The Glaze→protocol adapter maps each parsed `part {}` to the matching typed field and
  **errors on an unknown slot name** (typo → load-time Glaze diagnostic on reload, not a
  silent no-render).
- Extend the Glaze surface syntax so a style block names parts and pseudo-states, each
  compiling to one `StylePlan` field:

  ```glaze
  style select {
    trigger { fill surface.raised; radius radius.md; border border.subtle 1px }
    menu    { fill surface.raised; shadow shadow.md }
    item            { pad 6px 10px }
    item:hover      { fill accent.hover }    // dynamic → shader layer, hover uniform
    item:selected   { fill accent.solid }    // selected uniform
  }
  ```
- Resolution chain in `render.rs`: `slot("track")` → explicit slot plan → default
  `style` plan → active theme token. Unstyled components still look correct.

Reference slot vocabularies for the components built later:

| Component | Slots |
|---|---|
| `Slider` | `track`, `range`, `thumb` |
| `Select` | `trigger`, `menu`, `item`, `indicator` |
| `Checkbox` | `box`, `check`, `label` |
| `Toggle` (retrofit) | `track`, `knob` |
| `Bar` (retrofit) | `track`, `fill` |
| `Tabs` (retrofit) | `strip`, `tab`, `indicator` |
| `Table` (retrofit) | `header`, `row`, `row:zebra`, `cell` |
| `Dialog` | `scrim`, `panel`, `title`, `body`, `footer` |
| `Chart` | `axis`, `grid`, `series[n]`, `point`, `label` |

### 1d. Retrofit (proves the contract on shipping widgets)
Move `Toggle` (track/knob), `Bar` (track/fill), `Tabs` (strip/indicator), and `Table`
(header/row/zebra) off their hardcoded `render.rs` styling onto their typed
`<Name>Style` structs. Validates the design and **deletes existing hardcoded look from
`render.rs`**. These four are also the first concrete `<Name>Style` structs, so they
pin down the slot-name→field adapter pattern the rest of the components follow.

---

## Phase 2 — Numeric & continuous controls

No overlay needed; runs alongside Phase 1 as its proving ground. Reuse
`on_drag`/`on_release`; one new `*-change` event each. All slot-styled from day one.

- **`Slider`** `{id, value, min, max, step}` → `slider-change`. Slots `track`/`range`/`thumb`; states `hover`/`press`/`disabled`. First continuous control — needs a drag-track hit region in `render.rs`.
- **`Stepper` / `NumberInput`** `{id, value, min, max, step}` → `number-change`. Slots `field`/`button`/`button:press`. Reuses `Button` click; optional drag.
- **`Checkbox`** `{id, label, checked}` → reuses `toggle`. Slots `box`/`check`; `checked` drives `check` opacity (dynamic → free in Glaze).
- **`Radio` / `RadioGroup`** `{id, options:[{id,label}], selected}` → `radio-select` (mirrors `tab-select`). Slots `dot`/`ring`/`item:selected`.
- **`Rating`** `{id, value, max}` → `rating-change`. Slots `star`/`star:filled`.

## Phase 3 — Floating overlay layer + selection / menus / feedback

Build a `FloatingLayer` substrate: anchored content rendered above all panes on a
dedicated `RenderLayers`, with outside-click dismissal and a z-stack. This is the big
lift; each dependent is then cheap and slot-styled.

- **`Select` / `Combobox`** `{id, options, value, searchable}` → `select-change`. Slots `trigger`/`menu`/`item`/`item:hover`/`item:selected`/`indicator`.
- **`Menu` / `DropdownMenu` / `ContextMenu`** `{trigger, items}` → `menu-select`. Slots `menu`/`item`/`separator`/`shortcut`.
- **`Tooltip`** — wrap-any-element + hover delay (reuses `on_hover`). Slots `bubble`/`arrow`.
- **`Popover` / `HoverCard`** — arbitrary floating child. Slot `surface`.
- **`Dialog` / `Sheet` / `Alert`** `{open, title, body, actions}` — modal scrim + focus trap. Slots `scrim`/`panel`/`title`/`body`/`footer`.
- **`Toast`** — transient, auto-dismiss via `on_frame` timer; emitted programmatically. Slots `toast`/`toast:enter`/`toast:exit`.

## Phase 4 — Rich data widgets

- **`Chart`** (line / bar / area / sparkline) — built on `Canvas` `Rect`/`Text`, generalizing `Bar`. Slots `axis`/`grid`/`series[n]`/`point`/`tooltip`. Per-series color/glow as Glaze **shader slots**.
- **`DatePicker` / `Calendar`** — day-cell grid inside a `Popover`. Slots `cell`/`cell:today`/`cell:selected`/`cell:disabled`/`header`.
- **`ColorPicker`** — full HSV. The SV-square and hue-strip are natural Glaze **shader slots** (gradients computed in WGSL). We already render swatches and parse oklch.

## Phase 5 — Composites & navigation (built from the above)

`Accordion`/`Collapsible`/`Disclosure` (`collapse-toggle`), `Tree`/`Outline` (recursive
`ListItem` + disclosure), `Card` (header/body/footer slots), `Avatar` (`Image`/initials
in a circle), `Image` (inline flow — lift the Canvas sprite texture path), `Spinner`
(indeterminate; `on_frame` rotation), `Skeleton` (shimmer = Glaze shader slot),
`SegmentedControl` (`Tabs` semantics as a control), `Breadcrumb`, `Pagination`,
`Command palette` (ties into the existing suggestion-drawer / `tbsuggest` infra),
`Resizable`/`Splitter`.

---

## Critical path

All four named priorities (Select/Menus, Slider/numeric, Charts/DatePicker,
Dialog/Toast/Tooltip) depend on the same two foundations:

1. **Slot-based styling** (Phase 1) — so nothing is hardcoded.
2. **The overlay layer** (Phase 3) — for Select / Menu / Dialog / Toast / DatePicker.

So the critical path is **Phase 1 → Phase 3**, with **Phase 2** (numeric controls, no
overlay) running in parallel as the proving ground for slots, and **Phase 4** riding on
both.

## How each new component plugs in (mechanical checklist)

For every component:
1. New `Element` variant + its typed `<Name>Style` struct (one `StylePlan` per slot) in
   `protocol.rs`.
2. Slot-name→field entry in the Glaze→protocol adapter (so `part {}` names validate).
3. Sub-layout: emit internal Taffy nodes for flow slots, or compute value-driven slot
   rects in the render arm (`layout.rs` / `render.rs`).
4. Render arm in `render.rs` = compute slot rects → `paint_plan` each slot. Styling comes
   **only** from the resolved per-slot plans — never hardcoded.
5. Optional new `*-change` event in the handler table (`AUTHORING.md`) for both the Rhai
   and subprocess hosts.

(So a component is ~5 coordinated edits, not 3 — the `<Name>Style` struct and the adapter
entry are the price of the typed-fields decision.)

## Risks & open questions

- **Draw-call budget (biggest).** See "The cost" above. Must be measured early — a
  Slider or Table that 10×'s entity count is a real regression even if it looks right.
- **State-transition smoothing is host-side infra.** Per-box eased `StateInputs` must
  exist before any `:hover` fade works; it is not in `render.rs` today and Phase 1 owns
  building it.
- **Cross-component token / `env` inheritance.** A `Dialog` containing a `Select`: does
  the inner component inherit the outer's tokens/`env`? GLAZE.md has `env` inheritance for
  styles; the *component nesting* story (which env a child slot resolves against) is
  unspecified and needs a decision before Phase 3's composites.
- **Subprocess `StylePlan` is mostly theoretical.** Subprocess/Rhai authors will use the
  `Style` sugar, not hand-write WGSL layer stacks. The "three front-ends" framing is
  honest only because sugar lowers to a plan — the raw-plan-over-the-wire path may never
  see real use. Fine, but don't build elaborate subprocess plan-authoring ergonomics.
- **Big-bang ordering risk.** All new components gate on Phase 1 (the seam rewrite). 1a is
  deliberately a no-behavior-change step to de-risk, but if the layer-stack compositor or
  the draw-call budget proves harder than expected, *zero* new components ship until it
  lands. Mitigation: keep 1a/1b/1c independently revertible; do not start Phase 2 until 1a
  is green in the retrofit widgets.
