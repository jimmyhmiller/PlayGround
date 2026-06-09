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

### 1b. Widen the layer stack beyond what `Style` can say — **DONE (2026-06-08)**
- Add `Layer::LinearGradient`, per-side `Border`, `spread`/`inset` `Shadow`, and
  multi-shader stacking to the compositor. These are things the flat `Style` literally
  cannot express — landing them proves the seam is wider than the old shim.

**Status:** shipped. `glaze::Layer` + `protocol::GlazeLayer` gained
`LinearGradient { angle, stops }`, `Border { …, sides: Sides }`, and
`Shadow { …, offset_x, spread, inset }`. New Glaze props (no parser change needed —
prop parsing is generic): `gradient [angle] <color> [offset] …`,
`border_top/right/bottom/left`, widened `shadow <color> [blur] [offset_y] [spread]`,
and `inset_shadow`. Rendering (`render.rs::paint_glaze_layers`): gradients + inset
shadows lower to generated WGSL through the **existing** `GlazeMaterial` path (rounded
clip for free); partial-side borders paint as sharp edge rects; outset shadows inflate
the rounded panel by `spread`. Verified end-to-end via the `glaze_ui` showcase row
(`widget-snapshot` PNG): both gradients, per-side border, inset "well", and
drop-shadow+spread all render through the real widget pipeline. 20 glaze tests + 26
widget_bevy lib tests green. NOTE: `1a` was already in the tree before this work
(ordered `glaze_layers` stack + hover/focus/press interaction uniforms). Multi-shader
stacking already worked via the ordered layer list. Still open in Phase 1: `1c` slots
(typed per-component style structs + `part {}` syntax), `1d` retrofit, and
`selected`/`disabled` state uniforms.

### 1c. Slots — typed per-component style structs — **STARTED (2026-06-08)**

**Status:** the Glaze language side is shipped, and the first concrete `<Name>Style`
struct (`BarStyle`) proves the contract end-to-end.
- **Glaze:** `part {}` blocks parse to `ast::Item::Part { name, body }` (generic prop
  parsing made this a one-line lookahead — `name {` is a part, `name args…` a prop).
  `Program::resolve_slots(style, variant, states) -> CompiledSlots { base, slots }`
  compiles the root box (`base`, top-level props outside any part) plus one
  `CompiledStyle` per named slot. Parts inherit top-level `let`s but don't leak their
  own to siblings; nested parts and parts-inside-overlays are hard errors. `:state {}`
  overlays work inside a part. The 3-pass resolver was extracted into a shared
  `compile_items`. (`CompiledSlots::slot(name)`, `slot_names()`.) 5 glaze tests.
- **Protocol/adapter (1d-Bar):** `protocol::BarStyle { track, fill }` (each
  `Option<Style>`, with `BarStyle::SLOTS`) is the first typed slot struct;
  `Element::Bar` carries `style: Option<BarStyle>`. `glaze_style::to_bar_style`
  validates each `part {}` name against `BarStyle::SLOTS` and returns a **load-time
  error naming the bad slot** (`bar has no slot \`trakc\``) — the typed-fields
  decision's payoff. `render::render_bar_at` does the value-driven sub-layout (track =
  full rect, fill = `value/max` of it) and paints each slot via `paint_style_background`
  (so a slot plan is a full `Style` — a Bar fill can be a 1b gradient), falling back to
  the flat `color`/`track` colors when a slot is absent. 2 adapter tests.
- **Verified:** `glaze_ui` Phase-1c section — an `Element::Bar` with a teal→violet
  gradient `fill` over an inset-shadow `track`, snapshot through the real pipeline
  (`widget-snapshot`). 25 glaze + 28 widget_bevy lib tests green.
- **1d retrofit COMPLETE (2026-06-08)** — all four components now slot-driven, hardcoded
  styling deleted from `render.rs` (fallbacks kept for unstyled use):
  - **`Bar`** `{track, fill}` — value-driven (fill = value/max of track). First slot struct.
  - **`Toggle`** `{track, knob}` — `track` resolved with the `:checked` state at the widget
    (CPU discrete-state model: pass `["checked"]` when on), `knob` x-position value-driven.
    `Element::Toggle.style` is now `Option<ToggleStyle>` (was `Option<Style>`; serde-graceful,
    all literal sites passed `None`).
  - **`Tabs`** `{strip, tab, tab_selected, indicator}` — `resolve_tabs_style(prog, name)`
    precomputes the resting + `:selected` `tab` plans (the doc's "per-state plan, host swaps"
    model); renderer swaps `tab_selected` in for the active tab and paints the `indicator`
    (a 1b gradient in the demo) under it.
  - **`Table`** `{panel, header, zebra}` — light retrofit of the three background surfaces;
    cell text / dividers / per-cell geometry stay the renderer's job.
  - Adapters: `to_bar_style`/`to_toggle_style`/`to_table_style`/`resolve_tabs_style`, each
    validating `part {}` names via `validate_slots(.., <Name>Style::SLOTS, ..)` → load-time
    error naming the bad slot.
  - Verified: `glaze_ui` Phase-1c/1d sections snapshot through the real pipeline. 25 glaze +
    31 widget_bevy lib tests green.
- **Still open:** Glaze `part:state` shorthand if wanted (today it's `part { :state { } }`);
  ListItem retrofit; `selected`/`disabled` *continuous* shader uniforms (only hover/focus/press
  exist — needs per-element state source into `paint_shader_layer`). Phase 1 is otherwise done;
  Phase 2 (net-new components: Slider/Checkbox/…) can start — Slider needs new drag-value event
  plumbing (`slider-change` HostEvent + a drag hit region; ClickKind has no drag-value today).

### The `<Name>Style` template (followed by all four retrofits)
1. `protocol`: `<Name>Style { slot: Option<Style>, … }` + `Name::SLOTS: &[&str]`; element field
   `style: Option<<Name>Style>`.
2. `glaze_style`: `to_<name>_style(&CompiledSlots) -> Result<<Name>Style, String>` calling
   `validate_slots`, or `resolve_<name>_style(prog, name)` when per-state plans are needed.
3. `render`: render arm computes slot rects (value-driven or Taffy) → `paint_style_background`
   each slot plan, falling back to the prior hardcoded paint when the slot is absent.

### 1c (original plan) — Slots — typed per-component style structs
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

- **`Slider`** `{id, value, min, max, step}` → `slider-change`. Slots `track`/`range`/`thumb`; states `hover`/`press`/`disabled`. First continuous control — needs a drag-track hit region in `render.rs`. **DONE (2026-06-08)** — the first net-new component on the slot system. `Element::Slider` + `SliderStyle{track,range,thumb}` + `HostEvent::SliderChange{id,value}`. Drag plumbing: `WidgetTargets.sliders: Vec<SliderTarget>` (each carries the value-mapping `value_x0`/`value_span` + `min/max/step`); `SliderTarget::value_at(x)` does the clamp+step-snap (unit-tested); `begin/update/end_slider_drag` systems map press/drag/release on `PaneContentPressed/Dragged/Released` and emit `SliderChange` through **both** the subprocess (`WidgetIO`) and rhai (`RhaiWidget::send_slider_change` → `HostToWorker::SliderChange` → `on_slider_change(id,value)`) channels. `render_slider_at` does value-driven sub-layout: full `track`, leading `range` to the thumb centre, `thumb` handle — each `paint_style_background`'d from its slot plan with a sensible fallback. Verified: `glaze_ui` slider (gradient range + glowing-thumb overlay shader) snapshot; value round-trips via `slider-change`. NOTE: when adding an `Element` variant, REBUILD all host bins (`widget-snapshot`, the main app) — a stale host rejects the new serde variant and silently drops the whole frame.
- **`Checkbox`** — **DONE (2026-06-08)**. `Element::Checkbox{id,label,checked,style}` + `CheckboxStyle{box,check}` (Rust field `square`; Glaze slot `box`). Reuses the **`toggle` event** (no new plumbing) — clicking emits `ClickKind::Toggle`. `:checked` resolved at the widget; the `check` mark renders only when checked. Layout mirrors Toggle (box + label row). Verified in `glaze_ui` (gradient tick, teal `:checked` border).
- **`Radio` / `RadioGroup`** — **DONE (2026-06-08)**. `Element::RadioGroup{id,options:Vec<TabItem>,selected,style}` + `RadioGroupStyle{ring,dot}` + new **`HostEvent::RadioSelect{id,option}`** (mirrors `TabSelect` across all paths: `ClickKind::RadioSelect`, subprocess press handler, rhai `HostToWorker::RadioSelect`→`on_radio_select`). Layout = column of label cells with left-padding reserving the ring (no nesting, like Tabs). Verified in `glaze_ui` (gradient `dot` on the selected option, accent ring affordance).
- **`Stepper` / `NumberInput`** — **DONE (2026-06-08)**. `Element::Stepper{id,value,min,max,step,style}` + `StepperStyle{field,button}` + new **`HostEvent::NumberChange{id,value}`** (wired like RadioSelect). The `−`/`+` buttons carry the **precomputed clamped target value** in `ClickKind::NumberChange{value}` — the renderer owns the arithmetic, so a click is a plain value-set (same trick as Toggle's `new_checked`). Arithmetic sub-layout (no Taffy children, like Slider): `[− button][value field][+ button]`, fixed `STEPPER_W`. Verified in `glaze_ui` (teal-bordered buttons, live value).
- **`Rating`** — deferred: poor slot fit (star glyphs aren't box-paintable; would degrade slots to glyph colors). Do later as a glyph-based special case if wanted.

**Phase 2 status: essentially DONE.** Slider, Checkbox, Radio, Stepper shipped — covering drag, toggle-reuse, new select-event, and number-change patterns. The slot system now has **8 components** (Bar, Toggle, Tabs, Table, Slider, Checkbox, RadioGroup, Stepper). Adapters: `to_{bar,toggle,table,slider,checkbox,radio,stepper}_style` + `resolve_tabs_style`. 38 widget_bevy lib + 25 glaze tests. **Next: Phase 3 — the floating overlay layer** (the big remaining gate for Select/Menu/Tooltip/Popover/Dialog/Sheet/Toast/DatePicker).
- **`Stepper` / `NumberInput`** `{id, value, min, max, step}` → `number-change`. Slots `field`/`button`/`button:press`. Reuses `Button` click; optional drag.
- **`Checkbox`** `{id, label, checked}` → reuses `toggle`. Slots `box`/`check`; `checked` drives `check` opacity (dynamic → free in Glaze).
- **`Radio` / `RadioGroup`** `{id, options:[{id,label}], selected}` → `radio-select` (mirrors `tab-select`). Slots `dot`/`ring`/`item:selected`.
- **`Rating`** `{id, value, max}` → `rating-change`. Slots `star`/`star:filled`.

## Phase 3 — Floating overlay layer + selection / menus / feedback

Build a `FloatingLayer` substrate: anchored content rendered above all panes on a
dedicated `RenderLayers`, with outside-click dismissal and a z-stack. This is the big
lift; each dependent is then cheap and slot-styled.

**SUBSTRATE DONE (2026-06-09) — and the first consumer (`Select`) ships.** Key finding:
the floating substrate already existed — terminal-bevy's `MENU_OVERLAY_LAYER` (32) + an
order-100,000 camera (used by the drawer / radial / context menu). Phase 3 is a *portal*:
widget Elements render their floating part onto that layer via the EXISTING render path.
- `WidgetOverlayLayer(usize)` resource (default 32) — the host reserves the layer +
  provides the camera (terminal-bevy already does; the snapshot tool now does too).
- Open state is **host-owned** (`WidgetOpenSelect` resource, one at a time) — the widget
  only tracks `value`. `ClickKind::SelectTrigger` toggles it (no widget event).
- `render_select_overlay` spawns a `WidgetOverlayRoot` at the trigger anchor converted
  content-local → window → overlay-world (`to_world(p)=(p.x-w/2, h/2-p.y)`), renders the
  menu via `paint_style_background`/`paint_rounded_panel` (full slot styling: `menu`,
  `item`, `item_selected`), and records item + trigger **window-space** rects.
- `stamp_overlay_layers` (PostUpdate, `.before(CheckVisibility)`) propagates the overlay
  `RenderLayers` to the root's descendants (paint helpers don't set layers themselves).
- `handle_overlay_input` reads raw mouse (like context_menu, not pane events, so clicks
  outside the pane still register): item-click → `SelectChange` + close; outside/Escape →
  close. The trigger rect is recorded so the toggling click is ignored by dismiss.
- `handle_widget_press` guard: while a dropdown is open on a pane, non-trigger presses
  route to the overlay (so elements under the menu don't fire).
- Verified: `widget-snapshot --open-select country` shows the dropdown floating below the
  trigger, escaping pane bounds, `item:selected` highlight on the chosen row.

- **`Select` / `Combobox`** `{id, options, value}` → `select-change`. **DONE.** Slots
  `trigger`/`menu`/`item`/`item:selected` (via `SelectStyle` + `resolve_select_style` dual
  resolve). `searchable` + `item:hover` not yet. The portal generalizes for free to:
- **`Menu` / `DropdownMenu` / `ContextMenu`** `{trigger, items}` → `menu-select`. Slots `menu`/`item`/`separator`/`shortcut`.
- **`Tooltip`** — **DONE (2026-06-09)**. `Element::Tooltip{label,text,style:TooltipStyle{bubble}}` —
  a link-styled in-pane `label` that shows a floating `text` bubble on hover (no event, passive).
  Proves the portal handles **hover-triggered** content: `update_tooltip_hover` (per-frame
  cursor→topmost-pane→hit-test, sets `ActiveTooltip`; only recomputes when a real cursor exists so a
  forced tooltip survives headless) + `render_tooltip_overlay` (bubble on the overlay layer, own
  `WidgetTooltipRoot` marker so it despawns independently of the select dropdown; `stamp_overlay_layers`
  now `Or<select-root, tooltip-root>`). `to_tooltip_style`; `widget-snapshot --show-tooltip` forces it.
  Verified: bubble floats below the label, escapes pane bounds. TODO: hover *delay*, arrow, wrap-any-element.
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
