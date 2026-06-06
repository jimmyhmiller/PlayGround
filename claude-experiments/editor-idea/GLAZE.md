# Glaze — a staged style/shader language for the widget system

> Working name. A *glaze* is the vitreous finish fired onto a crafted surface — fits
> the Atelier theme and gives a clean `.glz` extension. Bikeshed later.

## Why a language (not a data schema)

The current styling model is **inline `Style` structs per `Element`** (`widget-bevy/src/protocol.rs`):
no reuse, no states, no variants, no inheritance, and "hover/focus" is hardcoded
procedurally in `render.rs`. We already produce styles *programmatically* (the Rhai
`render()` handler computes `Style` every frame) — so we're effectively living in
Netscape's **JSSS** model from 1996. The goal here is to lift the reusable,
retunable, *animatable* parts into a real language whose constraints are inverted from
CSS's (we hold the whole tree in memory, one renderer, no progressive-render limit),
so we can revive the powerful-but-lost ideas (DSSSL math, PSL96 conditionals, Houdini
paint) without selectors/specificity.

## The thesis: Glaze is a two-stage language; the stage boundary is the GPU upload

Every value has a **binding time**:

- **static** — depends only on tokens, variant params, attrs, inherited `env`.
- **dynamic** — transitively depends on a per-frame input: `time`, `dt`, `mouse`,
  `hover`, `focus`, `press`, `selected`, `size`, `resolution`.

The compiler does **binding-time analysis** and **partial-evaluates the static stage now**
(on save / on state change), leaving a residual dynamic program that is **codegen'd to
WGSL** with its free dynamic variables declared as uniforms. The author never writes
"this is CPU, this is GPU" — staging decides. (This is Futamura proj-1 specialization
of a style w.r.t. its static inputs; it's the same analysis under the partial
evaluators in `partial-new`.)

```
lex → parse → resolve(tokens/variants/env)
            → type + stage inference
            → partial-eval(static) → residual
            → lower:
                static residue  → layer-plan + uniform write-list   (CPU, re-eval on state change)
                dynamic residue → generated WGSL fragment(s)         (GPU, fed per frame)
```

The generated WGSL drops straight into the existing `DynamicMaterial` /
`UserBuffer` naga-introspection path in `style-bevy/src/dynamic.rs` (the host already
parses a `struct UserData{}`, learns field→offset, and binds by name via
`uniform_set`). So for own-quad shader layers there is **almost no new GPU plumbing** —
the net-new runtime work is per-element material instancing + draw-call keying.

## Surface syntax (brace + property, infix expressions)

```glaze
// ---- tokens: 3-tier, aliases resolve; all hot-reloadable ----
token gold.500       = oklch(0.72 0.11 85)
token slate.900      = oklch(0.16 0.01 250)
token accent.solid   = gold.500                 // semantic alias
token accent.hover   = lighten(accent.solid, 0.06)
token danger.solid   = oklch(0.58 0.16 25)
token surface.raised = oklch(0.28 0.01 250)
token border.subtle  = oklch(0.40 0.01 250 / 0.5)
token focus.ring     = accent.solid
token radius.md      = 8px

// ---- a style is a pure function (variant, state, attrs, env) -> layer stack ----
style card {
  fill   surface.raised
  radius radius.md
  border border.subtle 1px
  shadow shadow.sm
  pad    16px
}

style button(intent) {                          // `intent` is a static variant param
  fill   intent == danger ? danger.solid : accent.solid   // static branch -> folds
  pad    8px 12px
  radius radius.md

  overlay shader {                              // a glow layer composited on top
    let pulse = 0.5 + 0.5 * sin(time * 2)       // time -> dynamic
    emit smoothstep(0, 1, hover) * pulse * vec4(1, 1, 1, 0.25)   // hover -> dynamic
  }

  :focus { border focus.ring 2px }              // discrete state overlay
}
```

### Two kinds of "state", deliberately distinct

- **Discrete pseudo-states** `:hover / :focus / :press / :selected / :disabled` select
  an **alternate layer-plan** (CPU, chosen on state change). This is the CSS keeper.
- **Continuous dynamic builtins** `hover / focus / press` (0→1 eased amounts), plus
  `time / dt / mouse / size`, are **uniforms** consumed inside `shader{}`/`overlay{}`
  bodies for *animation within* a plan (GPU). This is the Houdini/Shadertoy keeper.

They're orthogonal: discrete state = *which* plan; continuous builtins = motion inside it.

## Type & stage system

- Base types: `scalar`, `length` (`px | % | em | fr`), `vec2..4`, `color` (oklch-native),
  `bool`, `token-ref`. HM-style inference, no annotations needed in practice.
- Binding-time lattice `static ⊑ dynamic`, inferred bottom-up. Seeds:
  - static: token refs, variant params, `attr.*`, `env.*`
  - dynamic: `time dt mouse hover focus press selected size resolution`
  - join: any operation with a dynamic operand yields dynamic.
- The math vocabulary is the **intersection of CSS-math and shader-math** so the same
  expressions run on either side of the stage boundary:
  `+ - * /`, `mix`, `clamp`, `smoothstep`, `step`, `min/max`, `sin/cos/floor/fract`,
  `length/dot/normalize`, swizzles, ternary `?:`.
- **Color arithmetic** (open, see below): `+` / `* scalar` operate in **linear-rgb**
  (light accumulation — what glows/shaders want); `mix / lighten / darken / saturate`
  operate in **oklch** (perceptual). Literals are `oklch(L C H [/ a])`, also `#rrggbb`.
- **No silent failure** (house rule): a type error, an unresolved token, or WGSL that
  won't compile renders a **loud error material** carrying the message — never silent
  black/pink.

## Worked example — what `button(intent: primary)` compiles to

Static fold: `intent == danger` is false → `fill` folds to the `accent.solid` constant.
`surface.raised`, `focus.ring`, `radius.md` all fold to constants. The `overlay shader{}`
references `time` and `hover` (dynamic) → lowered to WGSL.

**Compiled artifact (the IR the renderer consumes):**

```
CompiledStyle "button" variant {intent: primary} {
  box: { pad: [8,12], radius: 8 }
  plans: {
    base:   [ fill(lin #c9a96a), glow#0 ]
    :focus: [ fill(lin #c9a96a), glow#0, border(lin focus.ring, 2px) ]
  }
  shaders: {
    glow#0: { uniforms: [ time:f32 <dyn:clock>, hover:f32 <dyn:hover> ], wgsl: <below> }
  }
}
```

**Generated WGSL for `glow#0`** (fed by the existing `DynamicMaterial` UBO path):

```wgsl
struct UserData { time: f32, hover: f32, _pad: vec2<f32> }
@group(2) @binding(0) var<uniform> u: UserData;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let pulse = 0.5 + 0.5 * sin(u.time * 2.0);
    let a = smoothstep(0.0, 1.0, u.hover) * pulse * 0.25;   // white glow, premultiplied
    return vec4<f32>(a, a, a, a);
}
```

`time` ← engine clock, `hover` ← per-element eased hover amount. Both already exist as
engine-fed dynamic inputs for the dust shader; here they're bound per element.

## Runtime integration

- New crate **`crates/glaze`** — pure compiler (lex/parse/typecheck/stage/lower → IR +
  WGSL). No Bevy, mirrors how `editor-core` is pure logic. Unit-testable in isolation.
- **`style-bevy`** loads `*.glz` files, holds a `CompiledStyleSet`, hot-reloads on save
  (recompile → swap material shader handles + rebind uniforms). Generated WGSL feeds the
  existing `DynamicMaterial`.
- **`widget-bevy`** `Element`s reference a style by name + pass variant/attrs:
  `style: "button"`, `variant: { intent: "primary" }`. Inline `Style` stays as an
  escape hatch (highest precedence). The renderer's resolution step consumes the
  `CompiledStyle` (chooses the discrete-state plan, paints the layer-plan, instantiates
  `DynamicMaterial`s for shader layers). Both hosting paths (in-process Rhai + subprocess
  NDJSON) reference styles by name — styles live engine-side and are shared.
- **Live editing** = recompile on `.glz` save + per-frame dynamic uniform feed. A future
  structural editor / inspector edits the AST and rebinds named uniforms live
  (`glaze_set("button.glow.intensity", x)`), the strongest form of "runtime editable."

## Resolution precedence (flat layered merge, no specificity)

`component defaults  <  inherited env (color/font/size)  <  named style (+variant)  <
 discrete-state plan overlay  <  inline Style on the element`

Each layer is a partial result; later layers override only fields they set. Predictable;
no cascade arithmetic.

## Staged build plan (each stage is runnable & shippable)

0. **Skeleton** — `crates/glaze`: lexer + parser → AST + golden tests. No semantics yet.
1. **Static styles** — token resolution (3-tier) + type-check + static-only layer-plan IR
   (`fill/border/shadow/radius/pad`, no shaders). Wire into the `widget-bevy` renderer:
   `Element.style: "name"` resolves to a `CompiledStyle`. *Delivers reuse + live retuning
   for ordinary styles — a real win on its own.*
2. **Variants + discrete states** — variant params + `:hover/:focus/:selected` plans.
   Replace the hardcoded hover/focus substitution in `render.rs`.
3. **Staging + shaders (the payoff)** — binding-time inference + `shader{}`/`overlay{}` +
   WGSL backend → generated `DynamicMaterial`s. Per-element material instancing +
   draw-call keying. Loud error material.
4. **Live tooling** — hot-reload recompile path, inspector hooks, `glaze_set` uniform
   tweak, structural-editor integration.
5. **Component library** — express default component styles in Glaze; expand the standard
   component set (Select, Checkbox, Slider, Menu, Tooltip, Accordion, Card, Icon, …).

## Open questions (sensible defaults chosen; revisit)

- **Name / extension** — `Glaze` / `.glz` (placeholder).
- **Color arithmetic spaces** — linear-rgb for `+`/`*`, oklch for `mix`/`lighten`
  (proposed above). Confirm.
- **Variant param typing** — open enums (`intent == danger`) vs declared `variant intent
  { primary danger ghost }`. Declared gives exhaustiveness + editor affordances.
- **User functions / mixins** — allow `fn`/`mixin` in `.glz`, or keep styles flat?
- **Layout** — stays in the `Element` protocol (Taffy), or does Glaze own layout too?
  (L2 keeps the tree in Rhai/subprocess, so: stays in Element for now.)
- **Material instancing / batching** — share `DynamicMaterial`s keyed by
  `(compiled-shader, uniform-signature)` to bound draw calls; the main net-new risk.
- **Render-to-texture tier (backdrop/layer-effects)** — explicitly out of L2; revisit as
  L2.5 once per-element materials are proven.
```

