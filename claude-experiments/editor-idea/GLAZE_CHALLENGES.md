# Glaze styling challenges

A battery of deliberately hard styling/visual targets to pressure-test the Glaze
shader language. Each must be expressible as a Glaze `overlay shader {}` (or expose a
language gap we then fix). Rendered together by `glaze_gallery` (one snapshot, one tile
each).

Status: ☐ todo · ◐ partial · ☑ done

All ☑ — rendered together by `glaze_gallery` (snapshot `/tmp/glaze_gallery.png`).

| # | Challenge | Exercises | Status |
|---|-----------|-----------|--------|
| 1 | Linear gradient | `mix`, `uv` swizzle | ☑ |
| 2 | Radial gradient | `length`, `clamp`, `mix` | ☑ |
| 3 | Conic hue wheel | `atan2`, scalar hue→rgb | ☑ |
| 4 | Rounded-rect SDF + border | box SDF, `size`, `max/min/abs`, swizzle | ☑ |
| 5 | Neon ring (glow) | ring SDF, `exp` falloff | ☑ |
| 6 | Diagonal stripes | `fract`, `step` | ☑ |
| 7 | Checkerboard | `floor`, parity via `fract` | ☑ |
| 8 | Dot grid | cell `fract`, circle SDF | ☑ |
| 9 | Vignette | radial `smoothstep` darken | ☑ |
| 10 | Value noise | hash `fract(sin(dot()))`, bilinear `mix` | ☑ |
| 11 | Bevel / fake-3D button | vertical shade gradient | ☑ |
| 12 | Plasma | summed `sin`, hue map | ☑ |

### What the battery taught us
- The generic WGSL passthrough means any WGSL builtin (`atan2`, `exp`, `floor`, `fract`,
  `dot`, …) works in a shader as long as it isn't *statically* folded — so noise, SDFs,
  hue math, plasma all just work.
- **Fixed gap:** swizzling a token color (`slate.x`) — now folds to a scalar in `eval`.
- Standing constraints (motivate future work): WGSL strict typing on `clamp/smoothstep/mix`
  vector args; no `fn`/mixins (SDF helpers inlined); no loops (fBm/Voronoi unrolled); no
  texture/backdrop input (real frosted-glass / chromatic aberration out of reach).

## Known language gaps this surfaces (fix as we go)
- WGSL is strictly typed: `clamp/smoothstep/mix` need matching vector arg types
  (scalar bounds on a vector are invalid). Author scalar-wise or splat explicitly.
- No `fn`/mixins yet → SDF helpers must be inlined per shader (motivates user functions).
- No loops → multi-octave fBm / Voronoi must be hand-unrolled (or skipped).
- No texture/backdrop input → real frosted-glass, chromatic aberration, bloom are out.
