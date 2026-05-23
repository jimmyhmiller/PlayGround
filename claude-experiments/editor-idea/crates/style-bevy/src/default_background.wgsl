// Default dust overlay — projects that haven't dropped their own
// `background.wgsl` into `<project>/.editor/shaders/` get this. Runs
// as a full-canvas quad ABOVE the panes (z=900); blend mode means
// alpha=0 pixels are fully transparent, so dust speckles fall on top
// of windows, terminals, sidebars — anything.
//
// `proj.dust_seconds` drives intensity; at 0 the output is fully
// transparent (no override). At 24h it's a visible grain.

#import bevy_sprite::mesh2d_vertex_output::VertexOutput
#import style_bevy::prelude::{world, proj, theme, wipe_mask, wipe_sampler}

// Cheap hash, good enough for static-ish noise dust. Not for security.
fn hash21(p: vec2<f32>) -> f32 {
    let q = fract(p * vec2(123.34, 456.21));
    let r = q + dot(q, q + 78.233);
    return fract(r.x * r.y);
}

fn dust_amount(seconds: f32) -> f32 {
    // 0 at fresh, 1 at ~24h. Square-root curve so the first hour is
    // visibly different from "just touched it" without making old
    // projects pitch black.
    let hours = seconds / 3600.0;
    return clamp(sqrt(hours / 24.0), 0.0, 1.0);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Quad covers the full window — `in.uv` ranges 0..1 across the
    // canvas. Resolution lets us scale noise so it doesn't stretch.
    let px = in.uv * world.resolution;

    // Per-project opt-out: theme.rhai sets `dust_intensity` (default 1.0).
    // Setting it to 0 disables dust on this project entirely; other
    // values scale the visible amount.
    let dust = dust_amount(proj.dust_seconds) * theme.dust_intensity;
    if (dust <= 0.001) {
        // Fast path: nothing to paint. Letting it through would still
        // pass alpha=0 to the blend stage; this just saves a few ALU
        // ops on the no-override case.
        return vec4(0.0, 0.0, 0.0, 0.0);
    }

    // Three octaves of cheap value noise summed for a non-uniform
    // grain. Each cell ends up with a slightly different intensity.
    let n0 = hash21(floor(px * 0.50));
    let n1 = hash21(floor(px * 1.10) + 17.0);
    let n2 = hash21(floor(px * 2.30) + 53.0);
    let grain = (n0 * 0.5 + n1 * 0.3 + n2 * 0.2);

    // Slow drift so the texture isn't completely static — old projects
    // should feel "alive but neglected", not frozen.
    let drift = 0.5 + 0.5 * sin(world.time * 0.02 + px.x * 0.001);

    // Persistent finger-smear: the host paints recent mouse motion
    // into `wipe_mask` (R channel = how much dust has been cleaned
    // here). Once the user wipes a region, it stays clean until the
    // host clears the mask. The mask is UV-mapped across the whole
    // canvas, so sampling at `in.uv` lands at the right spot
    // regardless of window aspect.
    let wipe = clamp(textureSample(wipe_mask, wipe_sampler, in.uv).r, 0.0, 1.0);

    // ---- Windex spray-and-wipe (temporary, doesn't modify the mask) ----
    //
    // Tune freely — saving this file hot-reloads via AssetServer.
    //
    // Three phases keyed off `world.windex_progress` (0..1):
    //   0.00..0.25   spray drops fly in; a streak appears at front.
    //   0.25..0.85   wipe-front sweeps left→right; behind = clean.
    //   0.85..1.00   sheen fades; dust returns to natural state.
    var windex_clean = 0.0;
    var windex_sheen = 0.0;
    var spray_alpha = 0.0;
    if (world.windex_active != 0u) {
        let p = world.windex_progress;
        let res = world.resolution;
        let band = 220.0;
        let ease = p * p;
        let front_x = -band + ease * (res.x + band * 2.0);
        let cleared = smoothstep(front_x + 12.0, front_x - 12.0, px.x);
        let sheen_band = smoothstep(front_x - band, front_x, px.x)
                       - smoothstep(front_x, front_x + 18.0, px.x);
        let fade = 1.0 - smoothstep(0.85, 1.0, p);
        windex_clean = cleared * fade;
        windex_sheen = max(0.0, sheen_band) * fade;

        if (p < 0.25) {
            let spray_p = p / 0.25;
            for (var i = 0u; i < 6u; i = i + 1u) {
                let fi = f32(i);
                let drop_t = clamp((spray_p - fi * 0.08) * 1.5, 0.0, 1.0);
                if (drop_t <= 0.0) { continue; }
                let hx = fract(sin(fi * 12.9898 + 1.2) * 43758.5453);
                let hy = fract(sin(fi * 78.233 + 4.7) * 43758.5453);
                let cx = (0.55 + hx * 0.30) * res.x;
                let cy = (0.20 + hy * 0.60) * res.y;
                let radius = 12.0 + 24.0 * drop_t;
                let d = distance(px, vec2(cx, cy));
                let drop = max(0.0, 1.0 - d / radius);
                spray_alpha = spray_alpha + drop * drop * (1.0 - drop_t * 0.5);
            }
            spray_alpha = clamp(spray_alpha, 0.0, 1.0);
        }
    }

    let effective_wipe = clamp(max(wipe, windex_clean), 0.0, 1.0);
    let base_alpha = dust * grain * 0.55 * drift * (1.0 - effective_wipe);

    let sheen_col = vec3(0.92, 0.95, 1.0);
    let added_alpha = windex_sheen * 0.7 + spray_alpha * 0.55;
    let color = mix(theme.fg_muted.rgb, sheen_col, clamp(added_alpha, 0.0, 1.0));
    let alpha = clamp(base_alpha + added_alpha, 0.0, 1.0);
    return vec4(color, alpha);
}
