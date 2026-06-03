// Procedural skybox for the project-prism overview. Rendered on a large
// inside-out sphere; `world_position` gives the view direction. Driven by
// `params`: x = mode (0 dusk, 1 nebula, 2 space, 3 aurora), y = time.

#import bevy_pbr::forward_io::VertexOutput

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> params: vec4<f32>;

fn hash13(p3in: vec3<f32>) -> f32 {
    var p3 = fract(p3in * 0.1031);
    p3 = p3 + dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}

fn vnoise(x: vec3<f32>) -> f32 {
    let i = floor(x);
    let f = fract(x);
    let u = f * f * (3.0 - 2.0 * f);
    let n000 = hash13(i + vec3<f32>(0.0, 0.0, 0.0));
    let n100 = hash13(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash13(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash13(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash13(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash13(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash13(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash13(i + vec3<f32>(1.0, 1.0, 1.0));
    let nx00 = mix(n000, n100, u.x);
    let nx10 = mix(n010, n110, u.x);
    let nx01 = mix(n001, n101, u.x);
    let nx11 = mix(n011, n111, u.x);
    let nxy0 = mix(nx00, nx10, u.y);
    let nxy1 = mix(nx01, nx11, u.y);
    return mix(nxy0, nxy1, u.z);
}

fn fbm(p: vec3<f32>) -> f32 {
    var v = 0.0;
    var a = 0.5;
    var q = p;
    for (var i = 0; i < 4; i = i + 1) {
        v = v + a * vnoise(q);
        q = q * 2.02;
        a = a * 0.5;
    }
    return v;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let dir = normalize(in.world_position.xyz);
    let mode = params.x;
    let t = params.y;
    let h = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);

    var horizon = vec3<f32>(0.95, 0.42, 0.16);
    var zenith = vec3<f32>(0.05, 0.07, 0.20);
    var neb_col = vec3<f32>(0.95, 0.55, 0.25);
    var neb_amt = 0.18;
    var star_amt = 0.6;
    if (mode >= 0.5 && mode < 1.5) {
        // nebula
        horizon = vec3<f32>(0.20, 0.05, 0.30);
        zenith = vec3<f32>(0.02, 0.02, 0.09);
        neb_col = vec3<f32>(0.65, 0.20, 0.85);
        neb_amt = 0.6;
        star_amt = 0.9;
    } else if (mode >= 1.5 && mode < 2.5) {
        // space
        horizon = vec3<f32>(0.02, 0.03, 0.07);
        zenith = vec3<f32>(0.004, 0.004, 0.018);
        neb_col = vec3<f32>(0.15, 0.25, 0.55);
        neb_amt = 0.12;
        star_amt = 1.2;
    } else if (mode >= 2.5) {
        // aurora
        horizon = vec3<f32>(0.04, 0.28, 0.22);
        zenith = vec3<f32>(0.01, 0.04, 0.11);
        neb_col = vec3<f32>(0.10, 0.85, 0.55);
        neb_amt = 0.5;
        star_amt = 0.7;
    }

    // base vertical gradient
    let grad = smoothstep(0.0, 1.0, h);
    var col = mix(horizon, zenith, grad);

    // drifting nebula clouds
    let nb = fbm(dir * 3.0 + vec3<f32>(t * 0.012, 0.0, 0.0));
    let nb2 = fbm(dir * 6.5 - vec3<f32>(0.0, t * 0.009, 0.0));
    let cloud = pow(clamp(nb * nb2 * 2.2, 0.0, 1.0), 2.0);
    col = col + neb_col * cloud * neb_amt;

    // twinkling stars (denser higher up)
    let sd = dir * 260.0;
    let cell = floor(sd);
    let sh = hash13(cell);
    if (sh > 0.992) {
        let f = fract(sd) - 0.5;
        let d = length(f);
        let tw = 0.35 + 0.65 * sin(t * 3.0 + sh * 60.0);
        let star = smoothstep(0.10, 0.0, d) * max(tw, 0.0);
        col = col + vec3<f32>(star) * star_amt * (0.35 + 0.65 * h);
    }

    return vec4<f32>(col, 1.0);
}
