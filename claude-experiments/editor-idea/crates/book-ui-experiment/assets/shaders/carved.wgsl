// Real ray-marched lighting on a layered paper-cutout surface, with
// up to three independent lights and a perspective tilt camera.
//
// At each fragment we sample the surface (height, color), compute
// per-light shadow rays, sum the diffuse contributions, and add a
// constant ambient term. In tilt mode the fragment is itself the
// result of a perspective ray-march into the height field.
//
// Coordinates: x right, y down (screen-space), z up (toward viewer).

@group(2) @binding(0) var<uniform> params: vec4<f32>;
// params.xy = logical resolution, .z = time, .w = device scale

// Each light is two vec4s: dir.xyz + intensity, color.rgb + reserved.
@group(2) @binding(1) var<uniform> light0_dir: vec4<f32>;
@group(2) @binding(2) var<uniform> light0_col: vec4<f32>;
@group(2) @binding(3) var<uniform> light1_dir: vec4<f32>;
@group(2) @binding(4) var<uniform> light1_col: vec4<f32>;
@group(2) @binding(5) var<uniform> light2_dir: vec4<f32>;
@group(2) @binding(6) var<uniform> light2_col: vec4<f32>;

// shading.x = ambient, .y = shadow blur radius (world px),
// .z = max shadow strength multiplier, .w = sample count (1, 4, 8, 16, 32)
@group(2) @binding(7) var<uniform> shading: vec4<f32>;

// camera.xy = pan, .z = zoom, .w = tilt (radians)
@group(2) @binding(8) var<uniform> camera: vec4<f32>;

fn rounded_rect_sdf(p: vec2<f32>, hh: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - hh + vec2<f32>(r);
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

struct HC {
    height: f32,
    color: vec3<f32>,
}

fn apply_panel(hc_in: HC, p: vec2<f32>, c: vec2<f32>, hh: vec2<f32>, r: f32, z: f32, col: vec3<f32>) -> HC {
    if (rounded_rect_sdf(p - c, hh, r) < 0.0) {
        var hc: HC;
        hc.height = z;
        hc.color = col;
        return hc;
    }
    return hc_in;
}

fn inside_circle(p: vec2<f32>, c: vec2<f32>, r: f32) -> bool {
    let d = p - c;
    return dot(d, d) < r * r;
}

fn circle_sdf(p: vec2<f32>, c: vec2<f32>, r: f32) -> f32 {
    return length(p - c) - r;
}

// Isoceles triangle pointing up — exact SDF (max of signed line
// distances for a convex polygon, mirrored via abs(x)).
fn triangle_up_sdf(p: vec2<f32>, c: vec2<f32>, half_w: f32, h: f32) -> f32 {
    let q = vec2<f32>(abs(p.x - c.x), p.y - c.y);
    // Right edge: apex (0, -h) → base-right (half_w, h).
    let e = vec2<f32>(half_w, 2.0 * h);
    let n = vec2<f32>(e.y, -e.x) / max(length(e), 1e-4);
    let d_edge = dot(q - vec2<f32>(0.0, -h), n);
    let d_base = q.y - h;
    return max(d_edge, d_base);
}

fn set_hc(z: f32, col: vec3<f32>) -> HC {
    var h: HC;
    h.height = z;
    h.color = col;
    return h;
}

// AA half-width in pixels. ~0.7 spans ~1.4 px which kills the worst
// staircase aliasing without softening the carved feel.
const AA: f32 = 0.7;

// Apply a panel by SDF. Color blends across a 2*AA pixel band so the
// edge anti-aliases; height stays piecewise-constant (only set when
// fully inside) so the shadow ray-march sees crisp walls.
fn apply_sdf(hc_in: HC, sd: f32, z: f32, col: vec3<f32>) -> HC {
    var hc = hc_in;
    let cov = 1.0 - smoothstep(-AA, AA, sd);
    hc.color = mix(hc_in.color, col, cov);
    if (sd < 0.0) {
        hc.height = z;
    }
    return hc;
}

fn apply_circle_aa(hc: HC, p: vec2<f32>, c: vec2<f32>, r: f32, z: f32, col: vec3<f32>) -> HC {
    return apply_sdf(hc, circle_sdf(p, c, r), z, col);
}

fn apply_triangle_aa(hc: HC, p: vec2<f32>, c: vec2<f32>, half_w: f32, h: f32, z: f32, col: vec3<f32>) -> HC {
    return apply_sdf(hc, triangle_up_sdf(p, c, half_w, h), z, col);
}

fn apply_panel_aa(hc: HC, p: vec2<f32>, c: vec2<f32>, hh: vec2<f32>, r: f32, z: f32, col: vec3<f32>) -> HC {
    return apply_sdf(hc, rounded_rect_sdf(p - c, hh, r), z, col);
}

// Winter night-scene built for testing the lighting:
//   - large outer cream sheet (top layer)
//   - circular hole reveals a deep navy sky
//   - moon + stars sit just above the sky
//   - puffy cloud column on the left
//   - three layered snow hills (back→front, each higher than the last)
//   - pine trees on top of the hills
fn sample_surface(p: vec2<f32>) -> HC {
    var hc: HC;
    hc.height = -200.0;
    hc.color = vec3<f32>(0.06, 0.06, 0.08);

    let center = vec2<f32>(params.x * 0.5, params.y * 0.5);
    let cream  = vec3<f32>(0.88, 0.92, 0.94);

    // Outer cream sheet — full screen, AA'd just in case it's ever
    // bordered.
    hc = apply_panel_aa(
        hc, p, center,
        vec2<f32>(params.x * 0.5 + 50.0, params.y * 0.5 + 50.0),
        0.0, 12.0, cream,
    );

    let sky_r     = 380.0;
    let sky_color = vec3<f32>(0.18, 0.22, 0.42);

    // Sky cut — circular hole in the cream paper.
    hc = apply_circle_aa(hc, p, center, sky_r, -65.0, sky_color);

    // Cheap interior gate: skip all the inside-sky panels for pixels
    // well outside the sky, but keep the AA band so the inner panels
    // can still anti-alias against the sky color near its edge.
    let in_sky_band = circle_sdf(p, center, sky_r) < AA + 1.0;

    if (in_sky_band) {
        // Stars (tiny puffs just above sky).
        let star = vec3<f32>(1.0, 1.0, 0.96);
        hc = apply_circle_aa(hc, p, vec2<f32>(550.0, 215.0), 2.5, -58.0, star);
        hc = apply_circle_aa(hc, p, vec2<f32>(620.0, 175.0), 2.0, -58.0, star);
        hc = apply_circle_aa(hc, p, vec2<f32>(680.0, 250.0), 3.5, -58.0, star);
        hc = apply_circle_aa(hc, p, vec2<f32>(720.0, 180.0), 2.0, -58.0, star);
        hc = apply_circle_aa(hc, p, vec2<f32>(780.0, 230.0), 2.5, -58.0, star);
        hc = apply_circle_aa(hc, p, vec2<f32>(830.0, 290.0), 3.0, -58.0, star);
        hc = apply_circle_aa(hc, p, vec2<f32>(950.0, 320.0), 2.0, -58.0, star);
        hc = apply_circle_aa(hc, p, vec2<f32>(880.0, 380.0), 2.5, -58.0, star);
        hc = apply_circle_aa(hc, p, vec2<f32>(750.0, 350.0), 2.0, -58.0, star);
        hc = apply_circle_aa(hc, p, vec2<f32>(630.0, 300.0), 2.5, -58.0, star);
        hc = apply_circle_aa(hc, p, vec2<f32>(580.0, 380.0), 2.0, -58.0, star);
        hc = apply_circle_aa(hc, p, vec2<f32>(680.0, 420.0), 2.5, -58.0, star);
        hc = apply_circle_aa(hc, p, vec2<f32>(810.0, 450.0), 2.0, -58.0, star);
        hc = apply_circle_aa(hc, p, vec2<f32>(700.0, 280.0), 1.6, -58.0, star);
        hc = apply_circle_aa(hc, p, vec2<f32>(640.0, 240.0), 1.6, -58.0, star);
        hc = apply_circle_aa(hc, p, vec2<f32>(770.0, 380.0), 1.8, -58.0, star);
        hc = apply_circle_aa(hc, p, vec2<f32>(940.0, 250.0), 1.8, -58.0, star);
        hc = apply_circle_aa(hc, p, vec2<f32>(870.0, 200.0), 2.0, -58.0, star);
        hc = apply_circle_aa(hc, p, vec2<f32>(990.0, 400.0), 2.5, -58.0, star);
        hc = apply_circle_aa(hc, p, vec2<f32>(580.0, 460.0), 1.8, -58.0, star);
        hc = apply_circle_aa(hc, p, vec2<f32>(900.0, 470.0), 2.2, -58.0, star);

        // Moon.
        hc = apply_circle_aa(hc, p, vec2<f32>(center.x + 200.0, center.y - 240.0), 38.0, -50.0, vec3<f32>(0.97, 0.97, 0.92));

        // Cloud column on the left — union of overlapping puffs.
        let cloud_color = vec3<f32>(0.78, 0.84, 0.92);
        hc = apply_circle_aa(hc, p, vec2<f32>(425.0, 220.0), 60.0, -44.0, cloud_color);
        hc = apply_circle_aa(hc, p, vec2<f32>(440.0, 290.0), 75.0, -44.0, cloud_color);
        hc = apply_circle_aa(hc, p, vec2<f32>(420.0, 360.0), 70.0, -44.0, cloud_color);
        hc = apply_circle_aa(hc, p, vec2<f32>(450.0, 440.0), 80.0, -44.0, cloud_color);
        hc = apply_circle_aa(hc, p, vec2<f32>(420.0, 520.0), 65.0, -44.0, cloud_color);
        hc = apply_circle_aa(hc, p, vec2<f32>(380.0, 380.0), 55.0, -44.0, cloud_color);

        // Snow hills (back to front).
        hc = apply_circle_aa(hc, p, vec2<f32>(900.0, 1100.0), 600.0, -32.0, vec3<f32>(0.82, 0.88, 0.92));
        hc = apply_circle_aa(hc, p, vec2<f32>(450.0, 1100.0), 580.0, -20.0, vec3<f32>(0.92, 0.95, 0.97));
        hc = apply_circle_aa(hc, p, vec2<f32>(700.0, 1450.0), 950.0, -8.0,  vec3<f32>(0.98, 0.99, 1.0));

        // Pine trees.
        let tree_a = vec3<f32>(0.20, 0.30, 0.50);
        let tree_b = vec3<f32>(0.30, 0.40, 0.60);
        hc = apply_triangle_aa(hc, p, vec2<f32>(500.0, 600.0), 22.0, 55.0, 0.0, tree_b);
        hc = apply_triangle_aa(hc, p, vec2<f32>(560.0, 640.0), 25.0, 58.0, 0.0, tree_a);
        hc = apply_triangle_aa(hc, p, vec2<f32>(620.0, 605.0), 24.0, 56.0, 0.0, tree_b);
        hc = apply_triangle_aa(hc, p, vec2<f32>(700.0, 660.0), 30.0, 70.0, 0.0, tree_a);
        hc = apply_triangle_aa(hc, p, vec2<f32>(780.0, 645.0), 25.0, 60.0, 0.0, tree_b);
        hc = apply_triangle_aa(hc, p, vec2<f32>(850.0, 670.0), 28.0, 65.0, 0.0, tree_a);
        hc = apply_triangle_aa(hc, p, vec2<f32>(910.0, 650.0), 24.0, 58.0, 0.0, tree_b);
        hc = apply_triangle_aa(hc, p, vec2<f32>(970.0, 660.0), 22.0, 52.0, 0.0, tree_a);
    }

    return hc;
}

fn surface_height(p: vec2<f32>) -> f32 {
    return sample_surface(p).height;
}

// Single binary shadow ray. Returns 0 (occluded) or 1 (visible).
fn hard_shadow(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    let n_steps = 22;
    let max_t = max((0.5 - ro.z) / max(rd.z, 0.001), 0.001);
    let dt = max_t / f32(n_steps);
    for (var i = 1; i <= n_steps; i = i + 1) {
        let t = f32(i) * dt;
        let pos = ro + rd * t;
        if (pos.z < surface_height(pos.xy)) {
            return 0.0;
        }
    }
    return 1.0;
}

// Cheap 2D hash for per-pixel pattern rotation.
fn hash21(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453);
}

// Disc-blurred shadow: average N hard shadow rays cast from positions
// jittered within a disc of radius `blur` around the pixel. Produces
// a uniform-width soft edge regardless of distance to the caster.
// Vogel disc gives even coverage at low N; per-pixel rotation breaks
// up the residual rings at the cost of a little high-frequency noise.
fn blurred_shadow(world_pos: vec3<f32>, light_dir: vec3<f32>, blur: f32, n: i32) -> f32 {
    if (n <= 1 || blur < 0.05) {
        let ro = world_pos + vec3<f32>(0.0, 0.0, 0.05);
        return hard_shadow(ro, light_dir);
    }
    let golden = 2.39996323;
    let phase = hash21(floor(world_pos.xy)) * 6.28318;
    var sum = 0.0;
    for (var i = 0; i < n; i = i + 1) {
        let f = (f32(i) + 0.5) / f32(n);
        let r = blur * sqrt(f);
        let angle = f32(i) * golden + phase;
        let offset = vec2<f32>(cos(angle), sin(angle)) * r;
        let ro = world_pos + vec3<f32>(offset.x, offset.y, 0.05);
        sum = sum + hard_shadow(ro, light_dir);
    }
    return sum / f32(n);
}

fn light_contribution(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    light_dir: vec3<f32>,
    light_intensity: f32,
    light_color: vec3<f32>,
    blur: f32,
    n_samples: i32,
    max_shadow: f32,
) -> vec3<f32> {
    if (light_intensity < 0.001) {
        return vec3<f32>(0.0);
    }
    let l = normalize(light_dir);
    let d = max(dot(normal, l), 0.0);
    if (d <= 0.0) {
        return vec3<f32>(0.0);
    }
    let raw_shadow = blurred_shadow(world_pos, l, blur, n_samples);
    let shadow = 1.0 - max_shadow * (1.0 - raw_shadow);
    return light_color * (d * light_intensity * shadow);
}

// Combined shading: ambient + sum of per-light contributions, applied
// to the surface base color.
fn shade(world_pos: vec3<f32>, base_color: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let ambient = shading.x;
    let blur = shading.y;
    let max_shadow = shading.z;
    let n_samples = i32(shading.w + 0.5);

    var lit = vec3<f32>(ambient);
    lit = lit + light_contribution(world_pos, normal, light0_dir.xyz, light0_dir.w, light0_col.rgb, blur, n_samples, max_shadow);
    lit = lit + light_contribution(world_pos, normal, light1_dir.xyz, light1_dir.w, light1_col.rgb, blur, n_samples, max_shadow);
    lit = lit + light_contribution(world_pos, normal, light2_dir.xyz, light2_dir.w, light2_col.rgb, blur, n_samples, max_shadow);
    return base_color * lit;
}

// Perspective tilt-mode ray-march.
fn render_tilted(p_screen: vec2<f32>) -> vec3<f32> {
    let center = vec2<f32>(params.x, params.y) * 0.5;
    let zoom = max(camera.z, 0.001);
    let pan = camera.xy;
    let tilt = camera.w;

    let s = sin(tilt);
    let c = cos(tilt);

    let scene_center = vec3<f32>(center.x + pan.x, center.y + pan.y, 0.0);

    // Camera basis (X-axis pitch). At t=0 this is straight down.
    let right   = vec3<f32>(1.0, 0.0, 0.0);
    let up      = vec3<f32>(0.0, c, s);
    let forward = vec3<f32>(0.0, s, -c);

    let cam_dist = 1200.0;
    let cam_pos = scene_center - forward * cam_dist;

    // Pixel offsets normalized by half-screen-height.
    let px = (p_screen.x - center.x) / center.y;
    let py = (p_screen.y - center.y) / center.y;

    // Image-plane distance in normalized units. Calibrated so that
    // at zoom=1 the visible world-y span at z=0 is roughly the screen
    // height, matching the ortho top-down view.
    let d = (cam_dist / center.y) / zoom;
    let cs_dir = vec3<f32>(px, py, -d);
    let cs_n = normalize(cs_dir);
    let rd = right * cs_n.x + up * cs_n.y + forward * (-cs_n.z);
    let ro = cam_pos;

    let n_steps = 192;
    let max_t = 4000.0;
    let dt = max_t / f32(n_steps);

    var hit_t = -1.0;
    for (var i = 1; i <= n_steps; i = i + 1) {
        let t = f32(i) * dt;
        let pos = ro + rd * t;
        let h = surface_height(pos.xy);
        if (pos.z < h) {
            var lo = f32(i - 1) * dt;
            var hi = t;
            for (var j = 0; j < 7; j = j + 1) {
                let mid = (lo + hi) * 0.5;
                let m_pos = ro + rd * mid;
                if (m_pos.z < surface_height(m_pos.xy)) { hi = mid; } else { lo = mid; }
            }
            hit_t = hi;
            break;
        }
    }

    if (hit_t < 0.0) {
        return vec3<f32>(0.05, 0.04, 0.07);
    }

    let hit_pos = ro + rd * hit_t;
    let surf_below = sample_surface(hit_pos.xy);
    let above_surface = hit_pos.z - surf_below.height;

    if (above_surface > 0.6) {
        // Wall hit. Normal from gradient of height field.
        let eps = 0.7;
        let hxp = surface_height(hit_pos.xy + vec2<f32>(eps, 0.0));
        let hxm = surface_height(hit_pos.xy - vec2<f32>(eps, 0.0));
        let hyp = surface_height(hit_pos.xy + vec2<f32>(0.0, eps));
        let hym = surface_height(hit_pos.xy - vec2<f32>(0.0, eps));
        let g = vec2<f32>(hxp - hxm, hyp - hym);
        let gm = max(length(g), 0.001);
        let gnorm = g / gm;
        let normal = normalize(vec3<f32>(-gnorm.x, -gnorm.y, 0.18));
        let into = hit_pos.xy + gnorm * 1.5;
        let wall_color = sample_surface(into).color;
        return shade(hit_pos, wall_color * 0.82, normal);
    }

    let normal = vec3<f32>(0.0, 0.0, 1.0);
    return shade(hit_pos, surf_below.color, normal);
}

@fragment
fn fragment(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let dpi = max(params.w, 1.0);
    let p_screen = frag.xy / dpi;

    let tilt = camera.w;
    if (tilt > 0.01) {
        return vec4<f32>(render_tilted(p_screen), 1.0);
    }

    // Top-down (orthographic) path.
    let center = vec2<f32>(params.x, params.y) * 0.5;
    let zoom = max(camera.z, 0.001);
    let p = (p_screen - center) / zoom + center + camera.xy;

    let hc = sample_surface(p);
    let world_pos = vec3<f32>(p.x, p.y, hc.height);
    let normal = vec3<f32>(0.0, 0.0, 1.0);
    return vec4<f32>(shade(world_pos, hc.color, normal), 1.0);
}
