#version 330

// Cone-light darkness overlay.
//   * Runs in screen space — we draw a full-screen quad in screen coords.
//   * `player_pos`, `aim_dir`, and extra lights are also in screen coords.
//   * Output is black with alpha = 1 - clamp(ambient + total_light, 0, 1),
//     so drawn on top of the scene it darkens pixels outside the cone.

#define MAX_EXTRA_LIGHTS 8

in vec2 fragTexCoord;
in vec4 fragColor;
out vec4 finalColor;

uniform vec2 player_pos;          // screen-space
uniform vec2 aim_dir;             // unit vector
uniform float cos_half_angle;     // cos of half-cone angle
uniform float range;              // pixels
uniform float ambient;
uniform float intensity;
uniform vec2 resolution;          // screen width/height in pixels

uniform int extra_count;
uniform vec2  extras_pos[MAX_EXTRA_LIGHTS];
uniform vec2  extras_dir[MAX_EXTRA_LIGHTS];
uniform float extras_cos[MAX_EXTRA_LIGHTS];
uniform float extras_range[MAX_EXTRA_LIGHTS];
uniform float extras_intensity[MAX_EXTRA_LIGHTS];

float cone_contribution(vec2 world_pos, vec2 light_pos, vec2 light_dir,
                        float cos_half, float r, float i_intensity) {
    vec2 to_pixel = world_pos - light_pos;
    float dist = length(to_pixel);
    if (dist >= r) { return 0.0; }
    vec2 dir = to_pixel / max(dist, 0.0001);
    vec2 aim = normalize(light_dir);
    float edge = cos_half;
    float angular = smoothstep(edge - 0.004, edge + 0.004, dot(dir, aim));
    float radial = 1.0 - smoothstep(r * 0.15, r, dist);
    return angular * radial * i_intensity;
}

void main() {
    // gl_FragCoord is in window pixels; convert to our y-down screen space.
    vec2 screen_pos = vec2(gl_FragCoord.x, resolution.y - gl_FragCoord.y);

    float total = cone_contribution(screen_pos, player_pos, aim_dir,
                                    cos_half_angle, range, intensity);

    for (int i = 0; i < extra_count && i < MAX_EXTRA_LIGHTS; i++) {
        total += cone_contribution(screen_pos, extras_pos[i], extras_dir[i],
                                   extras_cos[i], extras_range[i], extras_intensity[i]);
    }

    float light = clamp(ambient + total, 0.0, 1.0);
    float darkness_alpha = 1.0 - light;
    finalColor = vec4(0.0, 0.0, 0.0, darkness_alpha);
}
