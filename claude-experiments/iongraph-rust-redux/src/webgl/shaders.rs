//! GLSL shader programs for WebGL 2 rendering
//!
//! Three shader programs:
//! 1. Shape shader - solid/stroked rectangles and triangles
//! 2. Path shader - tessellated stroke paths (arrows)
//! 3. Text shader - textured quads for text rendering

/// Vertex shader for shapes (rectangles, triangles)
///
/// Attributes:
/// - a_position: vec2 - vertex position in world coordinates
/// - a_color: vec4 - RGBA color
///
/// Uniforms:
/// - u_transform: mat4 - view-projection matrix from Viewport
pub const SHAPE_VERTEX: &str = r#"#version 300 es
precision highp float;

layout(location = 0) in vec2 a_position;
layout(location = 1) in vec4 a_color;

uniform mat4 u_transform;

out vec4 v_color;

void main() {
    gl_Position = u_transform * vec4(a_position, 0.0, 1.0);
    v_color = a_color;
}
"#;

/// Fragment shader for shapes
pub const SHAPE_FRAGMENT: &str = r#"#version 300 es
precision highp float;

in vec4 v_color;
out vec4 fragColor;

void main() {
    fragColor = v_color;
}
"#;

/// Vertex shader for paths (tessellated strokes)
///
/// Same as shape shader - lyon tessellation handles line width
pub const PATH_VERTEX: &str = r#"#version 300 es
precision highp float;

layout(location = 0) in vec2 a_position;
layout(location = 1) in vec4 a_color;

uniform mat4 u_transform;

out vec4 v_color;

void main() {
    gl_Position = u_transform * vec4(a_position, 0.0, 1.0);
    v_color = a_color;
}
"#;

/// Fragment shader for paths
pub const PATH_FRAGMENT: &str = r#"#version 300 es
precision highp float;

in vec4 v_color;
out vec4 fragColor;

void main() {
    fragColor = v_color;
}
"#;

/// Vertex shader for text (textured quads)
///
/// Attributes:
/// - a_position: vec2 - quad vertex position
/// - a_texcoord: vec2 - UV coordinates in text atlas
/// - a_color: vec4 - text color
pub const TEXT_VERTEX: &str = r#"#version 300 es
precision highp float;

layout(location = 0) in vec2 a_position;
layout(location = 1) in vec2 a_texcoord;
layout(location = 2) in vec4 a_color;

uniform mat4 u_transform;

out vec2 v_texcoord;
out vec4 v_color;

void main() {
    gl_Position = u_transform * vec4(a_position, 0.0, 1.0);
    v_texcoord = a_texcoord;
    v_color = a_color;
}
"#;

/// Fragment shader for text
/// Uses red channel from texture as alpha (single-channel text atlas)
pub const TEXT_FRAGMENT: &str = r#"#version 300 es
precision highp float;

uniform sampler2D u_texture;

in vec2 v_texcoord;
in vec4 v_color;
out vec4 fragColor;

void main() {
    float alpha = texture(u_texture, v_texcoord).r;
    fragColor = vec4(v_color.rgb, v_color.a * alpha);
}
"#;

/// Vertex shader for rounded rectangles (using SDF)
/// This is an alternative approach for smooth rounded corners
pub const ROUNDED_RECT_VERTEX: &str = r#"#version 300 es
precision highp float;

layout(location = 0) in vec2 a_position;
layout(location = 1) in vec2 a_rect_pos;    // Rectangle top-left
layout(location = 2) in vec2 a_rect_size;   // Rectangle dimensions
layout(location = 3) in float a_radius;     // Corner radius
layout(location = 4) in vec4 a_color;

uniform mat4 u_transform;

out vec2 v_local_pos;
out vec2 v_rect_size;
out float v_radius;
out vec4 v_color;

void main() {
    // Transform position
    vec2 world_pos = a_rect_pos + a_position * a_rect_size;
    gl_Position = u_transform * vec4(world_pos, 0.0, 1.0);

    // Pass local coordinates for SDF calculation
    v_local_pos = a_position * a_rect_size;
    v_rect_size = a_rect_size;
    v_radius = a_radius;
    v_color = a_color;
}
"#;

/// Fragment shader for rounded rectangles using SDF
pub const ROUNDED_RECT_FRAGMENT: &str = r#"#version 300 es
precision highp float;

in vec2 v_local_pos;
in vec2 v_rect_size;
in float v_radius;
in vec4 v_color;
out vec4 fragColor;

float roundedRectSDF(vec2 p, vec2 size, float radius) {
    vec2 q = abs(p - size * 0.5) - size * 0.5 + radius;
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - radius;
}

void main() {
    float d = roundedRectSDF(v_local_pos, v_rect_size, v_radius);
    float aa = fwidth(d);
    float alpha = 1.0 - smoothstep(-aa, aa, d);
    fragColor = vec4(v_color.rgb, v_color.a * alpha);
}
"#;

/// Attribute locations for consistent binding
pub mod attributes {
    pub const POSITION: u32 = 0;
    pub const COLOR: u32 = 1;
    pub const TEXCOORD: u32 = 1; // Shares with color for text shader
    pub const TEXT_COLOR: u32 = 2;
}

/// Uniform names
pub mod uniforms {
    pub const TRANSFORM: &str = "u_transform";
    pub const TEXTURE: &str = "u_texture";
}
