//! WGSL introspection: parse a user shader and learn its data shape.
//!
//! The generic dynamic-shader architecture asks: instead of typed Rust
//! `WorldUniforms`/`ProjectUniforms`/etc. structs whose layouts must
//! match the WGSL by hand, let the **shader** declare its uniform
//! struct and texture bindings; the host parses the WGSL and learns
//! what names exist, where to write each one in the uniform buffer,
//! and which binding slot each texture lives at.
//!
//! ## What we learn from a shader
//!
//! Given a WGSL like:
//!
//! ```wgsl
//! struct UserData {
//!     time: f32,
//!     dust_seconds: f32,
//!     mouse_world: vec2<f32>,
//!     focused_pane: vec4<f32>,
//! }
//! @group(0) @binding(0) var<uniform> user: UserData;
//! @group(0) @binding(1) var wipe_mask: texture_2d<f32>;
//! @group(0) @binding(2) var tex_sampler: sampler;
//! @group(0) @binding(3) var noise: texture_2d<f32>;
//! ```
//!
//! [`Schema::from_wgsl`] returns a `Schema` with:
//! - `uniform_block_size = 32` bytes
//! - `fields = { "time": (offset=0, F32), "dust_seconds": (offset=4, F32),
//!              "mouse_world": (offset=8, Vec2), "focused_pane": (offset=16, Vec4) }`
//! - `textures = { "wipe_mask": binding=1, "noise": binding=3 }`
//! - `samplers = { "tex_sampler": binding=2 }`
//!
//! Naga gives us `StructMember.offset` directly (it does std140 layout
//! during WGSL parse), so we don't have to compute alignment ourselves.

use std::collections::HashMap;

use naga::{TypeInner, VectorSize};

/// What kind of primitive a uniform field is. Drives how many bytes
/// `uniform_set` writes at the field's offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldKind {
    F32,
    Vec2,
    Vec3,
    Vec4,
}

impl FieldKind {
    /// Byte size of the value. (Vec3 occupies 12 bytes for the data
    /// itself; std140 padding to 16 is the layout's concern, not the
    /// write's.)
    pub fn size(self) -> usize {
        match self {
            FieldKind::F32 => 4,
            FieldKind::Vec2 => 8,
            FieldKind::Vec3 => 12,
            FieldKind::Vec4 => 16,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FieldLayout {
    pub offset: usize,
    pub kind: FieldKind,
}

#[derive(Default, Debug, Clone)]
pub struct Schema {
    /// Total bytes the uniform block occupies. The host's uniform
    /// buffer must be at least this large.
    pub uniform_block_size: usize,
    /// Field name → (offset, kind). Names that don't appear here are
    /// silently ignored by `uniform_set` (with a logged warning) so
    /// scripts can be loosely-coupled to shaders during iteration.
    pub fields: HashMap<String, FieldLayout>,
    /// Texture binding-slot name → @binding(N) index.
    pub textures: HashMap<String, u32>,
    /// Sampler binding-slot name → @binding(N) index.
    pub samplers: HashMap<String, u32>,
}

#[derive(Debug)]
pub enum SchemaError {
    Parse(String),
    NoUserStruct,
    UnsupportedField {
        struct_name: String,
        field_name: String,
        reason: String,
    },
}

impl std::fmt::Display for SchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "wgsl parse: {}", e),
            Self::NoUserStruct => write!(
                f,
                "shader has no `var<uniform> user: UserData;` declaration"
            ),
            Self::UnsupportedField { struct_name, field_name, reason } => write!(
                f,
                "{}::{}: {}",
                struct_name, field_name, reason
            ),
        }
    }
}

impl Schema {
    /// Parse a WGSL source and extract its uniform/texture schema.
    /// Looks for:
    /// - A global `var<uniform> user: SomeStruct;` declaration — that
    ///   struct's fields become the uniform schema.
    /// - All `texture_2d<f32>` and `sampler` globals — recorded by
    ///   their @binding number.
    ///
    /// Bevy preprocessor directives (`#import ...`, `#define_...`,
    /// `#ifdef ...`) are stripped before parsing, and `#{NAME}`
    /// placeholders are substituted with a stable index (Material2d
    /// always lives at @group(2), so `MATERIAL_BIND_GROUP` → 2).
    /// We only need the bindings + struct layout — *not* the imported
    /// symbols — so naga doesn't have to understand the imports.
    pub fn from_wgsl(source: &str) -> Result<Self, SchemaError> {
        let cleaned = preprocess_for_naga(source);
        let module = naga::front::wgsl::parse_str(&cleaned)
            .map_err(|e| SchemaError::Parse(format!("{:#}", e)))?;

        let mut schema = Schema::default();

        // 1) Find the `user` uniform global and walk its struct.
        let user_global = module
            .global_variables
            .iter()
            .find(|(_, gv)| {
                gv.name.as_deref() == Some("user")
                    && matches!(gv.space, naga::AddressSpace::Uniform)
            })
            .ok_or(SchemaError::NoUserStruct)?;
        let user_type = &module.types[user_global.1.ty];
        let TypeInner::Struct { members, span } = &user_type.inner else {
            return Err(SchemaError::UnsupportedField {
                struct_name: "user".into(),
                field_name: "<root>".into(),
                reason: "expected a struct type".into(),
            });
        };
        schema.uniform_block_size = *span as usize;

        for m in members {
            let Some(name) = m.name.clone() else { continue };
            let member_type = &module.types[m.ty];
            let kind = match &member_type.inner {
                TypeInner::Scalar(s) if s.kind == naga::ScalarKind::Float && s.width == 4 => {
                    FieldKind::F32
                }
                TypeInner::Vector { size: VectorSize::Bi, scalar }
                    if scalar.kind == naga::ScalarKind::Float && scalar.width == 4 =>
                {
                    FieldKind::Vec2
                }
                TypeInner::Vector { size: VectorSize::Tri, scalar }
                    if scalar.kind == naga::ScalarKind::Float && scalar.width == 4 =>
                {
                    FieldKind::Vec3
                }
                TypeInner::Vector { size: VectorSize::Quad, scalar }
                    if scalar.kind == naga::ScalarKind::Float && scalar.width == 4 =>
                {
                    FieldKind::Vec4
                }
                other => {
                    // For now, silently skip unsupported kinds (u32,
                    // matrices, arrays, etc.) — they won't show up in
                    // the field map, so `uniform_set` for them is a
                    // no-op with a warning. Adding support later means
                    // extending FieldKind + the byte-packer.
                    eprintln!(
                        "[introspect] skipping field `user.{}`: unsupported kind {:?}",
                        name, other
                    );
                    continue;
                }
            };
            schema.fields.insert(
                name,
                FieldLayout {
                    offset: m.offset as usize,
                    kind,
                },
            );
        }

        // 2) Walk the remaining globals to collect texture + sampler
        //    bindings. Image class doesn't matter for our purposes
        //    yet — anything declared as `texture_2d` gets a slot in
        //    the named-texture map.
        for (_, gv) in module.global_variables.iter() {
            let Some(name) = gv.name.clone() else { continue };
            if name == "user" {
                continue;
            }
            let Some(binding) = &gv.binding else { continue };
            let ty = &module.types[gv.ty];
            match &ty.inner {
                TypeInner::Image { .. } => {
                    schema.textures.insert(name, binding.binding);
                }
                TypeInner::Sampler { .. } => {
                    schema.samplers.insert(name, binding.binding);
                }
                _ => {}
            }
        }

        Ok(schema)
    }

    /// Pack a scalar into the uniform buffer at the given field's
    /// offset. Silently warns + drops the write if the field doesn't
    /// exist OR the kind doesn't match — this is the "loosely coupled
    /// script ↔ shader" contract: spell the name right on both sides.
    pub fn write_f32(&self, buffer: &mut [u8], name: &str, value: f32) {
        let Some(field) = self.fields.get(name) else {
            // Don't spam-warn every frame; the caller can decide whether
            // to track these. Returning silently is the convention.
            return;
        };
        if field.kind != FieldKind::F32 {
            return;
        }
        let bytes = value.to_le_bytes();
        buffer[field.offset..field.offset + 4].copy_from_slice(&bytes);
    }

    pub fn write_vec2(&self, buffer: &mut [u8], name: &str, value: [f32; 2]) {
        let Some(field) = self.fields.get(name) else { return };
        if field.kind != FieldKind::Vec2 {
            return;
        }
        for (i, v) in value.iter().enumerate() {
            let o = field.offset + i * 4;
            buffer[o..o + 4].copy_from_slice(&v.to_le_bytes());
        }
    }

    pub fn write_vec4(&self, buffer: &mut [u8], name: &str, value: [f32; 4]) {
        let Some(field) = self.fields.get(name) else { return };
        if field.kind != FieldKind::Vec4 {
            return;
        }
        for (i, v) in value.iter().enumerate() {
            let o = field.offset + i * 4;
            buffer[o..o + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
}

/// Strip Bevy-specific preprocessor lines + substitute `#{NAME}` so
/// naga can parse the shader for introspection. Does NOT preserve
/// the imported symbols (e.g. `VertexOutput` from
/// `bevy_sprite::mesh2d_vertex_output`) — but those don't appear in
/// the struct layout we care about, so naga just sees them as free
/// type references which we discard.
fn preprocess_for_naga(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    for line in source.lines() {
        let trimmed = line.trim_start();
        // Drop any line starting with `#` (#import, #define_*, #ifdef,
        // #else, #endif, ...). The actual shader compile still sees
        // them via Bevy's naga_oil preprocessor; we only need a
        // best-effort parse for introspection.
        if trimmed.starts_with('#') {
            continue;
        }
        // Substitute `#{IDENT}` placeholders inline. Material2d's
        // bind group is always 2 in practice. Anything else we don't
        // know → replace with `0` so the parse succeeds (the value
        // doesn't affect the schema we extract).
        let mut s = String::with_capacity(line.len());
        let mut rest = line;
        while let Some(start) = rest.find("#{") {
            s.push_str(&rest[..start]);
            let after = &rest[start + 2..];
            if let Some(end) = after.find('}') {
                let name = &after[..end];
                let val = match name {
                    "MATERIAL_BIND_GROUP" => "2",
                    _ => "0",
                };
                s.push_str(val);
                rest = &after[end + 1..];
            } else {
                // Unterminated — give up, emit the rest verbatim.
                s.push_str(&rest[start..]);
                rest = "";
                break;
            }
        }
        s.push_str(rest);
        out.push_str(&s);
        out.push('\n');
    }
    // Naga complains if the shader references types it can't resolve
    // (e.g. `VertexOutput` from the import we stripped). Provide a
    // throwaway definition so any function signature referencing it
    // still parses. This is harmless — we only walk the uniform/
    // texture globals, not the function bodies.
    out.push_str(
        r#"
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
};
"#,
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_schema() {
        let src = r#"
            struct UserData {
                time: f32,
                dust_seconds: f32,
                mouse_world: vec2<f32>,
                focused_pane: vec4<f32>,
            }
            @group(0) @binding(0) var<uniform> user: UserData;
            @group(0) @binding(1) var wipe_mask: texture_2d<f32>;
            @group(0) @binding(2) var tex_sampler: sampler;
        "#;
        let s = Schema::from_wgsl(src).expect("parse");
        assert_eq!(s.fields.get("time").unwrap().offset, 0);
        assert_eq!(s.fields.get("dust_seconds").unwrap().offset, 4);
        assert_eq!(s.fields.get("mouse_world").unwrap().offset, 8);
        assert_eq!(s.fields.get("focused_pane").unwrap().offset, 16);
        assert_eq!(s.textures.get("wipe_mask"), Some(&1));
        assert_eq!(s.samplers.get("tex_sampler"), Some(&2));
    }

    #[test]
    fn writes_into_buffer_at_offset() {
        let src = r#"
            struct UserData {
                a: f32,
                b: vec4<f32>,
            }
            @group(0) @binding(0) var<uniform> user: UserData;
        "#;
        let s = Schema::from_wgsl(src).unwrap();
        let mut buf = vec![0u8; s.uniform_block_size];
        s.write_f32(&mut buf, "a", 3.5);
        s.write_vec4(&mut buf, "b", [1.0, 2.0, 3.0, 4.0]);
        let a = f32::from_le_bytes(buf[0..4].try_into().unwrap());
        assert_eq!(a, 3.5);
        // b is at offset 16 (vec4 alignment).
        let b0 = f32::from_le_bytes(buf[16..20].try_into().unwrap());
        let b3 = f32::from_le_bytes(buf[28..32].try_into().unwrap());
        assert_eq!(b0, 1.0);
        assert_eq!(b3, 4.0);
    }
}
