use glam::Vec4;

/// Describes how a surface interacts with light.
/// This is pure data — the renderer decides how to realize it on the GPU.
#[derive(Clone, Debug)]
pub struct Material {
    pub base_color: Vec4, // RGBA
    pub roughness: f32,
    pub metallic: f32,
    pub opacity: f32,
    /// If true, this surface receives and casts shadows.
    pub shadow: bool,
    /// If true, the surface shows reflections of neighboring surfaces.
    pub reflective: bool,
    /// Emissive glow color+intensity (RGB + strength).
    pub emissive: Vec4,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color: Vec4::new(0.2, 0.2, 0.25, 1.0),
            roughness: 0.5,
            metallic: 0.0,
            opacity: 1.0,
            shadow: true,
            reflective: false,
            emissive: Vec4::ZERO,
        }
    }
}

impl Material {
    pub fn glass() -> Self {
        Self {
            base_color: Vec4::new(0.9, 0.95, 1.0, 0.3),
            roughness: 0.05,
            metallic: 0.0,
            opacity: 0.3,
            shadow: false,
            reflective: true,
            emissive: Vec4::ZERO,
        }
    }

    pub fn glowing(color: Vec4, strength: f32) -> Self {
        Self {
            base_color: color,
            emissive: Vec4::new(color.x, color.y, color.z, strength),
            shadow: false,
            ..Default::default()
        }
    }
}
