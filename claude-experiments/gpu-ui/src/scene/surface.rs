use glam::{Vec2, Vec3};
use super::Material;

/// A unique identifier for a surface in the scene.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SurfaceId(pub u64);

/// A Surface is the fundamental drawable unit — a rectangular region in 3D space
/// that can be deformed, lit, shadowed, and composited.
///
/// Think of each "window" or UI panel as a Surface. The scene graph arranges them;
/// the effects system can deform them; the renderer draws them.
#[derive(Clone, Debug)]
pub struct Surface {
    pub id: SurfaceId,
    pub size: Vec2,
    pub material: Material,
    /// Mesh subdivision level — higher means more vertices for deformation effects
    /// like wobbly windows. 1 = simple quad, 16 = 16x16 grid, etc.
    pub mesh_resolution: u32,
    /// Per-vertex offsets applied by the effects system (wobbly, wave, etc.).
    /// Length = (mesh_resolution+1)^2 when populated.
    /// These are in local surface space.
    pub vertex_offsets: Vec<Vec3>,
}

impl Surface {
    pub fn new(id: SurfaceId, size: Vec2) -> Self {
        Self {
            id,
            size,
            material: Material::default(),
            mesh_resolution: 1,
            vertex_offsets: Vec::new(),
        }
    }

    /// Create a surface ready for deformation effects (wobbly windows etc.)
    pub fn deformable(id: SurfaceId, size: Vec2, resolution: u32) -> Self {
        let verts = (resolution + 1) * (resolution + 1);
        Self {
            id,
            size,
            material: Material::default(),
            mesh_resolution: resolution,
            vertex_offsets: vec![Vec3::ZERO; verts as usize],
        }
    }

    pub fn with_material(mut self, material: Material) -> Self {
        self.material = material;
        self
    }

    pub fn vertex_count(&self) -> u32 {
        (self.mesh_resolution + 1) * (self.mesh_resolution + 1)
    }

    pub fn index_count(&self) -> u32 {
        self.mesh_resolution * self.mesh_resolution * 6
    }
}
