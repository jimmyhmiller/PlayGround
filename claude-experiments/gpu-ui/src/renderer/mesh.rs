use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};
use crate::scene::Surface;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

impl Vertex {
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // normal
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // uv
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

/// Generate a subdivided quad mesh for a surface, applying vertex offsets if present.
pub fn generate_surface_mesh(surface: &Surface) -> (Vec<Vertex>, Vec<u32>) {
    let res = surface.mesh_resolution;
    let verts_per_side = res + 1;
    let half_w = surface.size.x / 2.0;
    let half_h = surface.size.y / 2.0;

    let mut vertices = Vec::with_capacity((verts_per_side * verts_per_side) as usize);
    let mut indices = Vec::with_capacity((res * res * 6) as usize);

    for y in 0..verts_per_side {
        for x in 0..verts_per_side {
            let u = x as f32 / res as f32;
            let v = y as f32 / res as f32;
            let idx = (y * verts_per_side + x) as usize;

            let base_pos = Vec3::new(
                -half_w + u * surface.size.x,
                half_h - v * surface.size.y,
                0.0,
            );

            let offset = if idx < surface.vertex_offsets.len() {
                surface.vertex_offsets[idx]
            } else {
                Vec3::ZERO
            };

            let pos = base_pos + offset;

            vertices.push(Vertex {
                position: pos.into(),
                normal: [0.0, 0.0, 1.0], // Will be recalculated below
                uv: [u, v],
            });
        }
    }

    // Generate indices
    for y in 0..res {
        for x in 0..res {
            let tl = y * verts_per_side + x;
            let tr = tl + 1;
            let bl = tl + verts_per_side;
            let br = bl + 1;

            indices.push(tl);
            indices.push(bl);
            indices.push(tr);

            indices.push(tr);
            indices.push(bl);
            indices.push(br);
        }
    }

    // Recalculate normals from actual geometry (important for deformed meshes)
    recalculate_normals(&mut vertices, &indices);

    (vertices, indices)
}

fn recalculate_normals(vertices: &mut [Vertex], indices: &[u32]) {
    // Zero all normals
    for v in vertices.iter_mut() {
        v.normal = [0.0, 0.0, 0.0];
    }

    // Accumulate face normals
    for tri in indices.chunks(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let p0 = Vec3::from(vertices[i0].position);
        let p1 = Vec3::from(vertices[i1].position);
        let p2 = Vec3::from(vertices[i2].position);

        let normal = (p1 - p0).cross(p2 - p0);
        for &i in &[i0, i1, i2] {
            vertices[i].normal[0] += normal.x;
            vertices[i].normal[1] += normal.y;
            vertices[i].normal[2] += normal.z;
        }
    }

    // Normalize
    for v in vertices.iter_mut() {
        let n = Vec3::from(v.normal).normalize_or_zero();
        v.normal = n.into();
    }
}
