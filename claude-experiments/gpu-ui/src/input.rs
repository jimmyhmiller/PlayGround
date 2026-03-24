use glam::{Mat4, Vec2, Vec3, Vec4};
use crate::scene::{Scene, SurfaceId};

#[derive(Clone, Debug)]
pub struct HitResult {
    pub surface_id: SurfaceId,
    pub node_index: usize,
    pub uv: Vec2,
    pub world_point: Vec3,
    pub distance: f32,
}

pub fn hit_test(
    scene: &Scene,
    screen_pos: Vec2,
    screen_size: Vec2,
    view_proj: Mat4,
) -> Option<HitResult> {
    let ndc_x = (screen_pos.x / screen_size.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (screen_pos.y / screen_size.y) * 2.0;

    let inv_vp = view_proj.inverse();

    let near_ndc = Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
    let far_ndc = Vec4::new(ndc_x, ndc_y, 1.0, 1.0);

    let near_world = inv_vp * near_ndc;
    let far_world = inv_vp * far_ndc;

    let near = near_world.truncate() / near_world.w;
    let far = far_world.truncate() / far_world.w;

    let ray_origin = near;
    let ray_dir = (far - near).normalize();

    let mut best: Option<HitResult> = None;

    for (node_index, child) in scene.root.children.iter().enumerate() {
        if !child.visible {
            continue;
        }
        if let Some(ref surface) = child.surface {
            // Compute the average vertex offset to know where
            // the surface visually appears (may differ from transform
            // due to wobbly deformation during/after drag).
            let avg_offset = if surface.vertex_offsets.is_empty() {
                Vec3::ZERO
            } else {
                let sum: Vec3 = surface.vertex_offsets.iter().copied().sum();
                sum / surface.vertex_offsets.len() as f32
            };

            let world = scene.root.transform.to_matrix() * child.transform.to_matrix();
            let inv_world = world.inverse();

            let local_origin_h = inv_world * Vec4::new(ray_origin.x, ray_origin.y, ray_origin.z, 1.0);
            let local_dir_h = inv_world * Vec4::new(ray_dir.x, ray_dir.y, ray_dir.z, 0.0);
            let local_origin = local_origin_h.truncate() / local_origin_h.w;
            let local_dir = local_dir_h.truncate().normalize();

            if local_dir.z.abs() < 1e-6 {
                continue;
            }

            let t = -local_origin.z / local_dir.z;
            if t < 0.0 {
                continue;
            }

            // Shift the hit point by the negative of the average offset
            // so we're testing against where the surface visually IS,
            // not where the transform says it should be.
            let local_hit = local_origin + local_dir * t - avg_offset;
            let half_w = surface.size.x / 2.0;
            let half_h = surface.size.y / 2.0;

            if local_hit.x >= -half_w
                && local_hit.x <= half_w
                && local_hit.y >= -half_h
                && local_hit.y <= half_h
            {
                let u = (local_hit.x + half_w) / surface.size.x;
                let v = (-local_hit.y + half_h) / surface.size.y;

                let world_hit_h = world * Vec4::new(
                    local_hit.x + avg_offset.x,
                    local_hit.y + avg_offset.y,
                    local_hit.z + avg_offset.z,
                    1.0,
                );
                let world_hit = world_hit_h.truncate() / world_hit_h.w;
                let distance = (world_hit - ray_origin).length();

                let is_closer = best.as_ref().map_or(true, |b| distance < b.distance);
                if is_closer {
                    best = Some(HitResult {
                        surface_id: surface.id,
                        node_index,
                        uv: Vec2::new(u, v),
                        world_point: world_hit,
                        distance,
                    });
                }
            }
        }
    }

    best
}
