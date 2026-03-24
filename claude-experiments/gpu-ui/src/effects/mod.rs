mod wobbly;
mod cube_transition;

pub use wobbly::*;
pub use cube_transition::*;

#[derive(Clone, Debug)]
pub enum Effect {
    Wobbly(WobblyParams),
    CubeTransition(CubeTransitionParams),
}

pub fn process_effects(
    scene: &mut crate::scene::Scene,
    wobbly_mgr: &mut WobblyManager,
    dt: f32,
) {
    process_node_effects(&mut scene.root, wobbly_mgr, dt);
}

fn process_node_effects(
    node: &mut crate::scene::SceneNode,
    wobbly_mgr: &mut WobblyManager,
    dt: f32,
) {
    let effects = node.effects.clone();
    for effect in &effects {
        match effect {
            Effect::Wobbly(params) => {
                if let Some(ref mut surface) = node.surface {
                    let sim = wobbly_mgr.ensure(
                        surface.id,
                        surface.size.x,
                        surface.size.y,
                        params.grid_size,
                    );
                    sim.step(params, dt);

                    // Write Bezier-evaluated offsets into vertex positions.
                    // The sim works in local 2D (x=right, y=down).
                    // The mesh works in local 3D (x=right, y=UP, z=toward camera).
                    // So: sim_x → mesh_x, sim_y → -mesh_y
                    let res = surface.mesh_resolution;
                    let vps = res + 1;
                    let expected = (vps * vps) as usize;
                    if surface.vertex_offsets.len() != expected {
                        surface.vertex_offsets = vec![glam::Vec3::ZERO; expected];
                    }
                    for vy in 0..vps {
                        for vx in 0..vps {
                            let u = vx as f32 / res as f32;
                            let v = vy as f32 / res as f32;
                            let off = sim.deform_offset(u, v);
                            let idx = (vy * vps + vx) as usize;
                            surface.vertex_offsets[idx] =
                                glam::Vec3::new(off.x, -off.y, 0.0);
                        }
                    }
                }
            }
            Effect::CubeTransition(params) => {
                cube_transition::apply_cube_transition(node, params, 0.0);
            }
        }
    }
    for child in &mut node.children {
        process_node_effects(child, wobbly_mgr, dt);
    }
}
