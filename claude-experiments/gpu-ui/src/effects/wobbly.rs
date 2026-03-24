use glam::Vec2;
use std::collections::HashMap;
use crate::scene::SurfaceId;

#[derive(Clone, Debug)]
pub struct WobblyParams {
    pub stiffness: f32,
    pub damping: f32,
    /// Grid points per side (e.g. 4 = 4x4 = 16 points, 8 = 8x8 = 64 points)
    pub grid_size: usize,
}

impl Default for WobblyParams {
    fn default() -> Self {
        Self {
            stiffness: 200.0,
            damping: 8.0,
            grid_size: 6,
        }
    }
}

#[derive(Clone, Debug)]
pub struct WobblySimulation {
    rest: Vec<Vec2>,
    pos: Vec<Vec2>,
    vel: Vec<Vec2>,
    grabbed: i32,
    grab_target: Vec2,
    pub active: bool,
    width: f32,
    height: f32,
    grid_w: usize,
    grid_h: usize,
}

impl WobblySimulation {
    pub fn new(width: f32, height: f32, grid_size: usize) -> Self {
        let gw = grid_size;
        let gh = grid_size;
        let count = gw * gh;
        let mut rest = vec![Vec2::ZERO; count];
        let mut pos = vec![Vec2::ZERO; count];
        for j in 0..gh {
            for i in 0..gw {
                let idx = j * gw + i;
                let p = Vec2::new(
                    i as f32 / (gw - 1) as f32 * width,
                    j as f32 / (gh - 1) as f32 * height,
                );
                rest[idx] = p;
                pos[idx] = p;
            }
        }
        Self {
            rest,
            pos,
            vel: vec![Vec2::ZERO; count],
            grabbed: -1,
            grab_target: Vec2::ZERO,
            active: false,
            width,
            height,
            grid_w: gw,
            grid_h: gh,
        }
    }

    pub fn grid_size(&self) -> usize {
        self.grid_w
    }

    /// Resize the grid, preserving any in-flight wobble as best we can.
    pub fn resize_grid(&mut self, new_size: usize) {
        if new_size == self.grid_w {
            return;
        }
        // Just reinitialize — any active wobble is lost, but that's fine
        *self = WobblySimulation::new(self.width, self.height, new_size);
    }

    pub fn grab(&mut self, u: f32, v: f32) {
        let gi = (u * (self.grid_w - 1) as f32).round().clamp(0.0, (self.grid_w - 1) as f32) as usize;
        let gj = (v * (self.grid_h - 1) as f32).round().clamp(0.0, (self.grid_h - 1) as f32) as usize;
        self.grabbed = (gj * self.grid_w + gi) as i32;
        self.grab_target = self.pos[self.grabbed as usize];
        self.active = true;
    }

    pub fn drag(&mut self, dx: f32, dy: f32) {
        self.grab_target.x += dx;
        self.grab_target.y += dy;
    }

    pub fn release(&mut self) {
        self.grabbed = -1;
    }

    pub fn step(&mut self, params: &WobblyParams, dt: f32) {
        if !self.active {
            return;
        }

        // If grid size changed, resize
        if params.grid_size != self.grid_w {
            self.resize_grid(params.grid_size);
        }

        let max_step = 1.0 / 120.0;
        let mut remaining = dt.min(0.05);
        while remaining > 0.0 {
            let step = remaining.min(max_step);
            remaining -= step;
            self.integrate(params, step);
        }
    }

    fn integrate(&mut self, params: &WobblyParams, dt: f32) {
        let k = params.stiffness;
        let c = params.damping;
        let gw = self.grid_w;
        let gh = self.grid_h;
        let x_spacing = self.width / (gw - 1) as f32;
        let y_spacing = self.height / (gh - 1) as f32;

        if self.grabbed >= 0 {
            let gi = self.grabbed as usize;
            self.pos[gi] = self.grab_target;
            self.vel[gi] = Vec2::ZERO;
        }

        for j in 0..gh {
            for i in 0..gw {
                let idx = j * gw + i;
                if idx as i32 == self.grabbed {
                    continue;
                }

                let pos = self.pos[idx];
                let mut force = Vec2::ZERO;

                if i + 1 < gw {
                    let n = self.pos[idx + 1];
                    let rest_delta = Vec2::new(x_spacing, 0.0);
                    force += k * ((n - pos) - rest_delta);
                }
                if i > 0 {
                    let n = self.pos[idx - 1];
                    let rest_delta = Vec2::new(-x_spacing, 0.0);
                    force += k * ((n - pos) - rest_delta);
                }
                if j + 1 < gh {
                    let n = self.pos[idx + gw];
                    let rest_delta = Vec2::new(0.0, y_spacing);
                    force += k * ((n - pos) - rest_delta);
                }
                if j > 0 {
                    let n = self.pos[idx - gw];
                    let rest_delta = Vec2::new(0.0, -y_spacing);
                    force += k * ((n - pos) - rest_delta);
                }

                force -= c * self.vel[idx];

                self.vel[idx] += force * dt;
                self.pos[idx] += self.vel[idx] * dt;
            }
        }

        if self.grabbed < 0 {
            let count = gw * gh;
            let total_energy: f32 = (0..count)
                .map(|i| self.vel[i].length_squared() + (self.pos[i] - self.rest[i]).length_squared() * 0.01)
                .sum();

            if total_energy < 0.00001 {
                for i in 0..count {
                    self.pos[i] = self.rest[i];
                    self.vel[i] = Vec2::ZERO;
                }
                self.active = false;
            }
        }
    }

    pub fn deform_offset(&self, u: f32, v: f32) -> Vec2 {
        let gw = self.grid_w;
        let gh = self.grid_h;
        let deg_x = gw - 1;
        let deg_y = gh - 1;
        let mut deformed = Vec2::ZERO;
        let mut rest = Vec2::ZERO;
        for j in 0..gh {
            let bj = bernstein(j, deg_y, v);
            for i in 0..gw {
                let bi = bernstein(i, deg_x, u);
                let w = bi * bj;
                deformed += self.pos[j * gw + i] * w;
                rest += self.rest[j * gw + i] * w;
            }
        }
        deformed - rest
    }

    pub fn shift_positions(&mut self, dx: f32, dy: f32) {
        let shift = Vec2::new(dx, dy);
        for p in &mut self.pos {
            *p -= shift;
        }
        self.grab_target -= shift;
    }
}

fn bernstein(i: usize, n: usize, t: f32) -> f32 {
    binomial(n, i) as f32 * t.powi(i as i32) * (1.0 - t).powi((n - i) as i32)
}

fn binomial(n: usize, k: usize) -> u64 {
    if k > n { return 0; }
    let mut result: u64 = 1;
    for i in 0..k.min(n - k) {
        result = result * (n - i) as u64 / (i + 1) as u64;
    }
    result
}

pub struct WobblyManager {
    pub simulations: HashMap<SurfaceId, WobblySimulation>,
}

impl WobblyManager {
    pub fn new() -> Self {
        Self { simulations: HashMap::new() }
    }

    pub fn ensure(&mut self, id: SurfaceId, w: f32, h: f32, grid_size: usize) -> &mut WobblySimulation {
        let sim = self.simulations.entry(id)
            .or_insert_with(|| WobblySimulation::new(w, h, grid_size));
        if sim.grid_size() != grid_size {
            sim.resize_grid(grid_size);
        }
        sim
    }

    pub fn get_mut(&mut self, id: SurfaceId) -> Option<&mut WobblySimulation> {
        self.simulations.get_mut(&id)
    }
}
