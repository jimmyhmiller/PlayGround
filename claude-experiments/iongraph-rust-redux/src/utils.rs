// Port of utils.ts

pub fn clamp(x: f64, min: f64, max: f64) -> f64 {
    f64::max(min, f64::min(max, x))
}

pub fn filerp(current: f64, target: f64, r: f64, dt: f64) -> f64 {
    (current - target) * r.powf(dt) + target
}
