// Rust equivalent of examples/nbody.gcr — tight f64 + sqrt kernel (compute-only,
// no allocation). Faithful line-for-line translation; same loop count + inputs.
fn pair_dv(ax: f64, ay: f64, az: f64, _am: f64,
           bx: f64, by: f64, bz: f64, _bm: f64, dt: f64) -> f64 {
    let dx = ax - bx;
    let dy = ay - by;
    let dz = az - bz;
    let d2 = dx * dx + dy * dy + dz * dz;
    let dist = d2.sqrt();
    dt / (d2 * dist)
}
fn energy(x: f64, y: f64, z: f64, m: f64) -> f64 {
    let d = (x * x + y * y + z * z).sqrt() + 1.0;
    0.5 * m / d
}
fn main() {
    let n = 200000;
    let mut i = 0;
    let mut acc = 0.0;
    let mut px = 1.0;
    let mut py = 2.0;
    let mut pz = 3.0;
    let mut vx = 0.0;
    while i < n {
        let dv = pair_dv(px, py, pz, 1.0, 0.0, 0.0, 0.0, 1000.0, 0.01);
        vx = vx + dv;
        px = px + vx * 0.01;
        py = py + 0.001;
        pz = pz - 0.0005;
        acc = acc + energy(px, py, pz, 1.0);
        i = i + 1;
    }
    println!("{}", (acc * 1000.0) as i64);
}
