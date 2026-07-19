// Rust equivalent of examples/mandelbrot.gcr — f64 escape-count over a grid
// (compute-only, no allocation). Faithful translation; same grid + max_iter.
fn iter_count(cr: f64, ci: f64, max: i64) -> i64 {
    let mut zr = 0.0;
    let mut zi = 0.0;
    let mut i = 0;
    let mut done = 0;
    while i < max {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        if done == 0 {
            if zr2 + zi2 > 4.0 {
                done = i;
            } else {
                let new_zr = zr2 - zi2 + cr;
                let new_zi = 2.0 * zr * zi + ci;
                zr = new_zr;
                zi = new_zi;
            }
        }
        i = i + 1;
    }
    if done == 0 { max } else { done }
}
fn main() {
    let width = 80;
    let height = 50;
    let max_iter = 100;
    let mut py = 0;
    let mut checksum: i64 = 0;
    while py < height {
        let mut px = 0;
        while px < width {
            let cr = -2.5 + (px as f64) * 3.5 / 80.0;
            let ci = -1.25 + (py as f64) * 2.5 / 50.0;
            checksum = checksum + iter_count(cr, ci, max_iter);
            px = px + 1;
        }
        py = py + 1;
    }
    println!("{}", checksum);
}
