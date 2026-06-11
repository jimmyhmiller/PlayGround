// Mandelbrot escape-count sum — mirrors benchmarks/ail/mandelbrot.ail.

fn escape(c_re: f64, c_im: f64, max_iter: i64) -> i64 {
    let (mut z_re, mut z_im) = (0.0f64, 0.0f64);
    let mut i = 0;
    while i < max_iter {
        let zr2 = z_re * z_re;
        let zi2 = z_im * z_im;
        if zr2 + zi2 > 4.0 {
            return i;
        }
        let new_im = 2.0 * z_re * z_im + c_im;
        z_re = zr2 - zi2 + c_re;
        z_im = new_im;
        i += 1;
    }
    max_iter
}

fn main() {
    let (width, height, max_iter) = (1000i64, 1000i64, 100i64);
    let t0 = std::time::Instant::now();
    let mut acc: i64 = 0;
    for py in 0..height {
        let c_im = -1.25 + 2.5 * py as f64 / height as f64;
        for px in 0..width {
            let c_re = -2.0 + 3.0 * px as f64 / width as f64;
            acc += escape(c_re, c_im, max_iter);
        }
    }
    let ms = t0.elapsed().as_millis();
    println!("RESULT mandelbrot {} ms checksum={}", ms, acc);
}
