// N-body, 5 Jovian bodies — mirrors benchmarks/ail/nbody.ail (same
// column layout and op order so the energy checksum matches exactly).

const SOLAR_MASS: f64 = 39.478417604357434;
const DAYS_PER_YEAR: f64 = 365.24;
const N: usize = 5;

struct Sys {
    x: [f64; N], y: [f64; N], z: [f64; N],
    vx: [f64; N], vy: [f64; N], vz: [f64; N],
    m: [f64; N],
}

fn init_sys() -> Sys {
    let mut s = Sys {
        x: [0.0; N], y: [0.0; N], z: [0.0; N],
        vx: [0.0; N], vy: [0.0; N], vz: [0.0; N],
        m: [0.0; N],
    };
    let bodies: [(f64, f64, f64, f64, f64, f64, f64); N] = [
        // Sun.
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        // Jupiter.
        (4.84143144246472090, -1.16032004402742839, -0.103622044471123109,
         0.00166007664274403694, 0.00769901118419740425, -0.0000690460016972063023,
         0.000954791938424326609),
        // Saturn.
        (8.34336671824457987, 4.12479856412430479, -0.403523417114321381,
         -0.00276742510726862411, 0.00499852801234917238, 0.0000230417297573763929,
         0.000285885980666130812),
        // Uranus.
        (12.8943695621391310, -15.1111514016986312, -0.223307578892655734,
         0.00296460137564761618, 0.00237847173959480950, -0.0000296589568540237556,
         0.0000436624404335156298),
        // Neptune.
        (15.3796971148509165, -25.9193146099879641, 0.179258772950371181,
         0.00268067772490389322, 0.00162824170038242295, -0.0000951592254519715870,
         0.0000515138902046611451),
    ];
    for (i, b) in bodies.iter().enumerate() {
        s.x[i] = b.0;
        s.y[i] = b.1;
        s.z[i] = b.2;
        s.vx[i] = b.3 * DAYS_PER_YEAR;
        s.vy[i] = b.4 * DAYS_PER_YEAR;
        s.vz[i] = b.5 * DAYS_PER_YEAR;
        s.m[i] = b.6 * SOLAR_MASS;
    }
    s
}

fn offset_momentum(s: &mut Sys) {
    let (mut px, mut py, mut pz) = (0.0, 0.0, 0.0);
    for i in 0..N {
        px += s.vx[i] * s.m[i];
        py += s.vy[i] * s.m[i];
        pz += s.vz[i] * s.m[i];
    }
    s.vx[0] = -px / SOLAR_MASS;
    s.vy[0] = -py / SOLAR_MASS;
    s.vz[0] = -pz / SOLAR_MASS;
}

fn advance(s: &mut Sys, dt: f64) {
    for i in 0..N {
        for j in (i + 1)..N {
            let dx = s.x[i] - s.x[j];
            let dy = s.y[i] - s.y[j];
            let dz = s.z[i] - s.z[j];
            let d2 = dx * dx + dy * dy + dz * dz;
            let mag = dt / (d2 * d2.sqrt());
            let (mi, mj) = (s.m[i], s.m[j]);
            s.vx[i] -= dx * mj * mag;
            s.vy[i] -= dy * mj * mag;
            s.vz[i] -= dz * mj * mag;
            s.vx[j] += dx * mi * mag;
            s.vy[j] += dy * mi * mag;
            s.vz[j] += dz * mi * mag;
        }
    }
    for i in 0..N {
        s.x[i] += dt * s.vx[i];
        s.y[i] += dt * s.vy[i];
        s.z[i] += dt * s.vz[i];
    }
}

fn energy(s: &Sys) -> f64 {
    let mut e = 0.0;
    for i in 0..N {
        e += 0.5 * s.m[i]
            * (s.vx[i] * s.vx[i] + s.vy[i] * s.vy[i] + s.vz[i] * s.vz[i]);
        for j in (i + 1)..N {
            let dx = s.x[i] - s.x[j];
            let dy = s.y[i] - s.y[j];
            let dz = s.z[i] - s.z[j];
            e -= s.m[i] * s.m[j] / (dx * dx + dy * dy + dz * dz).sqrt();
        }
    }
    e
}

fn main() {
    let n = 500_000;
    let mut s = init_sys();
    offset_momentum(&mut s);
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        advance(&mut s, 0.01);
    }
    let ms = t0.elapsed().as_millis();
    let chk = (energy(&s) * 1_000_000_000.0) as i64;
    println!("RESULT nbody {} ms checksum={}", ms, chk);
}
