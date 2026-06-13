// Tight integer loop with xor/shift mix — mirrors benchmarks/ail/loop_mix.ail.
// ai-lang's `>>` builtin is a logical (zero-filling) shift, so use u64
// reinterpretation for the shift to match exactly.

fn main() {
    let n: i64 = 500_000_000;
    let t0 = std::time::Instant::now();
    let mut acc: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        acc = acc.wrapping_add(i) ^ (((acc as u64) >> 13) as i64);
        i += 1;
    }
    let ms = t0.elapsed().as_millis();
    println!("RESULT loop_mix {} ms checksum={}", ms, acc);
}
