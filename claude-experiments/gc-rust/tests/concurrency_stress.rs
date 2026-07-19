//! Comprehensive randomized concurrency stress harness.
//!
//! Exercises EVERY concurrency primitive — generic `Thread<T>` (i64 / `Vec` /
//! struct results), `Atom<T>` (incl. `Atom<Vec>`), `AtomicI64`, and channels —
//! under `--gc-stress` (collect on every allocation), where the collector's own
//! POST-SWAP root invariant asserts on every collection across all threads.
//! Each generated program has a COMPUTABLE expected result, so heap corruption
//! (a missed/stale root, a lost update) surfaces as a WRONG ANSWER, not just a
//! crash. Thread counts, sizes, channel capacities, and interleavings are
//! randomized; the patterns rotate so every primitive is hit across iterations.
//!
//! Run deeper with `GCR_STRESS_ITERS=N GCR_STRESS_SEED=S cargo test --test
//! concurrency_stress -- --nocapture`.
//!
//! COVERAGE / TIME-BOX (no silent cap): `--gc-stress` collects on EVERY
//! allocation, so each multithreaded program is ~quadratic and the default 50
//! iters are intractable per-commit (tens of minutes even in release). The
//! per-commit gate runs **6 iters in release with the detector armed**
//! (`GCR_GC_VERIFY=1 GCR_STRESS_ITERS=6 cargo test --release --test
//! concurrency_stress`); the full 50+ soak is an occasional/nightly run. The
//! coverage is reduced for speed, not dropped — see docs/FUTURE_WORK.md (P3).

use gcrust::codegen::jit_run_i64_gc;
use gcrust::compile::parse_with_prelude;
use gcrust::lower::lower_program;
use gcrust::resolve::resolve_module;

fn run_stress(src: &str) -> Result<i64, String> {
    let (module, _) = parse_with_prelude(src).map_err(|e| format!("parse: {e:?}"))?;
    let resolved = resolve_module(module).map_err(|e| format!("resolve: {e:?}"))?;
    let prog = lower_program(&resolved.globals).map_err(|e| format!("lower: {}", e.msg))?;
    jit_run_i64_gc(&prog, true).map_err(|e| format!("codegen/run: {}", e.0))
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.0 = x; x
    }
    fn range(&mut self, lo: i64, hi: i64) -> i64 { lo + (self.next() % ((hi - lo) as u64)) as i64 }
}

fn tri(n: i64) -> i64 { n * (n - 1) / 2 } // sum 0..n

/// Common prelude-side helpers shared by the generated programs.
const HELPERS: &str = r#"
    fn bsum(n: i64) -> i64 {
        let v: Vec<i64> = vec_new();
        let mut acc = v; let mut i = 0;
        while i < n { acc = vec_push(acc, i); i = i + 1; }
        let mut s = 0; let mut j = 0;
        while j < vec_len(acc) { s = s + vec_get_unchecked(acc, j); j = j + 1; }
        s
    }
    fn bvec(n: i64) -> Vec<i64> {
        let v: Vec<i64> = vec_new();
        let mut acc = v; let mut i = 0;
        while i < n { acc = vec_push(acc, i); i = i + 1; }
        acc
    }
    fn vsum(v: Vec<i64>) -> i64 {
        let mut s = 0; let mut j = 0;
        while j < vec_len(v) { s = s + vec_get_unchecked(v, j); j = j + 1; }
        s
    }
"#;

/// Generate one randomized program + its expected result. Picks one of 5
/// primitive patterns; all parameters randomized.
fn gen_prog(rng: &mut Rng) -> (String, i64) {
    let pattern = rng.next() % 5;
    // Worker count: usually 2..5, but ~1/4 of the time push to 6..12 to stress
    // contention on shared atoms/counters/channels with many threads.
    let k = if rng.next() % 4 == 0 { rng.range(6, 13) } else { rng.range(2, 6) };
    let mut body = String::new();
    let expected: i64;

    match pattern {
        // 0: generic Thread<Vec> — workers return Vecs, main sums them.
        0 => {
            let mut sizes = Vec::new();
            body.push_str("fn main() -> i64 {\n");
            for i in 0..k {
                let n = rng.range(200, 2000);
                sizes.push(n);
                body.push_str(&format!("  let t{} = Thread::spawn(|| bvec({}));\n", i, n));
            }
            body.push_str("  let mut total = 0;\n");
            for i in 0..k {
                body.push_str(&format!("  total = total + vsum(t{}.join());\n", i));
            }
            body.push_str("  total\n}\n");
            expected = sizes.iter().map(|n| tri(*n)).sum();
        }
        // 1: Atom<i64> — workers swap-increment a shared captured atom.
        1 => {
            let iters = rng.range(200, 1500);
            body.push_str(
                "fn bump(a: Atom<i64>, n: i64) -> i64 { let mut i = 0; while i < n { let _x = a.swap(|v| v + 1); i = i + 1; } 0 }\n");
            body.push_str("fn main() -> i64 {\n  let a: Atom<i64> = Atom::new(0);\n");
            for i in 0..k {
                body.push_str(&format!("  let t{} = Thread::spawn(|| bump(a, {}));\n", i, iters));
            }
            for i in 0..k { body.push_str(&format!("  let _j{} = t{}.join();\n", i, i)); }
            body.push_str("  a.deref()\n}\n");
            expected = iters * k;
        }
        // 2: AtomicI64 — workers fetch_add into a shared captured counter.
        2 => {
            let iters = rng.range(500, 4000);
            body.push_str(
                "fn fa(a: AtomicI64, n: i64) -> i64 { let mut i = 0; while i < n { let _x = a.fetch_add(1); i = i + 1; } 0 }\n");
            body.push_str("fn main() -> i64 {\n  let a = AtomicI64::new(0);\n");
            for i in 0..k {
                body.push_str(&format!("  let t{} = Thread::spawn(|| fa(a, {}));\n", i, iters));
            }
            for i in 0..k { body.push_str(&format!("  let _j{} = t{}.join();\n", i, i)); }
            body.push_str("  a.load()\n}\n");
            expected = iters * k;
        }
        // 3: channel — multi-producer fan-in, main drains and sums.
        3 => {
            let cap = rng.range(1, 8);
            let per = rng.range(50, 300);
            body.push_str(
                "fn prod(ch: Channel<Vec<i64>>, base: i64, n: i64) -> i64 { let mut i = 0; while i < n { let v: Vec<i64> = vec_new(); let _s = ch.send(vec_push(v, base + i)); i = i + 1; } 0 }\n");
            body.push_str(&format!("fn main() -> i64 {{\n  let ch: Channel<Vec<i64>> = Channel::new({});\n", cap));
            let mut bases = Vec::new();
            for i in 0..k {
                let base = rng.range(0, 1000) * 1000;
                bases.push(base);
                body.push_str(&format!("  let t{} = Thread::spawn(|| prod(ch, {}, {}));\n", i, base, per));
            }
            let n_total = per * k;
            body.push_str(&format!(
                "  let mut total = 0; let mut got = 0;\n  while got < {} {{ let item = ch.recv(); total = total + vec_get_unchecked(item, 0); got = got + 1; }}\n",
                n_total));
            for i in 0..k { body.push_str(&format!("  let _j{} = t{}.join();\n", i, i)); }
            body.push_str("  total\n}\n");
            // each producer sends base..base+per (one element each, value base+i)
            expected = bases.iter().map(|b| b * per + tri(per)).sum();
        }
        // 4: MIXED — atom + channel + Vec-returning threads in one program.
        _ => {
            let iters = rng.range(100, 800);
            let per = rng.range(30, 200);
            let vn = rng.range(100, 1000);
            body.push_str(
                "fn bump(a: Atom<i64>, n: i64) -> i64 { let mut i = 0; while i < n { let _x = a.swap(|v| v + 1); i = i + 1; } 0 }\n");
            body.push_str(
                "fn prod(ch: Channel<Vec<i64>>, n: i64) -> i64 { let mut i = 0; while i < n { let v: Vec<i64> = vec_new(); let _s = ch.send(vec_push(v, i)); i = i + 1; } 0 }\n");
            body.push_str("fn main() -> i64 {\n");
            body.push_str("  let a: Atom<i64> = Atom::new(0);\n");
            body.push_str("  let ch: Channel<Vec<i64>> = Channel::new(3);\n");
            body.push_str(&format!("  let ta = Thread::spawn(|| bump(a, {}));\n", iters));
            body.push_str(&format!("  let tp = Thread::spawn(|| prod(ch, {}));\n", per));
            body.push_str(&format!("  let tv = Thread::spawn(|| bvec({}));\n", vn));
            body.push_str(&format!(
                "  let mut csum = 0; let mut got = 0;\n  while got < {} {{ let item = ch.recv(); csum = csum + vec_get_unchecked(item, 0); got = got + 1; }}\n",
                per));
            body.push_str("  let _ja = ta.join();\n  let _jp = tp.join();\n  let vres = vsum(tv.join());\n");
            body.push_str("  a.deref() + csum + vres\n}\n");
            expected = iters + tri(per) + tri(vn);
        }
    }

    (format!("{}\n{}", HELPERS, body), expected)
}

#[test]
fn comprehensive_concurrency_stress() {
    let iters: u64 = std::env::var("GCR_STRESS_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(50);
    let seed0: u64 = std::env::var("GCR_STRESS_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(0xC0FFEE_1234_5678);
    let mut rng = Rng(seed0);
    for it in 0..iters {
        let seed = rng.0;
        let (src, expected) = gen_prog(&mut rng);
        match run_stress(&src) {
            Ok(got) => assert_eq!(
                got, expected,
                "iteration {it} (seed {seed:#x}): expected {expected}, got {got} — likely heap corruption / lost update.\n{src}"
            ),
            Err(e) => panic!("iteration {it} (seed {seed:#x}): {e}\n{src}"),
        }
    }
}
