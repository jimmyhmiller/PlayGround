//! Randomized multi-thread stress harness.
//!
//! TSan/Miri can't see our JIT/AOT mutator code, so external race detectors are
//! out. Instead we lean on the collector's OWN invariant checks: under
//! `--gc-stress` in a debug build, every collection asserts that all roots point
//! into live space after a flip (`heap.rs` POST-SWAP check) — a relocating-GC
//! "did any thread keep a stale pointer across a GC" detector that runs from
//! inside the engine, seeing exactly the JIT roots TSan cannot.
//!
//! On top of that, each generated program computes a result the harness can
//! predict, so heap corruption shows up as a WRONG ANSWER, not just a crash. We
//! vary thread count, per-thread allocation volume, and join/sleep interleaving,
//! and run many iterations to hit rare schedules.

use gcrust::codegen::jit_run_i64_gc;
use gcrust::compile::parse_with_prelude;
use gcrust::lower::lower_program;
use gcrust::resolve::resolve_module;

fn run_stress(src: &str) -> i64 {
    let (module, _) = parse_with_prelude(src).unwrap();
    let resolved = resolve_module(module).unwrap();
    let prog = lower_program(&resolved.globals).unwrap();
    jit_run_i64_gc(&prog, true).unwrap() // stress = semi-space + collect-every-alloc
}

/// A tiny deterministic PRNG (xorshift) so failures are reproducible from a seed.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % ((hi - lo) as u64)) as i64
    }
}

/// Sum of 0..n.
fn tri(n: i64) -> i64 {
    n * (n - 1) / 2
}

/// Generate a program that spawns `k` worker threads, each building+summing a
/// Vec of `sizes[i]` elements, optionally with a sleep and with the main thread
/// holding a live Vec across all the joins (so its roots must survive relocation
/// while it is blocked). Returns (source, expected_result).
fn gen_program(rng: &mut Rng) -> (String, i64) {
    let k = rng.range(2, 6); // 2..5 worker threads
    let main_live = rng.range(50, 200); // main's live Vec, held across joins
    // Half the time, workers ACCUMULATE into a shared atom they CAPTURE (rather
    // than only returning their result). This exercises captured closures + the
    // atom CAS under the general harness — the case the original harness never
    // hit, which hid the closure code-pointer bug.
    let use_atom = rng.next() % 2 == 0;
    let mut sizes = Vec::new();
    let mut src = String::new();
    src.push_str(
        "fn build_and_sum(n: i64) -> i64 {\n\
         \x20 let v: Vec<i64> = vec_new();\n\
         \x20 let mut acc = v;\n\
         \x20 let mut i = 0;\n\
         \x20 while i < n { acc = vec_push(acc, i); i = i + 1; }\n\
         \x20 let mut s = 0; let mut j = 0;\n\
         \x20 while j < vec_len(acc) { s = s + vec_get_unchecked(acc, j); j = j + 1; }\n\
         \x20 s\n\
         }\n\
         fn accum(a: Atom<i64>, n: i64) -> i64 {\n\
         \x20 let r = build_and_sum(n);\n\
         \x20 let _u = a.swap(|x| x + r);\n\
         \x20 0\n\
         }\n",
    );
    src.push_str("fn main() -> i64 {\n");
    // main's own live root, held across the spawns/joins.
    src.push_str("  let mine: Vec<i64> = vec_new();\n  let mut m = mine;\n  let mut mk = 0;\n");
    src.push_str(&format!(
        "  while mk < {} {{ m = vec_push(m, mk); mk = mk + 1; }}\n",
        main_live
    ));
    if use_atom {
        src.push_str("  let acc: Atom<i64> = Atom::new(0);\n");
    }
    // Spawn the workers. In atom mode each worker CAPTURES `acc` and adds its
    // sum into it; otherwise it returns its sum to be joined.
    for i in 0..k {
        let n = rng.range(500, 4000);
        sizes.push(n);
        if use_atom {
            src.push_str(&format!("  let t{} = Thread::spawn(|| accum(acc, {}));\n", i, n));
        } else {
            src.push_str(&format!("  let t{} = Thread::spawn(|| build_and_sum({}));\n", i, n));
        }
    }
    // Optionally a sleep on main between spawn and join (exercises the blocked
    // transition while children allocate).
    if rng.next() % 2 == 0 {
        src.push_str("  let _z = thread_sleep(1);\n");
    }
    // Main does some of its own allocation work too, concurrently.
    let main_work = rng.range(500, 3000);
    sizes.push(main_work);
    if use_atom {
        src.push_str(&format!("  let _local = accum(acc, {});\n", main_work));
        // Join all (results are 0 in atom mode; the sum lives in the atom).
        for i in 0..k {
            src.push_str(&format!("  let _j{} = t{}.join();\n", i, i));
        }
        src.push_str("  let total = acc.deref();\n");
    } else {
        src.push_str(&format!("  let local = build_and_sum({});\n", main_work));
        src.push_str("  let mut total = local");
        for i in 0..k {
            src.push_str(&format!(" + t{}.join()", i));
        }
        src.push_str(";\n");
    }
    src.push_str(
        "  let mut ms = 0; let mut mj = 0;\n\
         \x20 while mj < vec_len(m) { ms = ms + vec_get_unchecked(m, mj); mj = mj + 1; }\n\
         \x20 total + ms\n}\n",
    );

    // Expected: sum over all worker+main sizes of tri(size), plus main's live sum.
    let mut expected = 0;
    for n in &sizes {
        expected += tri(*n);
    }
    expected += tri(main_live);
    (src, expected)
}

#[test]
fn randomized_multithread_stress() {
    // Each iteration: a fresh randomized program, run under collect-every-alloc.
    // A wrong answer = heap corruption (missed/stale root); a hang = deadlock; a
    // panic = invariant-assert tripped inside the collector.
    let iters: u64 = std::env::var("GCR_STRESS_ITERS").ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    let seed0: u64 = std::env::var("GCR_STRESS_SEED").ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0x9E3779B97F4A7C15);
    let mut rng = Rng(seed0);
    for it in 0..iters {
        let seed = rng.0;
        let (src, expected) = gen_program(&mut rng);
        let got = run_stress(&src);
        assert_eq!(
            got, expected,
            "iteration {it} (seed {seed:#x}) mismatch — likely heap corruption.\nProgram:\n{src}"
        );
    }
}

/// Concurrent atom swaps from closures that CAPTURE the shared atom. This is the
/// case the original harness missed (its spawned closures took only literals, so
/// no GC pointer was captured) — it segfaulted on the closure code-pointer offset
/// bug. Under collect-every-alloc with the collector's invariant checks, an exact
/// final count proves both the captured-closure handoff and the CAS-retry are
/// correct under contention + relocation.
#[test]
fn concurrent_atom_swap_captured_under_stress() {
    let mut rng = Rng(0xD1B54A32D192ED03);
    for it in 0..15 {
        let k = rng.range(2, 5);
        let iters = rng.range(200, 800);
        let mut src = String::new();
        src.push_str(
            "fn bump(a: Atom<i64>, n: i64) -> i64 {\n\
             \x20 let mut i = 0;\n\
             \x20 while i < n { let _x = a.swap(|v| v + 1); i = i + 1; }\n\
             \x20 0\n}\n\
             fn main() -> i64 {\n\
             \x20 let a: Atom<i64> = Atom::new(0);\n",
        );
        for i in 0..k {
            src.push_str(&format!("  let t{} = Thread::spawn(|| bump(a, {}));\n", i, iters));
        }
        src.push_str(&format!("  let _l = bump(a, {});\n", iters));
        for i in 0..k {
            src.push_str(&format!("  let _j{} = t{}.join();\n", i, i));
        }
        src.push_str("  a.deref()\n}\n");
        let expected = iters * (k + 1);
        assert_eq!(run_stress(&src), expected, "iteration {it}: atom lost updates\n{src}");
    }
}

/// Spawn/join churn: repeatedly spawn-and-immediately-join in a loop, so the
/// thread registry + env handoff are exercised rapidly while GC fires.
#[test]
fn spawn_join_churn_stress() {
    let src = r#"
        fn work(n: i64) -> i64 {
            let v: Vec<i64> = vec_new();
            let mut acc = v;
            let mut i = 0;
            while i < n { acc = vec_push(acc, i); i = i + 1; }
            vec_len(acc)
        }
        fn main() -> i64 {
            let mut total = 0;
            let mut r = 0;
            while r < 20 {
                let t = Thread::spawn(|| work(300));
                total = total + t.join();
                r = r + 1;
            }
            total
        }
    "#;
    // 20 rounds * 300 = 6000
    assert_eq!(run_stress(src), 6000);
}
