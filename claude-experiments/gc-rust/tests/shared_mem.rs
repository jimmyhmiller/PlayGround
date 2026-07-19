//! Shared-memory concurrency primitives (M3): AtomicI64 (and later Atom<T>).
//!
//! The headline test is `atomic_counter_no_lost_updates`: N threads each do many
//! `fetch_add`s into one shared counter; an exact final total proves the atomic
//! is genuinely lock-free-correct under contention (a broken atomic loses
//! updates and undercounts).

use gcrust::codegen::{jit_run_i64, jit_run_i64_gc};
use gcrust::compile::parse_with_prelude;
use gcrust::lower::lower_program;
use gcrust::resolve::resolve_module;

fn run(src: &str) -> i64 {
    let (module, _) = parse_with_prelude(src).unwrap();
    let resolved = resolve_module(module).unwrap();
    let prog = lower_program(&resolved.globals).unwrap();
    jit_run_i64(&prog).unwrap()
}
fn run_stress(src: &str) -> i64 {
    let (module, _) = parse_with_prelude(src).unwrap();
    let resolved = resolve_module(module).unwrap();
    let prog = lower_program(&resolved.globals).unwrap();
    jit_run_i64_gc(&prog, true).unwrap()
}
fn lower_err(src: &str) -> Option<String> {
    let (module, _) = parse_with_prelude(src).unwrap();
    let resolved = resolve_module(module).unwrap();
    lower_program(&resolved.globals).err().map(|e| e.msg)
}

// --- Channels: bounded, blocking, GC values cross threads ---

#[test]
fn channel_producer_consumer() {
    // A producer thread sends 5 Vecs through a bounded(2) channel; main receives
    // and sums. Bounded buffer => producer blocks on send when full, consumer's
    // recv unblocks it (real parking). GC values cross threads via the heap buf.
    let src = r#"
        fn produce(ch: Channel<Vec<i64>>, n: i64) -> i64 {
            let mut i = 0;
            while i < n { let v: Vec<i64> = vec_new(); let _s = ch.send(vec_push(v, i)); i = i + 1; }
            let _c = ch.close();
            0
        }
        fn main() -> i64 {
            let ch: Channel<Vec<i64>> = Channel::new(2);
            let t = Thread::spawn(|| produce(ch, 5));
            let mut total = 0; let mut got = 0;
            while got < 5 { let item = ch.recv(); total = total + vec_get_unchecked(item, 0); got = got + 1; }
            let _j = t.join();
            total
        }
    "#;
    assert_eq!(run(src), 10);
}

#[test]
fn channel_under_gc_stress() {
    let src = r#"
        fn produce(ch: Channel<Vec<i64>>, n: i64) -> i64 {
            let mut i = 0;
            while i < n { let v: Vec<i64> = vec_new(); let _s = ch.send(vec_push(v, i)); i = i + 1; }
            0
        }
        fn main() -> i64 {
            let ch: Channel<Vec<i64>> = Channel::new(2);
            let t = Thread::spawn(|| produce(ch, 5));
            let mut total = 0; let mut got = 0;
            while got < 5 { let item = ch.recv(); total = total + vec_get_unchecked(item, 0); got = got + 1; }
            let _j = t.join();
            total
        }
    "#;
    assert_eq!(run_stress(src), 10);
}

#[test]
fn channel_multi_producer_fan_in() {
    let src = r#"
        fn produce(ch: Channel<Vec<i64>>, base: i64, n: i64) -> i64 {
            let mut i = 0;
            while i < n { let v: Vec<i64> = vec_new(); let _s = ch.send(vec_push(v, base + i)); i = i + 1; }
            0
        }
        fn main() -> i64 {
            let ch: Channel<Vec<i64>> = Channel::new(4);
            let t1 = Thread::spawn(|| produce(ch, 0, 100));
            let t2 = Thread::spawn(|| produce(ch, 1000, 100));
            let mut total = 0; let mut got = 0;
            while got < 200 { let item = ch.recv(); total = total + vec_get_unchecked(item, 0); got = got + 1; }
            let _a = t1.join(); let _b = t2.join();
            total
        }
    "#;
    assert_eq!(run(src), 109900);
}

// --- Send/Sync: shared mutation is sound, and a concurrent map is allowed ---

#[test]
fn spawn_rejects_capturing_mutable_ref_struct() {
    // A plain reference struct shared across threads is a data race (any holder
    // can mutate it). Capturing it into a spawn must be a COMPILE ERROR.
    let src = r#"
        struct Counter { n: i64 }
        fn main() -> i64 {
            let c = Counter { n: 0 };
            let t = Thread::spawn(|| c.n);
            t.join()
        }
    "#;
    let err = lower_err(src).expect("expected a Sync error");
    assert!(err.contains("not `Sync`") && err.contains("Counter"), "unexpected: {err}");
}

#[test]
fn spawn_allows_atom_capture() {
    // Atom is Sync — capturing it is fine.
    let src = r#"
        fn main() -> i64 {
            let a: Atom<i64> = Atom::new(5);
            let t = Thread::spawn(|| a.deref());
            t.join() + a.deref()
        }
    "#;
    assert_eq!(run(src), 10);
}

#[test]
fn spawn_allows_struct_built_from_atoms() {
    // A struct whose fields are all Atom is auto-Sync — the concurrent-map
    // pattern. Two threads concurrently swap its atoms; total is exact.
    let src = r#"
        struct Shared { a: Atom<i64>, b: Atom<i64> }
        fn worker(s: Shared, n: i64) -> i64 {
            let mut i = 0;
            while i < n { let _x = s.a.swap(|v| v + 1); let _y = s.b.swap(|v| v + 2); i = i + 1; }
            0
        }
        fn main() -> i64 {
            let s = Shared { a: Atom::new(0), b: Atom::new(0) };
            let t = Thread::spawn(|| worker(s, 1000));
            let _l = worker(s, 1000);
            let _j = t.join();
            s.a.deref() + s.b.deref()
        }
    "#;
    assert_eq!(run(src), 6000);
}

#[test]
fn atomic_i64_single_threaded_ops() {
    let src = r#"
        fn main() -> i64 {
            let a = AtomicI64::new(10);
            let _u = a.store(100);
            let prev = a.fetch_add(5);              // prev=100, now 105
            let ok = a.compare_and_set(105, 999);   // true, now 999
            prev + a.load() + (if ok { 1 } else { 0 })
        }
    "#;
    assert_eq!(run(src), 1100);
}

#[test]
fn atomic_counter_no_lost_updates() {
    let src = r#"
        fn bump(a: AtomicI64, n: i64) -> i64 {
            let mut i = 0;
            while i < n { let _p = a.fetch_add(1); i = i + 1; }
            0
        }
        fn main() -> i64 {
            let counter = AtomicI64::new(0);
            let t1 = Thread::spawn(|| bump(counter, 10000));
            let t2 = Thread::spawn(|| bump(counter, 10000));
            let t3 = Thread::spawn(|| bump(counter, 10000));
            let _l = bump(counter, 10000);
            let _a = t1.join(); let _b = t2.join(); let _c = t3.join();
            counter.load()
        }
    "#;
    assert_eq!(run(src), 40000);
}

// Regression: capturing a VALUE STRUCT (e.g. AtomicI64 = {cell: RawPtr}) into a
// closure used to silently store nothing (gen_make_closure dropped Repr::Value
// captures), so the captured cell read as null. This caught it.
#[test]
fn value_struct_capture_in_closure() {
    let src = r#"
        fn main() -> i64 {
            let a = AtomicI64::new(7);
            let f = || a.load();
            f()
        }
    "#;
    assert_eq!(run(src), 7);
}

// --- Clojure-style Atom<T> ---

#[test]
fn atom_single_threaded_ops() {
    let src = r#"
        fn main() -> i64 {
            let a: Atom<i64> = Atom::new(10);
            let _v = a.swap(|x| x + 5);            // 15
            let ok = a.compare_and_set(15, 100);   // true -> 100
            let _r = a.reset(7);                   // 7
            a.deref() + (if ok { 1 } else { 0 })
        }
    "#;
    assert_eq!(run(src), 8);
}

#[test]
fn atom_concurrent_swap_no_lost_updates() {
    // N threads each swap-increment; the CAS-retry loop must not lose updates.
    let src = r#"
        fn bump(a: Atom<i64>, iters: i64) -> i64 {
            let mut i = 0;
            while i < iters { let _n = a.swap(|x| x + 1); i = i + 1; }
            0
        }
        fn main() -> i64 {
            let a: Atom<i64> = Atom::new(0);
            let t1 = Thread::spawn(|| bump(a, 5000));
            let t2 = Thread::spawn(|| bump(a, 5000));
            let t3 = Thread::spawn(|| bump(a, 5000));
            let _l = bump(a, 5000);
            let _x = t1.join(); let _y = t2.join(); let _z = t3.join();
            a.deref()
        }
    "#;
    assert_eq!(run(src), 20000);
}

#[test]
fn atom_of_vec_single_threaded() {
    // The real Clojure use case: an atom holding an immutable Vec, swapped
    // functionally. Exercises pointer-CAS on a heap (Ref) value.
    let src = r#"
        fn main() -> i64 {
            let a: Atom<Vec<i64>> = Atom::new(vec_new());
            let _v = a.swap(|old| vec_push(old, 10));
            let _w = a.swap(|old| vec_push(old, 20));
            let cur = a.deref();
            vec_get_unchecked(cur, 0) + vec_get_unchecked(cur, 1)
        }
    "#;
    assert_eq!(run(src), 30);
}

#[test]
fn atom_of_vec_concurrent_swap_under_stress() {
    // Pointer-CAS on a RELOCATING heap value under contention: each retry builds
    // a fresh Vec and GC can move old+new mid-swap. No pushes may be lost.
    let src = r#"
        fn pushn(a: Atom<Vec<i64>>, n: i64) -> i64 {
            let mut i = 0;
            while i < n { let _v = a.swap(|old| vec_push(old, 1)); i = i + 1; }
            0
        }
        fn main() -> i64 {
            let a: Atom<Vec<i64>> = Atom::new(vec_new());
            let t1 = Thread::spawn(|| pushn(a, 300));
            let t2 = Thread::spawn(|| pushn(a, 300));
            let _l = pushn(a, 300);
            let _x = t1.join(); let _y = t2.join();
            vec_len(a.deref())
        }
    "#;
    assert_eq!(run_stress(src), 900);
}

#[test]
fn atom_concurrent_swap_under_stress() {
    let src = r#"
        fn bump(a: Atom<i64>, iters: i64) -> i64 {
            let mut i = 0;
            while i < iters { let _n = a.swap(|x| x + 1); i = i + 1; }
            0
        }
        fn main() -> i64 {
            let a: Atom<i64> = Atom::new(0);
            let t1 = Thread::spawn(|| bump(a, 1000));
            let t2 = Thread::spawn(|| bump(a, 1000));
            let _l = bump(a, 1000);
            let _x = t1.join(); let _y = t2.join();
            a.deref()
        }
    "#;
    assert_eq!(run_stress(src), 3000);
}

#[test]
fn atomic_counter_under_gc_stress() {
    // The counter handle is off-heap (stable), but the spawned closures + their
    // captures live on the moving heap; collect-every-alloc shakes out any
    // handoff/capture relocation issue.
    let src = r#"
        fn bump(a: AtomicI64, n: i64) -> i64 {
            let mut i = 0;
            while i < n { let _p = a.fetch_add(1); i = i + 1; }
            0
        }
        fn main() -> i64 {
            let counter = AtomicI64::new(0);
            let t1 = Thread::spawn(|| bump(counter, 2000));
            let t2 = Thread::spawn(|| bump(counter, 2000));
            let _l = bump(counter, 2000);
            let _a = t1.join(); let _b = t2.join();
            counter.load()
        }
    "#;
    assert_eq!(run_stress(src), 6000);
}
