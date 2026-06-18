//! Zig-style allocators as a library: an allocator is an explicit value (a
//! vtable of function pointers) threaded through, with no language support.

mod common;
use common::build_and_run;

#[test]
fn allocator_strategies_are_swappable() {
    // The same `boxed-add` runs with a malloc allocator and an arena allocator.
    assert_eq!(build_and_run(include_str!("../examples/allocators.coil")), 42);
}

#[test]
fn arena_bumps_and_reuses_one_block() {
    // Allocate several i64s from an arena, fill them, sum them. Per-alloc free is
    // a no-op; the whole block is freed once at the end.
    let src = r#"
        (include "lib/alloc.coil")
        (defn main [] (-> :i64)
          (let [a (arena-allocator 1024)]
            (let [p (cast (ptr i64) (alloc-bytes a 8))
                  q (cast (ptr i64) (alloc-bytes a 8))
                  r (cast (ptr i64) (alloc-bytes a 8))]
              (store! p 10) (store! q 14) (store! r 18)
              (iadd (load p) (iadd (load q) (load r))))))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn sizeof_computes_layout_sizes() {
    // sizeof reflects real LLVM layout: i32=4, a 3xi64 struct=24, [14 x i8]=14.
    let src = r#"
        (defstruct V3 [(x :i64) (y :i64) (z :i64)])
        (defn main [] (-> :i64)
          (iadd (sizeof V3) (iadd (sizeof :i32) (sizeof (array i8 14)))))
    "#;
    assert_eq!(build_and_run(src), 42); // 24 + 4 + 14
}

#[test]
fn alloc_one_sizes_with_sizeof() {
    // The `alloc-one` macro allocates exactly sizeof(T) bytes via the allocator.
    let src = r#"
        (include "lib/alloc.coil")
        (defstruct Pair [(a :i64) (b :i64)])
        (defn main [] (-> :i64)
          (let [al (malloc-allocator)
                p  (alloc-one al Pair)]
            (store! (field p a) 19) (store! (field p b) 23)
            (let [s (iadd (load (field p a)) (load (field p b)))]
              (free-bytes al (cast (ptr i8) p))
              s)))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn allocator_is_visible_in_the_signature() {
    // A function with no allocator parameter simply has no way to allocate
    // dynamically — the capability must be threaded in. (Here: it type-checks
    // only because the allocator is a parameter.)
    let src = r#"
        (include "lib/alloc.coil")
        (defn one [(a (ptr Allocator))] (-> :i64)
          (let [p (cast (ptr i64) (alloc-bytes a 8))]
            (store! p 42)
            (let [v (load p)] (free-bytes a (cast (ptr i8) p)) v)))
        (defn main [] (-> :i64) (one (malloc-allocator)))
    "#;
    assert_eq!(build_and_run(src), 42);
}
