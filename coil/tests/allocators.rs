//! Zig-style allocators as a library: an allocator is an explicit value (a
//! vtable of function pointers) threaded through, with no language support.
//! The interface is alignment-aware and signals failure with a sum type
//! (Option), never a null sentinel.

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
          (let [a (arena-allocator 1024)
                p (unwrap-ptr (create [i64] a))
                q (unwrap-ptr (create [i64] a))
                r (unwrap-ptr (create [i64] a))]
            (store! p 10) (store! q 14) (store! r 18)
            (iadd (load p) (iadd (load q) (load r)))))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn typed_create_sizes_and_aligns_itself() {
    // `create` allocates exactly sizeof(T) bytes; `destroy` infers T from the ptr.
    let src = r#"
        (include "lib/alloc.coil")
        (defstruct Pair [(a :i64) (b :i64)])
        (defn main [] (-> :i64)
          (let [al (malloc-allocator)]
            (match (create [Pair] al)
              (None [] 0)
              (Some [p]
                (do (store! (field p a) 19) (store! (field p b) 23)
                    (let [s (iadd (load (field p a)) (load (field p b)))]
                      (destroy al p)
                      s))))))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn alloc_slice_allocates_an_array() {
    // alloc-slice gives n contiguous elements; index walks them.
    let src = r#"
        (include "lib/alloc.coil")
        (defn main [] (-> :i64)
          (let [al (malloc-allocator)
                xs (unwrap-ptr (alloc-slice [i64] al 3))]
            (store! (index xs 0) 12)
            (store! (index xs 1) 14)
            (store! (index xs 2) 16)
            (iadd (load (index xs 0)) (iadd (load (index xs 1)) (load (index xs 2))))))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn allocation_failure_is_an_option() {
    // A tiny arena (8 bytes) can hand out one i64, then returns None.
    let src = r#"
        (include "lib/alloc.coil")
        (defn main [] (-> :i64)
          (let [a (arena-allocator 8)]
            (match (create [i64] a)
              (None [] 1)
              (Some [p]
                (match (create [i64] a)          ; second alloc overflows -> None
                  (None [] 42)
                  (Some [q] 2))))))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn allocator_is_visible_in_the_signature() {
    // A function with no allocator parameter has no way to allocate dynamically
    // -- the capability must be threaded in.
    let src = r#"
        (include "lib/alloc.coil")
        (defn one [(a (ptr Allocator))] (-> :i64)
          (match (create [i64] a)
            (None [] 0)
            (Some [p] (do (store! p 42) (let [v (load p)] (destroy a p) v)))))
        (defn main [] (-> :i64) (one (malloc-allocator)))
    "#;
    assert_eq!(build_and_run(src), 42);
}
