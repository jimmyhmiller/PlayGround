//! lib/mem.coil — low-level typed memory ops (copy/move/zero/fill/eql) as generic
//! wrappers over libc, sizing themselves with sizeof. No allocator, no other Coil
//! module — foundational.

mod common;
use common::build_and_run;

const IMPORT: &str = "(module app)\n(import \"lib/mem.coil\" :use *)\n";

fn run(body: &str) -> i32 {
    build_and_run(&format!("{IMPORT}{body}"))
}

#[test]
fn fill_copy_and_eql() {
    // fill src with 7, copy to dst, they compare equal; element survives.
    let code = run(
        r#"(defn main [] (-> :i64)
             (let [src (alloc-stack (array i64 4)) dst (alloc-stack (array i64 4))]
               (mem-fill (index src 0) 7 4)
               (mem-copy (index dst 0) (index src 0) 4)
               (iadd (load (index dst 3))
                     (if (mem-eql (index src 0) (index dst 0) 4) 100 0))))"#, // 7 + 100
    );
    assert_eq!(code, 107);
}

#[test]
fn zero_makes_eql_false() {
    let code = run(
        r#"(defn main [] (-> :i64)
             (let [src (alloc-stack (array i64 4)) dst (alloc-stack (array i64 4))]
               (mem-fill (index src 0) 7 4)
               (mem-fill (index dst 0) 7 4)
               (mem-zero (index src 0) 4)               ; src now zeros
               (if (mem-eql (index src 0) (index dst 0) 4) 1 42)))"#, // not equal → 42
    );
    assert_eq!(code, 42);
}

#[test]
fn mem_move_handles_overlap() {
    // Shift [1,2,3,4,5] right by one in place: dst overlaps src. memmove-safe.
    let code = run(
        r#"(defn main [] (-> :i64)
             (let [a (alloc-stack (array i64 5))]
               (mem-fill (index a 0) 0 5)
               (store! (index a 0) 1) (store! (index a 1) 2) (store! (index a 2) 3)
               (store! (index a 3) 4) (store! (index a 4) 5)
               (mem-move (index a 1) (index a 0) 4)     ; a = [1,1,2,3,4]
               (iadd (imul (load (index a 1)) 10) (load (index a 4)))))"#, // 1*10 + 4 = 14
    );
    assert_eq!(code, 14);
}
