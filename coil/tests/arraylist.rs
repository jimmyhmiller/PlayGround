//! `ArrayList T` — a growable array over slices + the allocator-as-value interface
//! (lib/arraylist.coil). An ordinary generic struct + functions/macros; no core
//! support, no ambient heap (it carries its `(ptr Allocator)`).

mod common;
use common::build_and_run;

// Importing the modules whose names a test references directly: alloc
// (malloc-allocator/arena), control (for), slice (slice-get/len on al-slice),
// result (Some/None when matching al-pop!). `:use *` doesn't re-export transitively.
const H: &str = concat!(
    "(module app)\n",
    "(import \"lib/arraylist.coil\" :use *)\n",
    "(import \"lib/alloc.coil\" :use *)\n",
    "(import \"lib/control.coil\" :use *)\n",
    "(import \"lib/slice.coil\" :use *)\n",
    "(import \"lib/result.coil\" :use *)\n",
);

fn run(body: &str) -> i32 {
    build_and_run(&format!("{H}{body}"))
}

#[test]
fn push_grow_get_len_and_for() {
    // Push 0..9 (grows past the initial cap 4), set element 0, iterate.
    let code = run(
        r#"(defn main [] (-> :i64)
             (let [a (malloc-allocator) (mut xs) (al-new [i64] a)]
               (for [i 0 10] (al-push! (mut xs) i))
               (al-set! (mut xs) 0 100)
               (let [(mut acc) 0]
                 (al-for [v xs] (store! acc (iadd (load acc) v)))
                 (let [r (iadd (load acc) (al-len xs))]   ; 145 + len 10 = 155
                   (al-free! (mut xs))
                   r))))"#,
    );
    assert_eq!(code, 155);
}

#[test]
fn pop_returns_some_then_none() {
    // Pop the only element (Some 7), then pop empty (None) → distinguish.
    let code = run(
        r#"(defn main [] (-> :i64)
             (let [a (malloc-allocator) (mut xs) (al-new [i64] a)]
               (al-push! (mut xs) 7)
               (let [first (al-pop! (mut xs)) second (al-pop! (mut xs))]
                 (match first
                   (None [] 0)
                   (Some [v]
                     (match second (None [] (iadd v 100)) (Some [_] 0)))))))"#,
    );
    assert_eq!(code, 107); // Some 7 then None → 7 + 100
}

#[test]
fn empty_predicate_and_len() {
    let code = run(
        r#"(defn main [] (-> :i64)
             (let [a (malloc-allocator) (mut xs) (al-new [i64] a)]
               (if (al-empty? xs)
                   (do (al-push! (mut xs) 5) (al-push! (mut xs) 6)
                       (if (al-empty? xs) 0 (al-len xs)))
                   99)))"#,
    );
    assert_eq!(code, 2);
}

#[test]
fn al_slice_views_the_elements() {
    // al-slice integrates with the slice library.
    let code = run(
        r#"(defn main [] (-> :i64)
             (let [a (malloc-allocator) (mut xs) (al-new [i64] a)]
               (al-push! (mut xs) 4) (al-push! (mut xs) 5) (al-push! (mut xs) 6)
               (let [s (al-slice xs)]
                 (iadd (slice-get s 1) (slice-len s)))))"#, // 5 + 3
    );
    assert_eq!(code, 8);
}

#[test]
fn grows_with_a_non_resizing_arena_allocator() {
    // Arena never resizes → exercises al-reserve!'s alloc-new + copy + free path.
    let code = run(
        r#"(defn main [] (-> :i64)
             (let [a (arena-allocator 4096) (mut xs) (al-new [i64] a)]
               (for [i 0 10] (al-push! (mut xs) i))
               (let [(mut acc) 0]
                 (al-for [v xs] (store! acc (iadd (load acc) v)))
                 (load acc))))"#, // 0+1+..+9 = 45
    );
    assert_eq!(code, 45);
}

#[test]
fn al_for_bad_binding_hard_errors() {
    let err =
        coil::check_source(&format!("{H}(defn main [] (-> :i64) (al-for [x] 1))")).unwrap_err();
    assert!(err.contains("binding must be [x arraylist]"), "got: {err}");
}
