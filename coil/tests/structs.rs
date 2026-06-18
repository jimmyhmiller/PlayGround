//! Structs and arrays, end to end via AOT.

mod common;
use common::build_and_run;

#[test]
fn struct_and_array_on_the_stack() {
    assert_eq!(build_and_run(include_str!("../examples/structs.coil")), 42);
}

#[test]
fn struct_on_the_heap() {
    let src = r#"
        (defstruct Pair [(a :i64) (b :i64)])
        (defn main [] (-> :i64)
          (let [p (alloc-heap Pair)]
            (store! (field p a) 19)
            (store! (field p b) 23)
            (let [s (iadd (load (field p a)) (load (field p b)))]
              (free p)
              s)))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn array_walk_with_recursion() {
    // Sum a[0..n) by recursing over an index. The array is on the heap because a
    // `frame` pointer can't cross a function boundary (escape rule).
    let src = r#"
        (defn sum [(a (ptr (array i64 4))) (i :i64) (n :i64)] (-> :i64)
          (if (icmp-ge i n) 0
              (iadd (load (index a i)) (sum a (iadd i 1) n))))
        (defn main [] (-> :i64)
          (let [a (alloc-heap (array i64 4))]
            (store! (index a 0) 10) (store! (index a 1) 11)
            (store! (index a 2) 9)  (store! (index a 3) 12)
            (let [s (sum a 0 4)] (free a) s)))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn nested_struct_by_value_field() {
    // A struct whose field is another (earlier-defined) struct, by value.
    let src = r#"
        (defstruct Inner [(v :i64)])
        (defstruct Outer [(lo Inner) (hi Inner)])
        (defn main [] (-> :i64)
          (let [o (alloc-stack Outer)]
            (store! (field (field o lo) v) 40)
            (store! (field (field o hi) v) 2)
            (iadd (load (field (field o lo) v)) (load (field (field o hi) v)))))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn struct_returned_through_a_heap_pointer() {
    let src = r#"
        (defstruct Box [(v :i64)])
        (defn make [(v :i64)] (-> (ptr Box))
          (let [b (alloc-heap Box)] (store! (field b v) v) b))
        (defn main [] (-> :i64)
          (let [b (make 42)]
            (let [v (load (field b v))] (free b) v)))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn rejects_unknown_struct() {
    let src = "(defn main [] (-> :i64) (let [p (alloc-heap Nope)] 0))";
    assert!(coil::check_source(src).unwrap_err().contains("unknown struct 'Nope'"));
}

#[test]
fn rejects_unknown_field() {
    let src = r#"
        (defstruct P [(x :i64)])
        (defn main [] (-> :i64)
          (let [p (alloc-stack P)] (load (field p y))))
    "#;
    assert!(coil::check_source(src).unwrap_err().contains("no field 'y'"));
}

#[test]
fn rejects_field_on_non_struct() {
    let src = "(defn main [] (-> :i64) (let [p (alloc-heap i64)] (load (field p x))))";
    assert!(coil::check_source(src).unwrap_err().contains("field access"));
}
