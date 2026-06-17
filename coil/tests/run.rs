//! Core language, end to end via AOT (build native exe, check exit code).
//! Rejections are checked with the front-end-only `check_source`.

mod common;
use common::build_and_run;

#[test]
fn arithmetic_and_let() {
    let src = r#"
        (defn main [] (-> :i64)
          (let [x 20 y 22] (iadd x y)))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn recursion_and_if() {
    let src = r#"
        (defn fib [(n :i64)] (-> :i64)
          (if (icmp-le n 1) n (iadd (fib (isub n 1)) (fib (isub n 2)))))
        (defn main [] (-> :i64) (fib 10))
    "#;
    assert_eq!(build_and_run(src), 55);
}

#[test]
fn custom_convention_runs_and_emits_fastcc() {
    let src = r#"
        (defcc fast2 :params [rax rdx] :ret rax
          :clobber [rax rdx rcx] :preserve [rbx rbp] :native fast)
        (defn add :cc fast2 [(a :i64) (b :i64)] (-> :i64) (iadd a b))
        (defn main [] (-> :i64) (add 20 22))
    "#;
    assert_eq!(build_and_run(src), 42);
    let ir = coil::emit_ir(src).unwrap();
    assert!(ir.contains("fastcc"), "expected fastcc in IR:\n{ir}");
}

#[test]
fn shim_convention_with_exotic_registers() {
    let src = r#"
        (defcc reg2 :params [rax rdx] :ret rax
          :clobber [rax rdx rcx rsi rdi r8 r9 r10 r11] :lower shim)
        (defn sub2 :cc reg2 [(a :i64) (b :i64)] (-> :i64) (isub a b))
        (defn main [] (-> :i64)
          (let [x (iadd 40 10)] (sub2 x 8)))
    "#;
    assert_eq!(build_and_run(src), 42);
    let ir = coil::emit_ir(src).unwrap();
    assert!(ir.contains("naked"), "expected a naked trampoline:\n{ir}");
    assert!(ir.contains("__impl"), "expected a ccc impl:\n{ir}");
}

#[test]
fn shim_convention_recurses() {
    let src = r#"
        (defcc reg2 :params [rax rdx] :ret rax
          :clobber [rax rdx rcx rsi rdi r8 r9 r10 r11] :lower shim)
        (defn fact :cc reg2 [(n :i64) (acc :i64)] (-> :i64)
          (if (icmp-le n 1) acc (fact (isub n 1) (imul n acc))))
        (defn main [] (-> :i64) (fact 5 1))
    "#;
    assert_eq!(build_and_run(src), 120);
}

#[test]
fn heap_alloc_store_load_free() {
    let src = r#"
        (defn main [] (-> :i64)
          (let [p (alloc heap)]
            (store! p 42)
            (let [v (load p)] (free p) v)))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn static_and_frame_regions() {
    let static_src = r#"
        (defn main [] (-> :i64)
          (let [g (alloc static)] (store! g 99) (load g)))
    "#;
    assert_eq!(build_and_run(static_src), 99);

    let frame_src = r#"
        (defn main [] (-> :i64)
          (let [p (alloc frame)] (store! p 7) (iadd (load p) 35)))
    "#;
    assert_eq!(build_and_run(frame_src), 42);
}

#[test]
fn heap_pointer_crosses_function_boundary() {
    let src = r#"
        (defn make [(v :i64)] (-> (ptr heap))
          (let [p (alloc heap)] (store! p v) p))
        (defn main [] (-> :i64)
          (let [p (make 42)]
            (let [v (load p)] (free p) v)))
    "#;
    assert_eq!(build_and_run(src), 42);
}

// ---- rejections (front-end only) ---------------------------------------

#[test]
fn rejects_escaping_frame_pointer() {
    let src = r#"
        (defn bad [] (-> (ptr frame)) (alloc frame))
        (defn main [] (-> :i64) 0)
    "#;
    assert!(coil::check_source(src).unwrap_err().contains("frame pointer"));
}

#[test]
fn rejects_freeing_non_heap() {
    let src = r#"
        (defn main [] (-> :i64)
          (let [g (alloc static)] (free g)))
    "#;
    assert!(coil::check_source(src).unwrap_err().contains("cannot free"));
}

#[test]
fn rejects_loading_non_pointer() {
    let src = "(defn main [] (-> :i64) (load 5))";
    assert!(coil::check_source(src)
        .unwrap_err()
        .contains("load expects a pointer"));
}

#[test]
fn rejects_arity_mismatch() {
    let src = r#"
        (defn add [(a :i64) (b :i64)] (-> :i64) (iadd a b))
        (defn main [] (-> :i64) (add 1))
    "#;
    assert!(coil::check_source(src).unwrap_err().contains("expects 2 args"));
}

#[test]
fn rejects_shim_without_ret() {
    let src = r#"
        (defcc bad :params [rax rdx] :lower shim)
        (defn add :cc bad [(a :i64) (b :i64)] (-> :i64) (iadd a b))
        (defn main [] (-> :i64) (add 1 2))
    "#;
    assert!(coil::check_source(src).unwrap_err().contains(":ret"));
}

#[test]
fn rejects_unbound_variable() {
    let src = "(defn main [] (-> :i64) (iadd x 1))";
    assert!(coil::check_source(src)
        .unwrap_err()
        .contains("unbound variable"));
}
