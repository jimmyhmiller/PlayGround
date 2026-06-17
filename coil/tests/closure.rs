//! Function pointers, indirect calls, and closures with manual memory.

mod common;
use common::build_and_run;

#[test]
fn function_pointer_indirect_call() {
    // Take a function's address and call through it.
    let src = r#"
        (defn add [(a :i64) (b :i64)] (-> :i64) (iadd a b))
        (defn main [] (-> :i64)
          (let [f (fnptr-of add)] (call-ptr f 40 2)))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn function_pointer_through_native_convention() {
    // fnptr-of / call-ptr honor a non-default native convention.
    let src = r#"
        (defcc fast2 :params [rax rdx] :ret rax :clobber [rax rdx rcx] :native fast)
        (defn add :cc fast2 [(a :i64) (b :i64)] (-> :i64) (iadd a b))
        (defn main [] (-> :i64) (call-ptr (fnptr-of add) 20 22))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn dispatch_table() {
    // Pick a function pointer out of an array and call it — a tiny vtable.
    let src = r#"
        (defn inc [(x :i64)] (-> :i64) (iadd x 1))
        (defn dec [(x :i64)] (-> :i64) (isub x 1))
        (defn main [] (-> :i64)
          (let [t (alloc frame (array (fnptr c [:i64] :i64) 2))]
            (store! (index t 0) (fnptr-of inc))
            (store! (index t 1) (fnptr-of dec))
            (iadd (call-ptr (load (index t 0)) 41)     ; inc 41 = 42
                  (call-ptr (load (index t 1)) 1))))    ; dec 1  = 0
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn heap_closures_heterogeneous_with_manual_free() {
    // Two closures with different captured environments, one Closure type, one
    // generic apply, allocated and freed by hand.
    assert_eq!(build_and_run(include_str!("../examples/closure.coil")), 42);
}

#[test]
fn closure_escapes_and_is_called_later() {
    // make-counter returns a closure (upward funarg); main calls it after.
    let src = r#"
        (defstruct Env [(base :i64)])
        (defn code [(e (ptr heap Env)) (x :i64)] (-> :i64)
          (iadd (load (field e base)) x))
        (defstruct Fn1 [(code (fnptr c [(ptr heap Env) :i64] :i64)) (env (ptr heap Env))])
        (defn make [(b :i64)] (-> (ptr heap Fn1))
          (let [e (alloc heap Env) c (alloc heap Fn1)]
            (store! (field e base) b)
            (store! (field c code) (fnptr-of code))
            (store! (field c env) e)
            c))
        (defn main [] (-> :i64)
          (let [f (make 40)]
            (let [r (call-ptr (load (field f code)) (load (field f env)) 2)]
              (free (load (field f env))) (free f)
              r)))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn rejects_fnptr_to_shim_convention() {
    let src = r#"
        (defcc reg2 :params [rax rdx] :ret rax :clobber [rax rdx] :lower shim)
        (defn f :cc reg2 [(a :i64) (b :i64)] (-> :i64) (iadd a b))
        (defn main [] (-> :i64) (call-ptr (fnptr-of f) 1 2))
    "#;
    assert!(coil::check_source(src).unwrap_err().contains("shim-convention"));
}

#[test]
fn rejects_callptr_wrong_arg_type() {
    let src = r#"
        (defn add [(a :i64) (b :i64)] (-> :i64) (iadd a b))
        (defn main [] (-> :i64)
          (let [p (alloc heap)] (call-ptr (fnptr-of add) p 2)))
    "#;
    assert!(coil::check_source(src).unwrap_err().contains("call-ptr argument 1"));
}
