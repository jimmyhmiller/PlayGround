//! `for-in` — iterate a collection binding each element, a macro over `for`/`loop`
//! (lib/control.coil). Dispatches syntactically on the collection form:
//! `(range lo hi)`, `(over arr len)` for fixed arrays, and `(iter next state)` —
//! the EXTENSIBLE protocol: any user type becomes iterable by supplying a
//! `next: (ptr S) -> (Option T)`. `(slice T)` collections are the follow-on (§3.4).

mod common;
use common::build_and_run;

const IMPORT: &str = "(module app)\n(import \"lib/control.coil\" :use *)\n";

fn run_with(body: &str) -> i32 {
    build_and_run(&format!("{IMPORT}{body}"))
}

#[test]
fn for_in_range_sums() {
    // 0+1+2+3+4 = 10.
    let code = run_with(
        r#"(defn main [] (-> :i64)
             (let [(mut acc) 0]
               (for-in [x (range 0 5)] (store! acc (iadd (load acc) x)))
               (load acc)))"#,
    );
    assert_eq!(code, 10);
}

#[test]
fn for_in_array_sums() {
    // Iterate a fixed array's elements: 10+20+30 = 60.
    let code = run_with(
        r#"(defn main [] (-> :i64)
             (let [arr (alloc-stack (array i64 3))]
               (store! (index arr 0) 10)
               (store! (index arr 1) 20)
               (store! (index arr 2) 30)
               (let [(mut acc) 0]
                 (for-in [x (over arr 3)] (store! acc (iadd (load acc) x)))
                 (load acc))))"#,
    );
    assert_eq!(code, 60);
}

#[test]
fn for_in_supports_break_and_continue() {
    // break/continue work inside for-in (it lowers to `for` over the loop
    // primitive): sum even elements until we hit 99 → 2+4 = 6.
    let code = run_with(
        r#"(defn main [] (-> :i64)
             (let [arr (alloc-stack (array i64 5))]
               (store! (index arr 0) 2)
               (store! (index arr 1) 99)
               (store! (index arr 2) 4)
               (store! (index arr 3) 99)
               (store! (index arr 4) 7)
               (let [(mut acc) 0]
                 (for-in [x (over arr 5)]
                   (if (icmp-eq x 99) (break) 0)
                   (store! acc (iadd (load acc) x)))
                 (load acc))))"#,
    );
    assert_eq!(code, 2); // first element 2, then hits 99 → break
}

#[test]
fn for_in_extensible_iter_protocol() {
    // A user collection becomes iterable by supplying next: (ptr S) -> (Option T).
    // Drives entirely through the loop primitive + call-ptr; no core support.
    let src = r#"(module app)
(import "lib/control.coil" :use *)
(import "lib/result.coil" :use *)
(defstruct RangeState [(cur i64) (hi i64)])
(defn rnext [(s (ptr RangeState))] (-> (Option i64))
  (if (icmp-ge (load (field s cur)) (load (field s hi))) (None)
    (let [v (load (field s cur))] (store! (field s cur) (iadd v 1)) (Some v))))
(defn main [] (-> :i64)
  (let [st (alloc-stack RangeState)]
    (store! (field st cur) 0) (store! (field st hi) 5)
    (let [(mut acc) 0]
      (for-in [x (iter (fnptr-of rnext) st)] (store! acc (iadd (load acc) x)))
      (load acc))))"#;
    assert_eq!(build_and_run(src), 10); // 0+1+2+3+4
}

#[test]
fn for_in_iter_multi_statement_body() {
    // Regression for the match-arm truncation: a for-in (iter) body with MORE
    // THAN ONE statement. The body lowers to (Some [x] stmt1 stmt2 …); if the
    // parser drops all but stmt1, the second statement silently vanishes (and a
    // body whose tail advances/terminates would infinite-loop). Both must run.
    let src = r#"(module app)
(import "lib/control.coil" :use *)
(import "lib/result.coil" :use *)
(defstruct RS [(cur i64) (hi i64)])
(defn rn [(s (ptr RS))] (-> (Option i64))
  (if (icmp-ge (load (field s cur)) (load (field s hi))) (None)
    (let [v (load (field s cur))] (store! (field s cur) (iadd v 1)) (Some v))))
(defn main [] (-> :i64)
  (let [st (alloc-stack RS) (mut cnt) 0 (mut acc) 0]
    (store! (field st cur) 0) (store! (field st hi) 5)
    (for-in [x (iter (fnptr-of rn) st)]
      (store! cnt (iadd (load cnt) 1))     ; stmt1
      (store! acc (iadd (load acc) x)))    ; stmt2 — must NOT be dropped
    (iadd (imul (load acc) 10) (load cnt))))"#;
    // acc = 0+1+2+3+4 = 10, cnt = 5 → 10*10 + 5 = 105. (Pre-fix: stmt2 dropped ⇒ 5.)
    assert_eq!(build_and_run(src), 105);
}

#[test]
fn for_in_bad_binding_hard_errors() {
    let err = coil::check_source(&format!(
        "{IMPORT}(defn main [] (-> :i64) (for-in [x] 1))"
    ))
    .unwrap_err();
    assert!(err.contains("binding must be [x COLL]"), "got: {err}");
}

#[test]
fn for_in_unknown_collection_hard_errors() {
    let err = coil::check_source(&format!(
        "{IMPORT}(defn main [] (-> :i64) (for-in [x (frobnicate 1 2)] 1))"
    ))
    .unwrap_err();
    assert!(err.contains("unsupported collection"), "got: {err}");
}
