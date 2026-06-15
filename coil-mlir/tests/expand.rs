//! Tests for the bootstrap expander (surface sugar → core forms), and a
//! read → expand → emit integration check against the RecordingBackend.

use coil::emit::Emitter;
use coil::expand::expand_all;
use coil::printer::print;
use coil::reader::read_all;
use coil::recording::RecordingBackend;

/// Read, expand, and print the (single) resulting core form.
fn expand1(src: &str) -> String {
    let forms = read_all(src).expect("read");
    let core = expand_all(&forms).expect("expand");
    assert_eq!(core.len(), 1, "expected one top-level form");
    print(&core[0])
}

fn expand_err(src: &str) -> String {
    let forms = read_all(src).expect("read");
    expand_all(&forms).unwrap_err().0
}

#[test]
fn defn_becomes_func_func() {
    let out = expand1("(defn add [(: a i32) (: b i32)] -> i32 (arith.addi a b))");
    assert!(out.starts_with("(op \"func.func\""), "{out}");
    assert!(out.contains(":sym_name \"add\""), "{out}");
    // signature uses fn-type, never -> (which is the threading macro)
    assert!(out.contains("(fn-type [i32 i32] [i32])"), "{out}");
    assert!(!out.contains("(-> ["), "{out}");
    // implicit return wraps the last expression
    assert!(out.contains("(func.return (arith.addi a b))"), "{out}");
    // a region with a block carrying the params
    assert!(out.contains(":regions [(region (block ^entry"), "{out}");
    assert!(out.contains("[(: a i32) (: b i32)]"), "{out}");
}

#[test]
fn void_defn_has_no_implicit_return() {
    let out = expand1("(defn store! [(: p !llvm.ptr)] (llvm.store x p))");
    assert!(!out.contains("func.return"), "{out}");
    assert!(out.contains("(fn-type [!llvm.ptr] [])"), "{out}");
}

#[test]
fn value_if_uses_scf_if_with_yields() {
    let out = expand1("(if {:result i32} c x y)");
    assert!(out.starts_with("(scf.if {:results [i32]} c"), "{out}");
    assert!(out.contains("(region (scf.yield x))"), "{out}");
    assert!(out.contains("(region (scf.yield y))"), "{out}");
}

#[test]
fn statement_if_yields_nothing() {
    let out = expand1("(if c (foo.bar))");
    assert!(out.starts_with("(scf.if c (region (foo.bar) (scf.yield)))"), "{out}");
}

#[test]
fn value_if_without_else_is_error() {
    let e = expand_err("(if {:result i32} c x)");
    assert!(e.contains("needs an else"), "{e}");
}

#[test]
fn when_is_single_region_scf_if() {
    let out = expand1("(when c (a.b) (c.d))");
    assert!(out.starts_with("(scf.if c (region (a.b) (c.d) (scf.yield)))"), "{out}");
}

#[test]
fn cond_nests_ifs() {
    // statement cond → nested scf.if (after the inner `if`s expand)
    let out = expand1("(cond c1 (a.x) c2 (b.y) :else (c.z))");
    // outermost is c1's scf.if; the else branch contains c2's scf.if
    assert!(out.starts_with("(scf.if c1"), "{out}");
    assert!(out.contains("c2"), "{out}");
    assert!(out.contains("(c.z)"), "{out}");
}

#[test]
fn thread_first_rewrites_nesting() {
    let out = expand1("(-> x (f a) (g) h)");
    assert_eq!(out, "(h (g (f x a)))");
}

#[test]
fn typed_const_lowers_to_arith_constant() {
    // read → expand → emit; (: 42 i32) must become an arith.constant op
    let forms = read_all("(defn k [] -> i32 (: 42 i32))").unwrap();
    let core = expand_all(&forms).unwrap();
    let mut be = RecordingBackend::new();
    {
        let mut em = Emitter::new(&mut be);
        em.emit_module(&core).expect("emit");
    }
    let log = be.log_text();
    assert!(log.contains("op arith.constant"), "{log}");
    assert!(log.contains("value=42"), "{log}");
    assert!(log.contains("op func.return"), "{log}");
    assert!(log.contains("op func.func"), "{log}");
}

#[test]
fn nested_sugar_inside_defn_expands() {
    // an `if` inside a defn body must also expand to scf.if
    let out = expand1(
        "(defn pick [(: c i1)] -> i32 (if {:result i32} c (: 1 i32) (: 0 i32)))",
    );
    assert!(out.contains("(func.return (scf.if {:results [i32]} c"), "{out}");
}
