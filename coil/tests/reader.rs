//! Reader tests, including round-trip (read → print → read yields equal Val)
//! and the SPEC §1 surface (maps, keywords, type/attr literals, comments,
//! reader macros).

use coil::value::Val;
use coil::{print, read_all, read_one};
use std::rc::Rc;

fn roundtrip(src: &str) {
    let a = read_all(src).expect("first read");
    let printed: String = a.iter().map(|v| print(v) + "\n").collect();
    let b = read_all(&printed).expect("re-read of printed output");
    assert_eq!(a, b, "round-trip mismatch for {src:?}\nprinted: {printed}");
}

#[test]
fn atoms() {
    assert_eq!(read_one("42").unwrap(), Val::Int(42));
    assert_eq!(read_one("-7").unwrap(), Val::Int(-7));
    assert_eq!(read_one("0xFF").unwrap(), Val::Int(255));
    assert_eq!(read_one("3.14").unwrap(), Val::Float(3.14));
    assert_eq!(read_one("1e9").unwrap(), Val::Float(1e9));
    assert_eq!(read_one("true").unwrap(), Val::Bool(true));
    assert_eq!(read_one("false").unwrap(), Val::Bool(false));
    assert_eq!(read_one("nil").unwrap(), Val::Nil);
    assert_eq!(read_one(":value").unwrap(), Val::keyword("value"));
}

#[test]
fn symbols_keep_sigils_and_dots() {
    for s in ["foo", "arith.addi", "my/helper", "+", "->", "@printf", "^bb1", "%0", "<=i"] {
        assert_eq!(read_one(s).unwrap(), Val::sym(s), "symbol {s}");
    }
}

#[test]
fn strings_with_escapes() {
    assert_eq!(read_one(r#""hi""#).unwrap(), Val::str("hi"));
    assert_eq!(read_one(r#""a\nb""#).unwrap(), Val::Str(Rc::from("a\nb")));
    assert_eq!(read_one(r#""null\0term""#).unwrap(), Val::Str(Rc::from("null\0term")));
}

#[test]
fn collections() {
    assert_eq!(
        read_one("(a b c)").unwrap(),
        Val::list(vec![Val::sym("a"), Val::sym("b"), Val::sym("c")])
    );
    assert_eq!(
        read_one("[1 2]").unwrap(),
        Val::vector(vec![Val::Int(1), Val::Int(2)])
    );
    assert_eq!(
        read_one("{:a 1 :b 2}").unwrap(),
        Val::map(vec![
            (Val::keyword("a"), Val::Int(1)),
            (Val::keyword("b"), Val::Int(2)),
        ])
    );
}

#[test]
fn odd_map_is_error() {
    assert!(read_one("{:a}").is_err());
}

#[test]
fn type_and_attr_literals() {
    assert_eq!(read_one("!llvm.ptr").unwrap(), Val::TypeLit(Rc::from("llvm.ptr")));
    // balanced angle/paren so inner spaces don't terminate the literal
    assert_eq!(
        read_one("!llvm.struct<(i64, i64)>").unwrap(),
        Val::TypeLit(Rc::from("llvm.struct<(i64, i64)>"))
    );
    assert_eq!(
        read_one("#llvm.linkage<internal>").unwrap(),
        Val::AttrLit(Rc::from("llvm.linkage<internal>"))
    );
    // bare `#` (followed by space) is the symbol `#`, not an attr literal
    assert_eq!(
        read_one("(# slt)").unwrap(),
        Val::list(vec![Val::sym("#"), Val::sym("slt")])
    );
}

#[test]
fn reader_macros_desugar() {
    assert_eq!(
        read_one("'x").unwrap(),
        Val::list(vec![Val::sym("quote"), Val::sym("x")])
    );
    assert_eq!(
        read_one("`(a ~b ~@c)").unwrap(),
        Val::list(vec![
            Val::sym("quasiquote"),
            Val::list(vec![
                Val::sym("a"),
                Val::list(vec![Val::sym("unquote"), Val::sym("b")]),
                Val::list(vec![Val::sym("unquote-splicing"), Val::sym("c")]),
            ]),
        ])
    );
}

#[test]
fn comments_are_skipped() {
    let forms = read_all(
        "; line comment\n  (a #_ ignored b) #| block |# c",
    )
    .unwrap();
    assert_eq!(
        forms,
        vec![
            Val::list(vec![Val::sym("a"), Val::sym("b")]),
            Val::sym("c"),
        ]
    );
}

#[test]
fn unterminated_is_error() {
    assert!(read_all("(a b").is_err());
    assert!(read_all("\"oops").is_err());
    assert!(read_all("#| open").is_err());
}

#[test]
fn roundtrips() {
    roundtrip("(defn add [(: a i32) (: b i32)] -> i32 (func.return (arith.addi a b)))");
    roundtrip("(arith.constant {:value (: 42 i64)})");
    roundtrip("!llvm.struct<(i64, i64)>");
    roundtrip("`(let [t# ~x] (if t# t# false))");
    roundtrip("{:sym_name \"main\" :function_type (-> [] [i32])}");
    roundtrip("(scf.if {:results [i32]} c (region (scf.yield n)))");
}
