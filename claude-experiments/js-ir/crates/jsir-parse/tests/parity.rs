//! Differential oracle: the no-AST `text → tokens → JSIR` path must produce the
//! exact same IR text as the trusted swc-based converter (`source_to_ir`), which
//! is itself byte-exact vs upstream's golden `jshir.mlir`.

/// In-subset snippets. Each must lower identically via both paths.
const SNIPPETS: &[&str] = &[
    "1 + 2 + 3;",
    "1 * 2 + 3;",
    "1 + 2 * 3;",
    "a = 3;",
    "a;",
    "-a;",
    "a++;",
    "++a;",
    "!a;",
    "~a;",
    "a == b;",
    "a === b;",
    "a != b && c;", // && is excluded → this should error in our parser; kept out below
    "a < b;",
    "a & b | c ^ d;",
    "a << 2;",
    "x = y = 1;",
    "a += 1;",
    "x = 'hello';", // string in expression position (a *leading* string is a directive — out of subset)
    "true;",
    "false;",
    "null;",
    "1n;",
    "let a = 0;",
    "var b = 1;",
    "const c = 2, d = 3;",
    "let x = 1 + 2 * 3;",
    "a = b + c;",
    "foo = bar * 2 - 1;",
];

fn check(src: &str) {
    let ours = jsir_parse::parse_to_ir_text(src)
        .unwrap_or_else(|e| panic!("our parser failed on {src:?}: {e}"));
    let theirs = jsir_swc::source_to_ir(src)
        .map(|op| op.print())
        .unwrap_or_else(|e| panic!("source_to_ir failed on {src:?}: {e}"));
    if ours != theirs {
        if let Some(diff) = jsir_oracle::byte_diff(&theirs, &ours) {
            panic!("DIVERGED on {src:?}:\n{diff}\n--- ours ---\n{ours}\n--- theirs ---\n{theirs}");
        }
    }
}

#[test]
fn snippets_match_converter() {
    for src in SNIPPETS {
        // skip the logical-op snippet (out of subset; documented)
        if src.contains("&&") || src.contains("||") || src.contains("??") {
            continue;
        }
        check(src);
    }
}

/// Drive every in-subset fixture's real `input.js` through both paths.
#[test]
fn in_subset_fixtures_match() {
    let in_subset = ["binary_expression", "load_store_identifier", "unary_expression",
        "update_expression", "variable_declaration"];
    for f in jsir_oracle::list_fixtures() {
        if !in_subset.contains(&f.name.as_str()) {
            continue;
        }
        let src = f.input_js().expect("input.js");
        let ours = match jsir_parse::parse_to_ir_text(&src) {
            Ok(s) => s,
            Err(e) => panic!("{}: our parser failed: {e}", f.name),
        };
        let theirs = jsir_swc::source_to_ir(&src).expect("source_to_ir").print();
        assert_eq!(ours, theirs, "fixture {} diverged", f.name);
        // And our output is itself byte-exact vs the golden jshir.
        if let Some(expected) = f.expected_jshir() {
            assert!(
                jsir_oracle::filecheck_equivalent(&expected, &ours),
                "fixture {} not byte-exact vs golden",
                f.name
            );
        }
    }
}

#[test]
fn dce_removes_dead_vars() {
    let src = "var dead = 1 + 2; var live = 3; live = live + 1;";
    let m = jsir_parse::parse_to_module(src).unwrap();
    let (out, removed) = jsir_ir::build::dce(&m);
    let txt = out.print();
    assert_eq!(removed, 1, "should remove exactly `dead`");
    assert!(!txt.contains("\"dead\""), "dead var still present:\n{txt}");
    assert!(txt.contains("\"live\""), "live var was wrongly removed:\n{txt}");
}

#[test]
fn dce_in_place_matches_copying() {
    let src = "var dead = 1 + 2; var live = 3; live = live + 1; var d2 = 9;";
    let copying = {
        let m = jsir_parse::parse_to_module(src).unwrap();
        jsir_ir::build::dce(&m).0.print()
    };
    let inplace = {
        let mut m = jsir_parse::parse_to_module(src).unwrap();
        let removed = jsir_ir::build::dce_in_place(&mut m);
        assert_eq!(removed, 2, "dead + d2");
        m.print()
    };
    assert_eq!(copying, inplace, "in-place DCE output must equal copying DCE");
    assert!(!inplace.contains("\"dead\"") && !inplace.contains("\"d2\""));
    assert!(inplace.contains("\"live\""));
}

#[test]
fn tape_is_rpn_balanced() {
    use jsir_parse::tk;
    // statements are roots; every node consumes `nargs` and produces 1, so the
    // stack ends at the number of top-level statements and never goes negative.
    let cases = [
        ("var a = 1 + 2 * 3; a = a + 1;", 2),
        ("-x; y++; ++z; var p = 5, q = 6;", 4),
        ("a; b; c;", 3),
    ];
    for (src, stmts) in cases {
        let tape = jsir_parse::parse_to_tape(src).unwrap();
        let mut stack: i64 = 0;
        let mut roots = 0;
        for n in &tape {
            stack -= n.nargs as i64;
            assert!(stack >= 0, "RPN underflow in {src:?} at {n:?}");
            stack += 1;
            if matches!(n.kind, tk::EXPR_STMT | tk::VAR_DECL) {
                roots += 1;
            }
        }
        assert_eq!(stack, stmts, "root count for {src:?}");
        assert_eq!(roots, stmts, "statement node count for {src:?}");
    }
}

/// The js-ir DCE eliminates the *same set* oxc's `Minifier::dce` does.
#[test]
fn dce_same_elimination_as_oxc() {
    use std::collections::BTreeSet;
    use jsir_ir::{IrRead, OpId};
    use oxc_allocator::Allocator;
    use oxc_ast::ast::{BindingPatternKind, Statement};
    use oxc_minifier::{CompressOptions, Minifier, MinifierOptions};
    use oxc_parser::Parser;
    use oxc_span::SourceType;

    // dead/d2: declared, never read → removable. live/keep: read → kept.
    let src = "var dead = 1 + 2; var live = 3; live = live + 1; var d2 = 7; var keep = 0; keep = keep + 1;";

    // ours
    let m = jsir_parse::parse_to_module(src).unwrap();
    let (out, _) = jsir_ir::build::dce(&m);
    let mut ours = BTreeSet::new();
    for i in 0..out.op_count() {
        let op = OpId(i as u32);
        if out.op_name(op) == "jsir.variable_declaration" {
            for r in out.regions(op) {
                for b in out.region_blocks(*r) {
                    for &o in out.block_ops(*b) {
                        if out.op_name(o) == "jsir.identifier_ref" {
                            if let Some(n) = out.str_attr(o, "name") {
                                ours.insert(n.to_string());
                            }
                        }
                    }
                }
            }
        }
    }
    // oxc
    let alloc = Allocator::default();
    let mut ret = Parser::new(&alloc, src, SourceType::default()).parse();
    let opts = MinifierOptions { mangle: None, compress: Some(CompressOptions::default()) };
    Minifier::new(opts).dce(&alloc, &mut ret.program);
    let mut oxc = BTreeSet::new();
    for stmt in &ret.program.body {
        if let Statement::VariableDeclaration(d) = stmt {
            for decl in &d.declarations {
                if let BindingPatternKind::BindingIdentifier(bi) = &decl.id.kind {
                    oxc.insert(bi.name.to_string());
                }
            }
        }
    }
    assert_eq!(ours, oxc, "ours kept {ours:?}, oxc kept {oxc:?}");
    assert!(ours.contains("live") && ours.contains("keep"));
    assert!(!ours.contains("dead") && !ours.contains("d2"));
}
