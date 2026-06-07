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
