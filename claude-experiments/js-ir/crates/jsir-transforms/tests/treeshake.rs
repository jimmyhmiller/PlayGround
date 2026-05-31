//! Cross-module tree-shaking tests.

use std::collections::BTreeMap;

use jsir_transforms::tree_shake;

fn project(files: &[(&str, &str)]) -> BTreeMap<String, String> {
    files.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

#[test]
fn drops_unreachable_module_and_unused_export() {
    let p = project(&[
        ("app.js", "import { add } from './math.js';\nglobalThis.r = add(1, 2);\n"),
        (
            "math.js",
            "export function add(a, b) { return a + b; }\nexport function sub(a, b) { return a - b; }\n",
        ),
        ("orphan.js", "export function nobody() { return 99; }\n"),
    ]);

    let res = tree_shake(&p, "app.js").expect("tree shake");

    // orphan.js is imported by nothing -> dropped entirely.
    assert!(!res.modules.contains_key("orphan.js"), "orphan dropped");
    assert_eq!(res.stats.modules_dropped, 1);

    // math.js keeps `add` (used) and drops `sub` (dead export).
    let math = &res.modules["math.js"];
    assert!(math.contains("function add"), "kept used export: {math}");
    assert!(!math.contains("sub"), "dropped dead export: {math}");
    assert!(res.stats.dead_exports_removed >= 1);

    // app.js keeps the import it uses.
    assert!(res.modules["app.js"].contains("add"));
}

#[test]
fn keeps_transitively_used_exports() {
    let p = project(&[
        ("main.js", "import { a } from './lib.js';\nglobalThis.x = a();\n"),
        (
            "lib.js",
            "function helper() { return 1; }\nexport function a() { return helper(); }\nexport function b() { return 2; }\n",
        ),
    ]);
    let res = tree_shake(&p, "main.js").expect("shake");
    let lib = &res.modules["lib.js"];
    assert!(lib.contains("function a"), "kept used export: {lib}");
    assert!(lib.contains("function helper"), "kept transitive dep: {lib}");
    assert!(!lib.contains("function b"), "dropped unused export: {lib}");
}

#[test]
fn drops_unused_imports() {
    let p = project(&[
        ("e.js", "import { used, unused } from './m.js';\nglobalThis.v = used;\n"),
        ("m.js", "export const used = 1;\nexport const unused = 2;\n"),
    ]);
    let res = tree_shake(&p, "e.js").expect("shake");
    let e = &res.modules["e.js"];
    assert!(e.contains("used"), "kept used import: {e}");
    // `unused` is dropped from the import list (no longer referenced).
    assert!(!e.contains("unused"), "dropped dead import specifier: {e}");
    assert!(res.stats.dead_imports_removed >= 1);
}

#[test]
fn keeps_side_effect_import() {
    let p = project(&[
        ("a.js", "import './poly.js';\nglobalThis.z = 1;\n"),
        ("poly.js", "globalThis.patched = true;\n"),
    ]);
    let res = tree_shake(&p, "a.js").expect("shake");
    // poly.js is reachable (side-effect import) and the bare import is kept.
    assert!(res.modules.contains_key("poly.js"), "side-effect module kept");
    assert!(res.modules["a.js"].contains("poly.js"), "bare import kept: {}", res.modules["a.js"]);
}
