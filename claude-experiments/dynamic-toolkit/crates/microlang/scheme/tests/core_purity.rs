//! Structural enforcement of the library/language split: the CORE crate must
//! contain no language-specific concept in CODE. Comments may reference other
//! languages to explain design intent; code may not.
use std::fs;
use std::path::Path;

#[test]
fn core_has_no_language_names_in_code() {
    let core_src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../src");
    let forbidden = ["scheme", "clojure", "call/cc", "call_cc", "syntax-rules", "syntax_rules"];
    let mut leaks = Vec::new();
    visit(&core_src, &forbidden, &mut leaks);
    assert!(
        leaks.is_empty(),
        "language concept leaked into core CODE (not comments):\n{}",
        leaks.join("\n")
    );
}

fn visit(dir: &Path, forbidden: &[&str], leaks: &mut Vec<String>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            visit(&path, forbidden, leaks);
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let text = fs::read_to_string(&path).unwrap();
        for (n, line) in text.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("").to_lowercase();
            for term in forbidden {
                if code.contains(term) {
                    leaks.push(format!("{}:{}: {}", path.display(), n + 1, line.trim()));
                }
            }
        }
    }
}
