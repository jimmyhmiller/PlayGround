//! The Vite build manifest (`build.manifest`).
//!
//! When a Vite build sets `build.manifest`, Vite writes `.vite/manifest.json`: a
//! map from each entry's root-relative source path to the emitted chunk that
//! serves it, so a server (or a non-Vite backend integration) can look up which
//! hashed asset URL to inject for a given entry. Diffpack emits the same shape from
//! its OWN emitted files — it is not a transliteration of Vite's hashed chunk
//! layout, it reflects the files Diffpack actually wrote (`<name>.js`,
//! `<name>.css`), so a consumer reading the manifest finds the real assets on disk.
//!
//! Each record carries the fields the manifest contract defines: `file` (the entry
//! chunk), `src` (the source key), `isEntry: true`, and `css` (the extracted
//! stylesheet(s) for the entry). For a multi-page build the HTML document itself is
//! the entry key (`index.html`, `about.html`), matching how Vite keys HTML inputs.

use std::collections::BTreeMap;

/// One emitted page's manifest record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageRecord {
    /// The manifest key: the entry HTML document's path relative to the project
    /// root (e.g. `index.html`, `nested/about.html`).
    pub key: String,
    /// The emitted entry chunk file, relative to the output dir (e.g. `index.js`).
    pub file: String,
    /// The entry's extracted stylesheet file(s), relative to the output dir. Empty
    /// when the page produced no CSS.
    pub css: Vec<String>,
    /// The entry module's source path relative to the project root (the
    /// `<script type="module" src>` target, e.g. `src/main.tsx`), when known.
    pub src: Option<String>,
}

/// Renders the manifest JSON for the given pages, keyed by entry path. Keys are
/// sorted for a deterministic, reproducible artifact.
pub fn render(pages: &[PageRecord]) -> String {
    let mut records: BTreeMap<&str, &PageRecord> = BTreeMap::new();
    for page in pages {
        records.insert(page.key.as_str(), page);
    }
    let mut out = String::from("{\n");
    let mut first = true;
    for (key, page) in &records {
        if !first {
            out.push_str(",\n");
        }
        first = false;
        out.push_str(&format!("  {}: {{\n", json_string(key)));
        out.push_str(&format!("    \"file\": {},\n", json_string(&page.file)));
        if let Some(src) = &page.src {
            out.push_str(&format!("    \"src\": {},\n", json_string(src)));
        }
        out.push_str("    \"isEntry\": true");
        if !page.css.is_empty() {
            let css = page
                .css
                .iter()
                .map(|file| json_string(file))
                .collect::<Vec<_>>()
                .join(", ");
            out.push_str(&format!(",\n    \"css\": [{css}]"));
        }
        out.push_str("\n  }");
    }
    out.push_str("\n}\n");
    out
}

/// JSON-encodes a string (quotes + escapes).
fn json_string(value: &str) -> String {
    serde_json::Value::String(value.to_string()).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_a_single_entry_with_css() {
        let json = render(&[PageRecord {
            key: "index.html".to_string(),
            file: "index.js".to_string(),
            css: vec!["index.css".to_string()],
            src: Some("src/main.tsx".to_string()),
        }]);
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let entry = &value["index.html"];
        assert_eq!(entry["file"], "index.js");
        assert_eq!(entry["src"], "src/main.tsx");
        assert_eq!(entry["isEntry"], true);
        assert_eq!(entry["css"][0], "index.css");
    }

    #[test]
    fn renders_multiple_pages_deterministically_sorted() {
        let pages = vec![
            PageRecord {
                key: "about.html".to_string(),
                file: "about.js".to_string(),
                css: vec![],
                src: Some("src/about.tsx".to_string()),
            },
            PageRecord {
                key: "index.html".to_string(),
                file: "index.js".to_string(),
                css: vec!["index.css".to_string()],
                src: Some("src/main.tsx".to_string()),
            },
        ];
        let json = render(&pages);
        // Deterministic: about.html sorts before index.html.
        assert!(json.find("about.html").unwrap() < json.find("index.html").unwrap());
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["about.html"]["file"], "about.js");
        // A page with no CSS omits the `css` array entirely (Vite's shape).
        assert!(value["about.html"].get("css").is_none());
        assert!(value.as_object().unwrap().len() == 2);
    }
}
