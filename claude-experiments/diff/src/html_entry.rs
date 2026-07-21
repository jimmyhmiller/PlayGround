//! HTML as the build entry.
//!
//! A web application's real entry is its `index.html`: the module graph starts at
//! the `<script type="module" src="...">` tags inside it. This module extracts
//! those entry scripts and rewrites the document for the built output — the
//! original module script tags are removed and replaced by a head-injected
//! script (plus the extracted stylesheet link when the graph produced CSS),
//! matching the shape Vite emits so a built page behaves identically.
//!
//! The scan is a purpose-built tag scanner, not a general HTML parser: it
//! understands comments, quoted attribute values, and the `<script>` element,
//! which is all the entry contract needs. Anything outside that contract that
//! could change build meaning (an inline module script, a malformed tag) is a
//! hard error naming the file and line — never a silent skip.

use std::path::Path;

/// One local `<script type="module" src="...">` entry: the specifier and the
/// byte range of the whole element (open tag through `</script>`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleScript {
    pub src: String,
    pub start: usize,
    pub end: usize,
}

/// The parse of one entry HTML document.
#[derive(Debug, Clone)]
pub struct HtmlEntry {
    pub source: String,
    /// Local module scripts, in document order. External scripts (`http(s)://`,
    /// protocol-relative `//`) are left in place and not listed here.
    pub module_scripts: Vec<ModuleScript>,
}

/// What to inject into `<head>` when rewriting: the built entry scripts and
/// stylesheets, as public URLs.
#[derive(Debug, Clone, Default)]
pub struct HeadInjection {
    pub script_urls: Vec<String>,
    pub stylesheet_urls: Vec<String>,
}

/// Reads and parses `path` as an entry HTML document.
pub fn parse_file(path: &Path) -> Result<HtmlEntry, String> {
    let source = std::fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    parse(&source, &path.display().to_string())
}

/// Parses `source`, collecting local module-script entries. `origin` names the
/// document in errors.
pub fn parse(source: &str, origin: &str) -> Result<HtmlEntry, String> {
    let bytes = source.as_bytes();
    let mut module_scripts = Vec::new();
    let mut cursor = 0;
    while let Some(offset) = find_ci(source, cursor, "<") {
        // Comments hide their contents from the build.
        if source[offset..].starts_with("<!--") {
            let close = find_ci(source, offset + 4, "-->").map(|at| at + 3).ok_or_else(|| {
                format!("{origin}:{}: unterminated HTML comment", line_of(source, offset))
            })?;
            cursor = close;
            continue;
        }
        if !starts_with_ci(&source[offset..], "<script")
            || !source[offset + 7..].starts_with(|c: char| c.is_ascii_whitespace() || c == '>')
        {
            cursor = offset + 1;
            continue;
        }
        let line = line_of(source, offset);
        let open_end = find_tag_end(bytes, offset)
            .ok_or_else(|| format!("{origin}:{line}: unterminated <script> tag"))?;
        let attributes = parse_attributes(&source[offset + 7..open_end - 1]);
        let close = find_ci(source, open_end, "</script")
            .and_then(|at| find_ci(source, at, ">").map(|end| end + 1))
            .ok_or_else(|| format!("{origin}:{line}: <script> without </script>"))?;
        cursor = close;
        let is_module = attributes
            .iter()
            .any(|(name, value)| name == "type" && value.as_deref() == Some("module"));
        if !is_module {
            continue;
        }
        let src = attributes
            .iter()
            .find(|(name, _)| name == "src")
            .and_then(|(_, value)| value.clone());
        let Some(src) = src else {
            // An inline module script would need to be bundled as a virtual
            // entry; that is real work, not a skippable detail.
            return Err(format!(
                "{origin}:{line}: inline <script type=\"module\"> is not supported yet; \
                 move the code into a file and reference it with src"
            ));
        };
        if src.starts_with("http://") || src.starts_with("https://") || src.starts_with("//") {
            continue; // External: left in the document untouched.
        }
        module_scripts.push(ModuleScript { src, start: offset, end: close });
    }
    Ok(HtmlEntry { source: source.to_string(), module_scripts })
}

impl HtmlEntry {
    /// The built document: every local module script removed from its original
    /// position, and the built script/stylesheet references injected at the end
    /// of `<head>` (script first, then stylesheet — the order Vite emits). A
    /// document with no `</head>` is a hard error: there is no honest place to
    /// put the injection.
    pub fn rewrite(&self, origin: &str, injection: &HeadInjection) -> Result<String, String> {
        let mut output = String::with_capacity(self.source.len());
        let mut cursor = 0;
        for script in &self.module_scripts {
            output.push_str(&self.source[cursor..script.start]);
            cursor = script.end;
            // Removing the tag leaves its line blank; swallow one trailing
            // newline (and the indentation before the tag) so the built
            // document has no empty hole.
            let trailing = &self.source[cursor..];
            if let Some(rest) = trailing.strip_prefix('\n') {
                cursor += 1;
                let _ = rest;
                while let Some(last) = output.as_bytes().last() {
                    if *last == b' ' || *last == b'\t' {
                        output.pop();
                    } else {
                        break;
                    }
                }
            }
        }
        output.push_str(&self.source[cursor..]);

        let head_close = find_ci(&output, 0, "</head>")
            .ok_or_else(|| format!("{origin}: no </head> to inject the built entry into"))?;
        // Match the document's indentation: reuse the whitespace that precedes
        // `</head>` for each injected line, plus one level.
        let line_start = output[..head_close].rfind('\n').map(|at| at + 1).unwrap_or(0);
        let indent = &output[line_start..head_close];
        let indent = if indent.trim().is_empty() { indent.to_string() } else { String::new() };
        let mut injected = String::new();
        for url in &injection.script_urls {
            injected.push_str(&format!(
                "{indent}  <script type=\"module\" crossorigin src=\"{url}\"></script>\n"
            ));
        }
        for url in &injection.stylesheet_urls {
            injected.push_str(&format!(
                "{indent}  <link rel=\"stylesheet\" crossorigin href=\"{url}\">\n"
            ));
        }
        let mut result = String::with_capacity(output.len() + injected.len());
        result.push_str(&output[..line_start]);
        result.push_str(&injected);
        result.push_str(&output[line_start..]);
        Ok(result)
    }
}

/// The byte offset one past the `>` closing the tag that starts at `open`,
/// honoring quoted attribute values.
fn find_tag_end(bytes: &[u8], open: usize) -> Option<usize> {
    let mut index = open;
    let mut quote: Option<u8> = None;
    while index < bytes.len() {
        let byte = bytes[index];
        match quote {
            Some(active) => {
                if byte == active {
                    quote = None;
                }
            }
            None => match byte {
                b'"' | b'\'' => quote = Some(byte),
                b'>' => return Some(index + 1),
                _ => {}
            },
        }
        index += 1;
    }
    None
}

/// Parses a tag's attribute text into `(name, value)` pairs. Values are
/// unquoted; a bare attribute (`defer`) has value `None`. Attribute names are
/// lowercased (HTML attribute names are case-insensitive).
fn parse_attributes(text: &str) -> Vec<(String, Option<String>)> {
    let mut attributes = Vec::new();
    let mut rest = text.trim_start();
    while !rest.is_empty() {
        let name_end = rest
            .find(|c: char| c.is_ascii_whitespace() || c == '=')
            .unwrap_or(rest.len());
        let name = rest[..name_end].trim_end_matches('/').to_ascii_lowercase();
        rest = rest[name_end..].trim_start();
        if name.is_empty() {
            break;
        }
        let value = if let Some(after) = rest.strip_prefix('=') {
            let after = after.trim_start();
            if let Some(stripped) = after.strip_prefix('"') {
                let end = stripped.find('"').unwrap_or(stripped.len());
                rest = &stripped[(end + 1).min(stripped.len())..];
                Some(stripped[..end].to_string())
            } else if let Some(stripped) = after.strip_prefix('\'') {
                let end = stripped.find('\'').unwrap_or(stripped.len());
                rest = &stripped[(end + 1).min(stripped.len())..];
                Some(stripped[..end].to_string())
            } else {
                let end = after
                    .find(|c: char| c.is_ascii_whitespace())
                    .unwrap_or(after.len());
                rest = &after[end..];
                Some(after[..end].to_string())
            }
        } else {
            None
        };
        attributes.push((name, value));
        rest = rest.trim_start();
    }
    attributes
}

/// Case-insensitive `needle` search in `haystack` starting at `from`.
fn find_ci(haystack: &str, from: usize, needle: &str) -> Option<usize> {
    let lower = haystack[from..].to_ascii_lowercase();
    lower.find(&needle.to_ascii_lowercase()).map(|at| at + from)
}

fn starts_with_ci(text: &str, prefix: &str) -> bool {
    text.len() >= prefix.len() && text[..prefix.len()].eq_ignore_ascii_case(prefix)
}

fn line_of(source: &str, offset: usize) -> usize {
    source[..offset].bytes().filter(|byte| *byte == b'\n').count() + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOCUMENT: &str = r#"<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>app</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
"#;

    #[test]
    fn extracts_the_local_module_script() {
        let entry = parse(DOCUMENT, "index.html").unwrap();
        assert_eq!(entry.module_scripts.len(), 1);
        assert_eq!(entry.module_scripts[0].src, "/src/main.tsx");
    }

    #[test]
    fn external_and_non_module_scripts_are_not_entries() {
        let source = r#"<head></head><body>
<script src="/plain.js"></script>
<script type="module" src="https://cdn.example.com/x.js"></script>
<script type="module" src="//cdn.example.com/y.js"></script>
</body>"#;
        let entry = parse(source, "index.html").unwrap();
        assert!(entry.module_scripts.is_empty());
    }

    #[test]
    fn a_commented_out_module_script_is_ignored() {
        let source = "<head></head><body><!-- <script type=\"module\" src=\"/gone.js\"></script> --></body>";
        let entry = parse(source, "index.html").unwrap();
        assert!(entry.module_scripts.is_empty());
    }

    #[test]
    fn an_inline_module_script_is_a_hard_error_naming_the_line() {
        let source = "<head></head>\n<body>\n<script type=\"module\">boot()</script>\n</body>";
        let error = parse(source, "index.html").unwrap_err();
        assert!(error.contains("index.html:3"), "{error}");
        assert!(error.contains("inline"), "{error}");
    }

    #[test]
    fn rewrite_moves_the_entry_into_head_and_removes_the_original() {
        let entry = parse(DOCUMENT, "index.html").unwrap();
        let built = entry
            .rewrite(
                "index.html",
                &HeadInjection {
                    script_urls: vec!["/index.js".to_string()],
                    stylesheet_urls: vec!["/index.css".to_string()],
                },
            )
            .unwrap();
        assert!(
            built.contains("<script type=\"module\" crossorigin src=\"/index.js\"></script>"),
            "{built}"
        );
        assert!(
            built.contains("<link rel=\"stylesheet\" crossorigin href=\"/index.css\">"),
            "{built}"
        );
        assert!(!built.contains("/src/main.tsx"), "original tag removed: {built}");
        let head_close = built.find("</head>").unwrap();
        assert!(built.find("/index.js").unwrap() < head_close, "script is inside head: {built}");
        assert!(!built.contains("\n\n\n"), "no blank hole where the tag was: {built}");
    }

    #[test]
    fn a_document_without_head_is_a_hard_error() {
        let entry = parse("<body><script type=\"module\" src=\"/a.js\"></script></body>", "x.html").unwrap();
        let error = entry.rewrite("x.html", &HeadInjection::default()).unwrap_err();
        assert!(error.contains("</head>"), "{error}");
    }

    #[test]
    fn attribute_parsing_handles_quotes_and_case() {
        let source = "<head></head><SCRIPT TYPE='module' SRC='/src/app.ts' defer></SCRIPT>";
        let entry = parse(source, "index.html").unwrap();
        assert_eq!(entry.module_scripts.len(), 1);
        assert_eq!(entry.module_scripts[0].src, "/src/app.ts");
    }
}
