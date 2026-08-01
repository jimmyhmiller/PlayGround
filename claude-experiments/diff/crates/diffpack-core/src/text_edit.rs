//! Shared, deterministic source-edit primitives for transform providers.

use oxc_span::Span;

pub fn apply_edits(source: &str, preamble: String, mut edits: Vec<(Span, String)>) -> String {
    edits.sort_by_key(|(span, _)| span.start);
    let mut output = preamble;
    let mut cursor = 0_usize;
    for (span, replacement) in edits {
        let start = span.start as usize;
        let end = span.end as usize;
        if start < cursor {
            continue;
        }
        output.push_str(&source[cursor..start]);
        output.push_str(&replacement);
        cursor = end;
    }
    output.push_str(&source[cursor..]);
    output
}

pub fn quote(value: &str) -> String {
    serde_json::to_string(value).expect("serializing a JavaScript string cannot fail")
}
