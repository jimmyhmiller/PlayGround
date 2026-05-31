//! A minimal JSON value with a serializer that is byte-compatible with
//! `nlohmann::ordered_json::dump(2)`, which is what upstream jsir uses to emit
//! `ast.json` (see `maldoca/js/driver/driver.h`: `json.dump(2)`).
//!
//! We do NOT use serde for *output* because matching nlohmann's exact byte
//! layout (number formatting, string escaping, 2-space indent, `": "`
//! separators, empty-container handling) requires full control. `serde_json`
//! is used elsewhere only to *parse* input.
//!
//! Object key order is preserved (insertion order) to mirror `ordered_json`.

use std::fmt::Write as _;

/// An insertion-ordered JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// Integer value (e.g. `start`, `end`, `line`, comment uids).
    Int(i64),
    /// Floating-point value (e.g. NumericLiteral `value`). Integer-valued
    /// finite floats render with a trailing `.0` to match nlohmann.
    Float(f64),
    Str(String),
    Array(Vec<Json>),
    /// Object with preserved key order.
    Object(Vec<(String, Json)>),
}

impl Json {
    /// Convenience: build an object from ordered (key, value) pairs.
    pub fn object(pairs: Vec<(String, Json)>) -> Json {
        Json::Object(pairs)
    }

    /// Serialize byte-compatibly with `nlohmann::ordered_json::dump(2)`.
    pub fn dump2(&self) -> String {
        let mut out = String::new();
        self.write(&mut out, 0);
        out
    }

    fn write(&self, out: &mut String, indent: usize) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Int(n) => {
                let _ = write!(out, "{n}");
            }
            Json::Float(f) => out.push_str(&format_double(*f)),
            Json::Str(s) => write_json_string(out, s),
            Json::Array(items) => {
                if items.is_empty() {
                    // nlohmann dump(2) renders an empty array as "[]".
                    out.push_str("[]");
                    return;
                }
                out.push_str("[\n");
                let inner = indent + 2;
                for (i, item) in items.iter().enumerate() {
                    push_spaces(out, inner);
                    item.write(out, inner);
                    if i + 1 < items.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                push_spaces(out, indent);
                out.push(']');
            }
            Json::Object(pairs) => {
                if pairs.is_empty() {
                    out.push_str("{}");
                    return;
                }
                out.push_str("{\n");
                let inner = indent + 2;
                for (i, (k, v)) in pairs.iter().enumerate() {
                    push_spaces(out, inner);
                    write_json_string(out, k);
                    out.push_str(": ");
                    v.write(out, inner);
                    if i + 1 < pairs.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                push_spaces(out, indent);
                out.push('}');
            }
        }
    }
}

fn push_spaces(out: &mut String, n: usize) {
    for _ in 0..n {
        out.push(' ');
    }
}

/// Format a double the way nlohmann does: shortest round-trip representation,
/// but integer-valued finite doubles always carry a `.0` suffix.
///
/// nlohmann uses a Grisu-style shortest dtoa. Rust's `{}` for f64 is also a
/// shortest round-trip formatter, so the significant digits agree; the only
/// systematic difference is that Rust prints `1` for `1.0`, whereas nlohmann
/// prints `1.0`. We reconcile by appending `.0` when the formatted text has no
/// `.`, `e`, or `E`.
fn format_double(f: f64) -> String {
    if f.is_nan() {
        // nlohmann emits `null` for NaN/Infinity (JSON has no representation).
        return "null".to_string();
    }
    if f.is_infinite() {
        return "null".to_string();
    }
    let mut s = format!("{f}");
    if !s.contains(['.', 'e', 'E']) {
        s.push_str(".0");
    }
    s
}

/// Escape a string the way nlohmann's dump does (default ensure_ascii=false):
/// escape `"`, `\`, and control characters < 0x20 with `\uXXXX` (or the short
/// forms `\b \t \n \f \r`). Non-ASCII bytes are passed through (UTF-8).
fn write_json_string(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{09}' => out.push_str("\\t"),
            '\u{0a}' => out.push_str("\\n"),
            '\u{0c}' => out.push_str("\\f"),
            '\u{0d}' => out.push_str("\\r"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_containers() {
        assert_eq!(Json::Array(vec![]).dump2(), "[]");
        assert_eq!(Json::Object(vec![]).dump2(), "{}");
    }

    #[test]
    fn integer_valued_float_gets_dot_zero() {
        assert_eq!(format_double(1.0), "1.0");
        assert_eq!(format_double(3.0), "3.0");
        assert_eq!(format_double(0.0), "0.0");
        assert_eq!(format_double(-5.0), "-5.0");
        assert_eq!(format_double(0.5), "0.5");
        assert_eq!(format_double(0.1), "0.1");
    }

    #[test]
    fn nested_object_indent_matches_nlohmann() {
        // Mirrors the head of a fixture ast.json.
        let j = Json::object(vec![
            ("type".into(), Json::Str("File".into())),
            (
                "loc".into(),
                Json::object(vec![
                    (
                        "start".into(),
                        Json::object(vec![
                            ("line".into(), Json::Int(1)),
                            ("column".into(), Json::Int(0)),
                        ]),
                    ),
                    (
                        "end".into(),
                        Json::object(vec![
                            ("line".into(), Json::Int(3)),
                            ("column".into(), Json::Int(0)),
                        ]),
                    ),
                ]),
            ),
            ("start".into(), Json::Int(0)),
            ("end".into(), Json::Int(10)),
        ]);
        let expected = "{\n  \"type\": \"File\",\n  \"loc\": {\n    \"start\": {\n      \"line\": 1,\n      \"column\": 0\n    },\n    \"end\": {\n      \"line\": 3,\n      \"column\": 0\n    }\n  },\n  \"start\": 0,\n  \"end\": 10\n}";
        assert_eq!(j.dump2(), expected);
    }

    #[test]
    fn numeric_literal_value_renders_as_float() {
        let j = Json::object(vec![
            ("type".into(), Json::Str("NumericLiteral".into())),
            ("value".into(), Json::Float(1.0)),
        ]);
        assert_eq!(j.dump2(), "{\n  \"type\": \"NumericLiteral\",\n  \"value\": 1.0\n}");
    }

    #[test]
    fn string_escaping() {
        let mut s = String::new();
        write_json_string(&mut s, "a\"b\\c\nd");
        assert_eq!(s, "\"a\\\"b\\\\c\\nd\"");
    }
}
