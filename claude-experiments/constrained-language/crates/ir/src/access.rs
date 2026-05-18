use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use thiserror::Error;

/// A path naming a slice of state, e.g. `subscribers[$event.repo]`,
/// `in_progress[*].status`, or `config.api_url`.
///
/// Serialized as a string in JSON.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessPath {
    pub cell: String,
    pub segments: Vec<AccessSegment>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessSegment {
    Field(String),
    Key(KeyBinding),
    Wildcard,
}

/// A binding source for a key inside a `[...]` segment.
///
/// v0.1 supports only `$event.<field>.<subfield>...`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyBinding {
    Event(Vec<String>),
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ParseError {
    #[error("access path is empty")]
    Empty,
    #[error("missing cell name")]
    MissingCell,
    #[error("empty field after `.`")]
    EmptyField,
    #[error("unclosed `[`")]
    UnclosedBracket,
    #[error("empty `[]` segment")]
    EmptyBracket,
    #[error("unsupported key expression: `{0}` (v0.1 supports `$event.<path>` and `*`)")]
    InvalidKey(String),
    #[error("unexpected character `{0}`")]
    Unexpected(char),
    #[error("`$event` reference must include at least one field, e.g. `$event.foo`")]
    EventNeedsField,
}

impl AccessPath {
    pub fn parse(input: &str) -> Result<Self, ParseError> {
        let s = input.trim();
        if s.is_empty() {
            return Err(ParseError::Empty);
        }
        let mut chars = s.chars().peekable();

        // Cell name: everything up to first `.` or `[`.
        let mut cell = String::new();
        while let Some(&c) = chars.peek() {
            if c == '.' || c == '[' {
                break;
            }
            cell.push(c);
            chars.next();
        }
        if cell.is_empty() {
            return Err(ParseError::MissingCell);
        }

        let mut segments = Vec::new();
        while let Some(&c) = chars.peek() {
            match c {
                '.' => {
                    chars.next();
                    let mut field = String::new();
                    while let Some(&c) = chars.peek() {
                        if c == '.' || c == '[' {
                            break;
                        }
                        field.push(c);
                        chars.next();
                    }
                    if field.is_empty() {
                        return Err(ParseError::EmptyField);
                    }
                    segments.push(AccessSegment::Field(field));
                }
                '[' => {
                    chars.next();
                    let mut inner = String::new();
                    let mut closed = false;
                    for c in chars.by_ref() {
                        if c == ']' {
                            closed = true;
                            break;
                        }
                        inner.push(c);
                    }
                    if !closed {
                        return Err(ParseError::UnclosedBracket);
                    }
                    let inner = inner.trim();
                    if inner.is_empty() {
                        return Err(ParseError::EmptyBracket);
                    }
                    if inner == "*" {
                        segments.push(AccessSegment::Wildcard);
                    } else if let Some(rest) = inner.strip_prefix("$event") {
                        let rest = rest.trim_start();
                        let path = if let Some(after_dot) = rest.strip_prefix('.') {
                            let parts: Vec<String> =
                                after_dot.split('.').map(str::to_string).collect();
                            if parts.iter().any(|p| p.is_empty()) {
                                return Err(ParseError::EmptyField);
                            }
                            parts
                        } else {
                            return Err(ParseError::EventNeedsField);
                        };
                        segments.push(AccessSegment::Key(KeyBinding::Event(path)));
                    } else {
                        return Err(ParseError::InvalidKey(inner.to_string()));
                    }
                }
                other => return Err(ParseError::Unexpected(other)),
            }
        }

        Ok(AccessPath { cell, segments })
    }
}

impl fmt::Display for AccessPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.cell)?;
        for seg in &self.segments {
            match seg {
                AccessSegment::Field(name) => write!(f, ".{}", name)?,
                AccessSegment::Key(KeyBinding::Event(path)) => {
                    write!(f, "[$event.{}]", path.join("."))?
                }
                AccessSegment::Wildcard => f.write_str("[*]")?,
            }
        }
        Ok(())
    }
}

impl Serialize for AccessPath {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for AccessPath {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        AccessPath::parse(&s).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> AccessPath {
        AccessPath::parse(s).expect("parse")
    }

    #[test]
    fn parses_bare_cell() {
        let a = p("config");
        assert_eq!(a.cell, "config");
        assert!(a.segments.is_empty());
    }

    #[test]
    fn parses_field_navigation() {
        let a = p("config.api.url");
        assert_eq!(a.cell, "config");
        assert_eq!(
            a.segments,
            vec![
                AccessSegment::Field("api".into()),
                AccessSegment::Field("url".into()),
            ]
        );
    }

    #[test]
    fn parses_event_key() {
        let a = p("subscribers[$event.repo]");
        assert_eq!(a.cell, "subscribers");
        assert_eq!(
            a.segments,
            vec![AccessSegment::Key(KeyBinding::Event(vec!["repo".into()]))]
        );
    }

    #[test]
    fn parses_wildcard() {
        let a = p("in_progress[*]");
        assert_eq!(a.cell, "in_progress");
        assert_eq!(a.segments, vec![AccessSegment::Wildcard]);
    }

    #[test]
    fn parses_key_then_field() {
        let a = p("in_progress[$event.deploy_id].status");
        assert_eq!(a.cell, "in_progress");
        assert_eq!(
            a.segments,
            vec![
                AccessSegment::Key(KeyBinding::Event(vec!["deploy_id".into()])),
                AccessSegment::Field("status".into()),
            ]
        );
    }

    #[test]
    fn parses_nested_event_path() {
        let a = p("subscribers[$event.context.repo]");
        assert_eq!(
            a.segments,
            vec![AccessSegment::Key(KeyBinding::Event(vec![
                "context".into(),
                "repo".into()
            ]))]
        );
    }

    #[test]
    fn rejects_empty() {
        assert_eq!(AccessPath::parse(""), Err(ParseError::Empty));
        assert_eq!(AccessPath::parse("   "), Err(ParseError::Empty));
    }

    #[test]
    fn rejects_bad_keys() {
        assert!(matches!(
            AccessPath::parse("x[]"),
            Err(ParseError::EmptyBracket)
        ));
        assert!(matches!(
            AccessPath::parse("x[$event]"),
            Err(ParseError::EventNeedsField)
        ));
        assert!(matches!(
            AccessPath::parse("x[123]"),
            Err(ParseError::InvalidKey(_))
        ));
        assert!(matches!(
            AccessPath::parse("x[$event.foo"),
            Err(ParseError::UnclosedBracket)
        ));
    }

    #[test]
    fn display_roundtrips() {
        for s in [
            "config",
            "config.api.url",
            "in_progress[*]",
            "subscribers[$event.repo]",
            "in_progress[$event.deploy_id].status",
            "subscribers[$event.context.repo]",
        ] {
            let parsed = AccessPath::parse(s).unwrap();
            assert_eq!(parsed.to_string(), s);
        }
    }
}
