//! Zero-dependency JSON parser (optimized)

use super::value::{Number, Value};
use std::collections::HashMap;

/// JSON parsing error
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
    pub position: usize,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JSON parse error at position {}: {}", self.position, self.message)
    }
}

impl std::error::Error for ParseError {}

/// High-performance JSON parser using byte slices
pub struct Parser<'a> {
    input: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    #[inline]
    pub fn new(input: &'a str) -> Self {
        Parser {
            input: input.as_bytes(),
            pos: 0,
        }
    }

    #[inline]
    fn error(&self, message: impl Into<String>) -> ParseError {
        ParseError {
            message: message.into(),
            position: self.pos,
        }
    }

    #[inline]
    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }

    #[inline]
    fn advance(&mut self) {
        self.pos += 1;
    }

    #[inline]
    fn skip_whitespace(&mut self) {
        while self.pos < self.input.len() {
            match self.input[self.pos] {
                b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                _ => break,
            }
        }
    }

    pub fn parse(&mut self) -> Result<Value, ParseError> {
        self.skip_whitespace();
        let value = self.parse_value()?;
        self.skip_whitespace();
        if self.pos < self.input.len() {
            return Err(self.error("unexpected trailing characters"));
        }
        Ok(value)
    }

    fn parse_value(&mut self) -> Result<Value, ParseError> {
        self.skip_whitespace();
        match self.peek() {
            Some(b'"') => self.parse_string(),
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b't') => self.parse_true(),
            Some(b'f') => self.parse_false(),
            Some(b'n') => self.parse_null(),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.parse_number(),
            Some(c) => Err(self.error(format!("unexpected character '{}'", c as char))),
            None => Err(self.error("unexpected end of input")),
        }
    }

    #[inline]
    fn parse_null(&mut self) -> Result<Value, ParseError> {
        if self.input[self.pos..].starts_with(b"null") {
            self.pos += 4;
            Ok(Value::Null)
        } else {
            Err(self.error("expected 'null'"))
        }
    }

    #[inline]
    fn parse_true(&mut self) -> Result<Value, ParseError> {
        if self.input[self.pos..].starts_with(b"true") {
            self.pos += 4;
            Ok(Value::Bool(true))
        } else {
            Err(self.error("expected 'true'"))
        }
    }

    #[inline]
    fn parse_false(&mut self) -> Result<Value, ParseError> {
        if self.input[self.pos..].starts_with(b"false") {
            self.pos += 5;
            Ok(Value::Bool(false))
        } else {
            Err(self.error("expected 'false'"))
        }
    }

    fn parse_string(&mut self) -> Result<Value, ParseError> {
        self.advance(); // skip opening quote
        let s = self.parse_string_content()?;
        Ok(Value::String(s))
    }

    fn parse_string_content(&mut self) -> Result<String, ParseError> {
        let start = self.pos;

        // Fast path: scan for end of string without escapes
        while self.pos < self.input.len() {
            match self.input[self.pos] {
                b'"' => {
                    // No escapes, use fast path
                    let s = unsafe {
                        std::str::from_utf8_unchecked(&self.input[start..self.pos])
                    };
                    self.advance();
                    return Ok(s.to_string());
                }
                b'\\' => {
                    // Has escapes, use slow path
                    break;
                }
                c if c < 0x20 => {
                    return Err(self.error("control characters not allowed in strings"));
                }
                _ => self.pos += 1,
            }
        }

        // Slow path: handle escapes
        let mut result = String::new();
        // Copy what we already scanned
        result.push_str(unsafe {
            std::str::from_utf8_unchecked(&self.input[start..self.pos])
        });

        loop {
            if self.pos >= self.input.len() {
                return Err(self.error("unterminated string"));
            }
            match self.input[self.pos] {
                b'"' => {
                    self.advance();
                    return Ok(result);
                }
                b'\\' => {
                    self.advance();
                    let escaped = self.parse_escape_sequence()?;
                    result.push(escaped);
                }
                c if c < 0x20 => {
                    return Err(self.error("control characters not allowed in strings"));
                }
                c => {
                    // Handle UTF-8 multi-byte sequences
                    if c < 0x80 {
                        result.push(c as char);
                        self.advance();
                    } else {
                        let start = self.pos;
                        // Determine length of UTF-8 sequence
                        let len = if c & 0xE0 == 0xC0 { 2 }
                        else if c & 0xF0 == 0xE0 { 3 }
                        else if c & 0xF8 == 0xF0 { 4 }
                        else { 1 };
                        self.pos += len;
                        if let Ok(s) = std::str::from_utf8(&self.input[start..self.pos]) {
                            result.push_str(s);
                        }
                    }
                }
            }
        }
    }

    fn parse_escape_sequence(&mut self) -> Result<char, ParseError> {
        if self.pos >= self.input.len() {
            return Err(self.error("unterminated escape sequence"));
        }
        let c = self.input[self.pos];
        self.advance();
        match c {
            b'"' => Ok('"'),
            b'\\' => Ok('\\'),
            b'/' => Ok('/'),
            b'b' => Ok('\u{0008}'),
            b'f' => Ok('\u{000C}'),
            b'n' => Ok('\n'),
            b'r' => Ok('\r'),
            b't' => Ok('\t'),
            b'u' => self.parse_unicode_escape(),
            _ => Err(self.error(format!("invalid escape sequence '\\{}'", c as char))),
        }
    }

    fn parse_unicode_escape(&mut self) -> Result<char, ParseError> {
        if self.pos + 4 > self.input.len() {
            return Err(self.error("unterminated unicode escape"));
        }

        let hex = &self.input[self.pos..self.pos + 4];
        self.pos += 4;

        let code = parse_hex_u16(hex)
            .ok_or_else(|| self.error("invalid unicode escape"))?;

        // Handle surrogate pairs
        if (0xD800..=0xDBFF).contains(&code) {
            if self.pos + 6 <= self.input.len()
                && self.input[self.pos] == b'\\'
                && self.input[self.pos + 1] == b'u'
            {
                self.pos += 2;
                let hex2 = &self.input[self.pos..self.pos + 4];
                self.pos += 4;

                if let Some(code2) = parse_hex_u16(hex2) {
                    if (0xDC00..=0xDFFF).contains(&code2) {
                        let combined = 0x10000 + ((code as u32 - 0xD800) << 10) + (code2 as u32 - 0xDC00);
                        return char::from_u32(combined)
                            .ok_or_else(|| self.error("invalid unicode code point"));
                    }
                }
            }
            return Err(self.error("invalid surrogate pair"));
        }

        char::from_u32(code as u32)
            .ok_or_else(|| self.error("invalid unicode code point"))
    }

    fn parse_number(&mut self) -> Result<Value, ParseError> {
        let start = self.pos;
        let mut is_float = false;

        // Optional minus
        if self.peek() == Some(b'-') {
            self.advance();
        }

        // Integer part
        match self.peek() {
            Some(b'0') => {
                self.advance();
            }
            Some(c) if c.is_ascii_digit() => {
                while let Some(c) = self.peek() {
                    if c.is_ascii_digit() {
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
            _ => return Err(self.error("expected digit")),
        }

        // Fractional part
        if self.peek() == Some(b'.') {
            is_float = true;
            self.advance();
            let mut has_digit = false;
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    has_digit = true;
                    self.advance();
                } else {
                    break;
                }
            }
            if !has_digit {
                return Err(self.error("expected digit after decimal point"));
            }
        }

        // Exponent part
        if let Some(b'e') | Some(b'E') = self.peek() {
            is_float = true;
            self.advance();
            if let Some(b'+') | Some(b'-') = self.peek() {
                self.advance();
            }
            let mut has_digit = false;
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    has_digit = true;
                    self.advance();
                } else {
                    break;
                }
            }
            if !has_digit {
                return Err(self.error("expected digit in exponent"));
            }
        }

        let num_str = unsafe {
            std::str::from_utf8_unchecked(&self.input[start..self.pos])
        };

        if is_float {
            let n: f64 = num_str.parse()
                .map_err(|_| ParseError {
                    message: format!("invalid number '{}'", num_str),
                    position: start,
                })?;
            Ok(Value::Number(Number::Float(n)))
        } else {
            // Try parsing as i64 first
            if let Ok(n) = num_str.parse::<i64>() {
                Ok(Value::Number(Number::Int(n)))
            } else {
                // Fall back to f64 for very large numbers
                let n: f64 = num_str.parse()
                    .map_err(|_| ParseError {
                        message: format!("invalid number '{}'", num_str),
                        position: start,
                    })?;
                Ok(Value::Number(Number::Float(n)))
            }
        }
    }

    fn parse_array(&mut self) -> Result<Value, ParseError> {
        self.advance(); // skip '['
        self.skip_whitespace();

        if self.peek() == Some(b']') {
            self.advance();
            return Ok(Value::Array(Vec::new()));
        }

        let mut elements = Vec::new();
        loop {
            elements.push(self.parse_value()?);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => {
                    self.advance();
                    self.skip_whitespace();
                }
                Some(b']') => {
                    self.advance();
                    return Ok(Value::Array(elements));
                }
                _ => return Err(self.error("expected ',' or ']'")),
            }
        }
    }

    fn parse_object(&mut self) -> Result<Value, ParseError> {
        self.advance(); // skip '{'
        self.skip_whitespace();

        if self.peek() == Some(b'}') {
            self.advance();
            return Ok(Value::Object(HashMap::new()));
        }

        let mut map = HashMap::new();
        loop {
            self.skip_whitespace();
            if self.peek() != Some(b'"') {
                return Err(self.error("expected string key"));
            }
            self.advance(); // consume opening quote
            let key = self.parse_string_content()?;

            self.skip_whitespace();
            if self.peek() != Some(b':') {
                return Err(self.error("expected ':'"));
            }
            self.advance();

            let value = self.parse_value()?;
            map.insert(key, value);

            self.skip_whitespace();
            match self.peek() {
                Some(b',') => {
                    self.advance();
                    self.skip_whitespace();
                }
                Some(b'}') => {
                    self.advance();
                    return Ok(Value::Object(map));
                }
                _ => return Err(self.error("expected ',' or '}'")),
            }
        }
    }
}

#[inline]
fn parse_hex_u16(bytes: &[u8]) -> Option<u16> {
    let mut result: u16 = 0;
    for &b in bytes {
        let digit = match b {
            b'0'..=b'9' => b - b'0',
            b'a'..=b'f' => b - b'a' + 10,
            b'A'..=b'F' => b - b'A' + 10,
            _ => return None,
        };
        result = result * 16 + digit as u16;
    }
    Some(result)
}

/// Parse a JSON string into a Value
pub fn from_str(input: &str) -> Result<Value, ParseError> {
    Parser::new(input).parse()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null() {
        assert_eq!(from_str("null").unwrap(), Value::Null);
    }

    #[test]
    fn test_bool() {
        assert_eq!(from_str("true").unwrap(), Value::Bool(true));
        assert_eq!(from_str("false").unwrap(), Value::Bool(false));
    }

    #[test]
    fn test_numbers() {
        assert_eq!(from_str("0").unwrap(), Value::Number(Number::Int(0)));
        assert_eq!(from_str("42").unwrap(), Value::Number(Number::Int(42)));
        assert_eq!(from_str("-17").unwrap(), Value::Number(Number::Int(-17)));
        assert_eq!(from_str("3.14").unwrap(), Value::Number(Number::Float(3.14)));
        assert_eq!(from_str("1e10").unwrap(), Value::Number(Number::Float(1e10)));
        assert_eq!(from_str("2.5E-3").unwrap(), Value::Number(Number::Float(2.5e-3)));
    }

    #[test]
    fn test_strings() {
        assert_eq!(from_str(r#""""#).unwrap(), Value::String("".to_string()));
        assert_eq!(from_str(r#""hello""#).unwrap(), Value::String("hello".to_string()));
        assert_eq!(from_str(r#""hello\nworld""#).unwrap(), Value::String("hello\nworld".to_string()));
        assert_eq!(from_str(r#""tab\there""#).unwrap(), Value::String("tab\there".to_string()));
        assert_eq!(from_str(r#""quote\"here""#).unwrap(), Value::String("quote\"here".to_string()));
        assert_eq!(from_str(r#""\u0041""#).unwrap(), Value::String("A".to_string()));
    }

    #[test]
    fn test_arrays() {
        assert_eq!(from_str("[]").unwrap(), Value::Array(vec![]));
        assert_eq!(from_str("[1, 2, 3]").unwrap(), Value::Array(vec![
            Value::Number(Number::Int(1)),
            Value::Number(Number::Int(2)),
            Value::Number(Number::Int(3)),
        ]));
        assert_eq!(from_str(r#"["a", "b"]"#).unwrap(), Value::Array(vec![
            Value::String("a".to_string()),
            Value::String("b".to_string()),
        ]));
    }

    #[test]
    fn test_objects() {
        assert_eq!(from_str("{}").unwrap(), Value::Object(HashMap::new()));
        let obj = from_str(r#"{"name": "test", "value": 42}"#).unwrap();
        assert_eq!(obj.get("name").unwrap().as_str(), Some("test"));
        assert_eq!(obj.get("value").unwrap().as_i64(), Some(42));
    }

    #[test]
    fn test_nested() {
        let json = r#"{"items": [1, 2, {"nested": true}]}"#;
        let value = from_str(json).unwrap();
        assert!(value.get("items").unwrap().is_array());
    }

    #[test]
    fn test_whitespace() {
        assert_eq!(from_str("  null  ").unwrap(), Value::Null);
        assert_eq!(from_str("{\n  \"key\"\n:\n  \"value\"\n}").unwrap().get("key").unwrap().as_str(), Some("value"));
    }
}
