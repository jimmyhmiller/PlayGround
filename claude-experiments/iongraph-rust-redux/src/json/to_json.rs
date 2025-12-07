//! JSON serialization - ToJson trait and Value::to_string()

use super::value::{Number, Value};
use std::collections::HashMap;

/// Trait for types that can be serialized to JSON
pub trait ToJson {
    fn to_json(&self) -> Value;
}

impl Value {
    /// Serialize to a compact JSON string
    pub fn to_string(&self) -> String {
        let mut output = String::new();
        self.write_to(&mut output, false, 0);
        output
    }

    /// Serialize to a pretty-printed JSON string
    pub fn to_string_pretty(&self) -> String {
        let mut output = String::new();
        self.write_to(&mut output, true, 0);
        output
    }

    fn write_to(&self, output: &mut String, pretty: bool, indent: usize) {
        match self {
            Value::Null => output.push_str("null"),
            Value::Bool(true) => output.push_str("true"),
            Value::Bool(false) => output.push_str("false"),
            Value::Number(n) => output.push_str(&n.to_string()),
            Value::String(s) => {
                output.push('"');
                for c in s.chars() {
                    match c {
                        '"' => output.push_str("\\\""),
                        '\\' => output.push_str("\\\\"),
                        '\n' => output.push_str("\\n"),
                        '\r' => output.push_str("\\r"),
                        '\t' => output.push_str("\\t"),
                        c if c.is_control() => {
                            output.push_str(&format!("\\u{:04x}", c as u32));
                        }
                        c => output.push(c),
                    }
                }
                output.push('"');
            }
            Value::Array(arr) => {
                if arr.is_empty() {
                    output.push_str("[]");
                } else if pretty {
                    output.push_str("[\n");
                    for (i, v) in arr.iter().enumerate() {
                        for _ in 0..indent + 2 {
                            output.push(' ');
                        }
                        v.write_to(output, true, indent + 2);
                        if i < arr.len() - 1 {
                            output.push(',');
                        }
                        output.push('\n');
                    }
                    for _ in 0..indent {
                        output.push(' ');
                    }
                    output.push(']');
                } else {
                    output.push('[');
                    for (i, v) in arr.iter().enumerate() {
                        if i > 0 {
                            output.push(',');
                        }
                        v.write_to(output, false, 0);
                    }
                    output.push(']');
                }
            }
            Value::Object(obj) => {
                if obj.is_empty() {
                    output.push_str("{}");
                } else if pretty {
                    output.push_str("{\n");
                    let mut keys: Vec<_> = obj.keys().collect();
                    keys.sort(); // Sort keys for deterministic output
                    for (i, key) in keys.iter().enumerate() {
                        let v = &obj[*key];
                        for _ in 0..indent + 2 {
                            output.push(' ');
                        }
                        output.push('"');
                        output.push_str(key);
                        output.push_str("\": ");
                        v.write_to(output, true, indent + 2);
                        if i < keys.len() - 1 {
                            output.push(',');
                        }
                        output.push('\n');
                    }
                    for _ in 0..indent {
                        output.push(' ');
                    }
                    output.push('}');
                } else {
                    output.push('{');
                    let mut keys: Vec<_> = obj.keys().collect();
                    keys.sort(); // Sort keys for deterministic output
                    for (i, key) in keys.iter().enumerate() {
                        let v = &obj[*key];
                        if i > 0 {
                            output.push(',');
                        }
                        output.push('"');
                        output.push_str(key);
                        output.push_str("\":");
                        v.write_to(output, false, 0);
                    }
                    output.push('}');
                }
            }
        }
    }
}

// Implement ToJson for common types
impl ToJson for bool {
    fn to_json(&self) -> Value {
        Value::Bool(*self)
    }
}

impl ToJson for i32 {
    fn to_json(&self) -> Value {
        Value::Number(Number::Int(*self as i64))
    }
}

impl ToJson for u32 {
    fn to_json(&self) -> Value {
        Value::Number(Number::Int(*self as i64))
    }
}

impl ToJson for i64 {
    fn to_json(&self) -> Value {
        Value::Number(Number::Int(*self))
    }
}

impl ToJson for u64 {
    fn to_json(&self) -> Value {
        Value::Number(Number::from(*self))
    }
}

impl ToJson for f64 {
    fn to_json(&self) -> Value {
        Value::Number(Number::Float(*self))
    }
}

impl ToJson for String {
    fn to_json(&self) -> Value {
        Value::String(self.clone())
    }
}

impl ToJson for &str {
    fn to_json(&self) -> Value {
        Value::String((*self).to_string())
    }
}

impl<T: ToJson> ToJson for Vec<T> {
    fn to_json(&self) -> Value {
        Value::Array(self.iter().map(|v| v.to_json()).collect())
    }
}

impl<T: ToJson> ToJson for Option<T> {
    fn to_json(&self) -> Value {
        match self {
            Some(v) => v.to_json(),
            None => Value::Null,
        }
    }
}

impl<V: ToJson> ToJson for HashMap<String, V> {
    fn to_json(&self) -> Value {
        Value::Object(self.iter().map(|(k, v)| (k.clone(), v.to_json())).collect())
    }
}

impl ToJson for Value {
    fn to_json(&self) -> Value {
        self.clone()
    }
}

/// Convert a Value to a JSON string (convenience function)
pub fn to_string(value: &Value) -> String {
    value.to_string()
}

/// Convert a Value to a pretty-printed JSON string (convenience function)
pub fn to_string_pretty(value: &Value) -> String {
    value.to_string_pretty()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_primitives() {
        assert_eq!(Value::Null.to_string(), "null");
        assert_eq!(Value::Bool(true).to_string(), "true");
        assert_eq!(Value::Bool(false).to_string(), "false");
        assert_eq!(Value::Number(Number::Int(42)).to_string(), "42");
        assert_eq!(Value::Number(Number::Float(3.14)).to_string(), "3.14");
        assert_eq!(Value::String("hello".to_string()).to_string(), "\"hello\"");
    }

    #[test]
    fn test_serialize_escapes() {
        assert_eq!(Value::String("line\nbreak".to_string()).to_string(), "\"line\\nbreak\"");
        assert_eq!(Value::String("tab\there".to_string()).to_string(), "\"tab\\there\"");
        assert_eq!(Value::String("quote\"here".to_string()).to_string(), "\"quote\\\"here\"");
    }

    #[test]
    fn test_serialize_array() {
        let arr = Value::Array(vec![
            Value::Number(Number::Int(1)),
            Value::Number(Number::Int(2)),
            Value::Number(Number::Int(3)),
        ]);
        assert_eq!(arr.to_string(), "[1,2,3]");
    }

    #[test]
    fn test_serialize_object() {
        let mut map = HashMap::new();
        map.insert("a".to_string(), Value::Number(Number::Int(1)));
        let obj = Value::Object(map);
        assert_eq!(obj.to_string(), "{\"a\":1}");
    }

    #[test]
    fn test_roundtrip() {
        use super::super::parser::from_str;
        let original = r#"{"array":[1,2,3],"bool":true,"null":null,"number":42,"string":"hello"}"#;
        let parsed = from_str(original).unwrap();
        let serialized = parsed.to_string();
        assert_eq!(serialized, original);
    }
}
