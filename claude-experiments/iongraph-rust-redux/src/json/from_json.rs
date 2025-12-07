//! FromJson trait for deserializing from JSON Value

use super::parser::ParseError;
use super::value::Value;
use std::collections::HashMap;

/// Trait for types that can be deserialized from a JSON Value
pub trait FromJson: Sized {
    fn from_json(value: &Value) -> Result<Self, ParseError>;
}

/// Helper to create a parse error
pub fn parse_error(msg: impl Into<String>) -> ParseError {
    ParseError {
        message: msg.into(),
        position: 0,
    }
}

// Primitive implementations

impl FromJson for bool {
    fn from_json(value: &Value) -> Result<Self, ParseError> {
        value.as_bool().ok_or_else(|| parse_error("expected boolean"))
    }
}

impl FromJson for i32 {
    fn from_json(value: &Value) -> Result<Self, ParseError> {
        value
            .as_i64()
            .and_then(|n| i32::try_from(n).ok())
            .ok_or_else(|| parse_error("expected i32"))
    }
}

impl FromJson for u32 {
    fn from_json(value: &Value) -> Result<Self, ParseError> {
        value
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| parse_error("expected u32"))
    }
}

impl FromJson for i64 {
    fn from_json(value: &Value) -> Result<Self, ParseError> {
        value.as_i64().ok_or_else(|| parse_error("expected i64"))
    }
}

impl FromJson for u64 {
    fn from_json(value: &Value) -> Result<Self, ParseError> {
        value.as_u64().ok_or_else(|| parse_error("expected u64"))
    }
}

impl FromJson for f64 {
    fn from_json(value: &Value) -> Result<Self, ParseError> {
        value.as_f64().ok_or_else(|| parse_error("expected number"))
    }
}

impl FromJson for String {
    fn from_json(value: &Value) -> Result<Self, ParseError> {
        value
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| parse_error("expected string"))
    }
}

impl<T: FromJson> FromJson for Option<T> {
    fn from_json(value: &Value) -> Result<Self, ParseError> {
        if value.is_null() {
            Ok(None)
        } else {
            T::from_json(value).map(Some)
        }
    }
}

impl<T: FromJson> FromJson for Vec<T> {
    fn from_json(value: &Value) -> Result<Self, ParseError> {
        let arr = value.as_array().ok_or_else(|| parse_error("expected array"))?;
        arr.iter().map(T::from_json).collect()
    }
}

impl<V: FromJson> FromJson for HashMap<String, V> {
    fn from_json(value: &Value) -> Result<Self, ParseError> {
        let obj = value.as_object().ok_or_else(|| parse_error("expected object"))?;
        obj.iter()
            .map(|(k, v)| V::from_json(v).map(|v| (k.clone(), v)))
            .collect()
    }
}

impl FromJson for Value {
    fn from_json(value: &Value) -> Result<Self, ParseError> {
        Ok(value.clone())
    }
}

/// Helper trait for getting optional fields with defaults
pub trait JsonObjectExt {
    /// Get a required field
    fn get_field<T: FromJson>(&self, key: &str) -> Result<T, ParseError>;

    /// Get an optional field (returns None if missing or null)
    fn get_field_opt<T: FromJson>(&self, key: &str) -> Result<Option<T>, ParseError>;

    /// Get a field with a default value
    fn get_field_or<T: FromJson>(&self, key: &str, default: T) -> Result<T, ParseError>;

    /// Get a field with a default from Default trait
    fn get_field_or_default<T: FromJson + Default>(&self, key: &str) -> Result<T, ParseError>;

    /// Get a field, trying multiple key names (for rename handling)
    fn get_field_renamed<T: FromJson>(&self, keys: &[&str]) -> Result<T, ParseError>;

    /// Get an optional field, trying multiple key names
    fn get_field_renamed_opt<T: FromJson>(&self, keys: &[&str]) -> Result<Option<T>, ParseError>;

    /// Get a field with default, trying multiple key names
    fn get_field_renamed_or_default<T: FromJson + Default>(
        &self,
        keys: &[&str],
    ) -> Result<T, ParseError>;
}

impl JsonObjectExt for Value {
    fn get_field<T: FromJson>(&self, key: &str) -> Result<T, ParseError> {
        let value = self
            .get(key)
            .ok_or_else(|| parse_error(format!("missing required field '{}'", key)))?;
        T::from_json(value)
    }

    fn get_field_opt<T: FromJson>(&self, key: &str) -> Result<Option<T>, ParseError> {
        match self.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => T::from_json(v).map(Some),
        }
    }

    fn get_field_or<T: FromJson>(&self, key: &str, default: T) -> Result<T, ParseError> {
        match self.get(key) {
            None | Some(Value::Null) => Ok(default),
            Some(v) => T::from_json(v),
        }
    }

    fn get_field_or_default<T: FromJson + Default>(&self, key: &str) -> Result<T, ParseError> {
        self.get_field_or(key, T::default())
    }

    fn get_field_renamed<T: FromJson>(&self, keys: &[&str]) -> Result<T, ParseError> {
        for key in keys {
            if let Some(v) = self.get(key) {
                return T::from_json(v);
            }
        }
        Err(parse_error(format!(
            "missing required field (tried: {})",
            keys.join(", ")
        )))
    }

    fn get_field_renamed_opt<T: FromJson>(&self, keys: &[&str]) -> Result<Option<T>, ParseError> {
        for key in keys {
            if let Some(v) = self.get(key) {
                if !v.is_null() {
                    return T::from_json(v).map(Some);
                }
            }
        }
        Ok(None)
    }

    fn get_field_renamed_or_default<T: FromJson + Default>(
        &self,
        keys: &[&str],
    ) -> Result<T, ParseError> {
        for key in keys {
            if let Some(v) = self.get(key) {
                if !v.is_null() {
                    return T::from_json(v);
                }
            }
        }
        Ok(T::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::parser::from_str;

    #[test]
    fn test_primitives() {
        assert_eq!(bool::from_json(&from_str("true").unwrap()).unwrap(), true);
        assert_eq!(i32::from_json(&from_str("42").unwrap()).unwrap(), 42);
        assert_eq!(u32::from_json(&from_str("42").unwrap()).unwrap(), 42);
        assert_eq!(
            String::from_json(&from_str("\"hello\"").unwrap()).unwrap(),
            "hello"
        );
    }

    #[test]
    fn test_option() {
        assert_eq!(
            Option::<i32>::from_json(&from_str("null").unwrap()).unwrap(),
            None
        );
        assert_eq!(
            Option::<i32>::from_json(&from_str("42").unwrap()).unwrap(),
            Some(42)
        );
    }

    #[test]
    fn test_vec() {
        assert_eq!(
            Vec::<i32>::from_json(&from_str("[1, 2, 3]").unwrap()).unwrap(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn test_object_ext() {
        let json = from_str(r#"{"name": "test", "count": 42}"#).unwrap();
        assert_eq!(json.get_field::<String>("name").unwrap(), "test");
        assert_eq!(json.get_field::<i32>("count").unwrap(), 42);
        assert_eq!(json.get_field_opt::<i32>("missing").unwrap(), None);
        assert_eq!(json.get_field_or("missing", 0).unwrap(), 0);
    }

    #[test]
    fn test_renamed() {
        let json = from_str(r#"{"loopDepth": 5}"#).unwrap();
        assert_eq!(
            json.get_field_renamed::<u32>(&["loop_depth", "loopDepth"])
                .unwrap(),
            5
        );

        let json2 = from_str(r#"{"loop_depth": 3}"#).unwrap();
        assert_eq!(
            json2
                .get_field_renamed::<u32>(&["loop_depth", "loopDepth"])
                .unwrap(),
            3
        );
    }
}
