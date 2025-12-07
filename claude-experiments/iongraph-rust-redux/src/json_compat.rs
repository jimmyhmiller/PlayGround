//! JSON compatibility layer
//!
//! This module provides a unified interface for JSON operations regardless of
//! whether the `serde` feature is enabled.
//!
//! - With `serde` feature: re-exports serde_json types
//! - Without `serde` feature: uses our custom zero-dependency implementation

#[cfg(feature = "serde")]
pub use serde_json::{from_str, to_string, to_string_pretty, Value};

#[cfg(not(feature = "serde"))]
pub use crate::json::{from_str, to_string, to_string_pretty, Value};

// Re-export error type
#[cfg(feature = "serde")]
pub type JsonError = serde_json::Error;

#[cfg(not(feature = "serde"))]
pub type JsonError = crate::json::ParseError;

// Re-export the json! macro
#[cfg(feature = "serde")]
pub use serde_json::json;

#[cfg(not(feature = "serde"))]
pub use crate::json;

/// Parse JSON string into a Value
#[cfg(feature = "serde")]
pub fn parse(input: &str) -> Result<Value, JsonError> {
    serde_json::from_str(input)
}

#[cfg(not(feature = "serde"))]
pub fn parse(input: &str) -> Result<Value, JsonError> {
    crate::json::from_str(input)
}

/// Parse JSON string directly into a typed value
#[cfg(feature = "serde")]
pub fn parse_as<T: serde::de::DeserializeOwned>(input: &str) -> Result<T, JsonError> {
    serde_json::from_str(input)
}

#[cfg(not(feature = "serde"))]
pub fn parse_as<T: crate::json::FromJson>(input: &str) -> Result<T, JsonError> {
    let value = crate::json::from_str(input)?;
    T::from_json(&value)
}

/// Convert Value to owned Value (for serde_json compatibility)
#[cfg(feature = "serde")]
pub fn from_value<T: serde::de::DeserializeOwned>(value: Value) -> Result<T, JsonError> {
    serde_json::from_value(value)
}

#[cfg(not(feature = "serde"))]
pub fn from_value<T: crate::json::FromJson>(value: &Value) -> Result<T, JsonError> {
    T::from_json(value)
}

/// Serialize a value to a pretty JSON string
#[cfg(feature = "serde")]
pub fn serialize_pretty<T: serde::Serialize>(value: &T) -> Result<String, JsonError> {
    serde_json::to_string_pretty(value)
}

#[cfg(not(feature = "serde"))]
pub fn serialize_pretty<T: crate::json::ToJson>(value: &T) -> Result<String, JsonError> {
    Ok(value.to_json().to_string_pretty())
}
