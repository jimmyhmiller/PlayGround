//! Zero-dependency JSON parsing and serialization
//!
//! This module provides a complete JSON implementation without external dependencies.
//! It is used when the `serde` feature is disabled.

mod from_json;
mod macros;
mod parser;
mod to_json;
mod value;

pub use from_json::{parse_error, FromJson, JsonObjectExt};
pub use macros::json;
pub use parser::{from_str, ParseError};
pub use to_json::{to_string, to_string_pretty, ToJson};
pub use value::{Number, Value};

/// Parse a JSON string and deserialize into a type
pub fn from_str_as<T: FromJson>(input: &str) -> Result<T, ParseError> {
    let value = from_str(input)?;
    T::from_json(&value)
}

/// Convert a value implementing ToJson to a JSON Value
pub fn to_value<T: ToJson>(value: &T) -> Value {
    value.to_json()
}
