//! JSON Value type - zero-dependency replacement for serde_json::Value

use std::collections::HashMap;
use std::fmt;

/// A JSON value representation
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Number(Number),
    String(String),
    Array(Vec<Value>),
    Object(HashMap<String, Value>),
}

/// A JSON number (can be integer or float)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Number {
    Int(i64),
    Float(f64),
}

impl Number {
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Number::Int(n) => Some(*n),
            Number::Float(f) => {
                if f.fract() == 0.0 && *f >= i64::MIN as f64 && *f <= i64::MAX as f64 {
                    Some(*f as i64)
                } else {
                    None
                }
            }
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Number::Int(n) if *n >= 0 => Some(*n as u64),
            Number::Float(f) if f.fract() == 0.0 && *f >= 0.0 && *f <= u64::MAX as f64 => {
                Some(*f as u64)
            }
            _ => None,
        }
    }

    pub fn as_f64(&self) -> f64 {
        match self {
            Number::Int(n) => *n as f64,
            Number::Float(f) => *f,
        }
    }
}

impl From<i64> for Number {
    fn from(n: i64) -> Self {
        Number::Int(n)
    }
}

impl From<u64> for Number {
    fn from(n: u64) -> Self {
        if n <= i64::MAX as u64 {
            Number::Int(n as i64)
        } else {
            Number::Float(n as f64)
        }
    }
}

impl From<f64> for Number {
    fn from(n: f64) -> Self {
        Number::Float(n)
    }
}

impl From<i32> for Number {
    fn from(n: i32) -> Self {
        Number::Int(n as i64)
    }
}

impl From<u32> for Number {
    fn from(n: u32) -> Self {
        Number::Int(n as i64)
    }
}

impl Value {
    /// Returns true if the value is null
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Returns true if the value is a boolean
    pub fn is_bool(&self) -> bool {
        matches!(self, Value::Bool(_))
    }

    /// Returns true if the value is a number
    pub fn is_number(&self) -> bool {
        matches!(self, Value::Number(_))
    }

    /// Returns true if the value is a string
    pub fn is_string(&self) -> bool {
        matches!(self, Value::String(_))
    }

    /// Returns true if the value is an array
    pub fn is_array(&self) -> bool {
        matches!(self, Value::Array(_))
    }

    /// Returns true if the value is an object
    pub fn is_object(&self) -> bool {
        matches!(self, Value::Object(_))
    }

    /// If the value is a boolean, returns it
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// If the value is a number, returns it as i64
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Number(n) => n.as_i64(),
            _ => None,
        }
    }

    /// If the value is a number, returns it as u64
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Number(n) => n.as_u64(),
            _ => None,
        }
    }

    /// If the value is a number, returns it as f64
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Number(n) => Some(n.as_f64()),
            _ => None,
        }
    }

    /// If the value is a string, returns it
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// If the value is an array, returns it
    pub fn as_array(&self) -> Option<&Vec<Value>> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    /// If the value is an array, returns a mutable reference
    pub fn as_array_mut(&mut self) -> Option<&mut Vec<Value>> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    /// If the value is an object, returns it
    pub fn as_object(&self) -> Option<&HashMap<String, Value>> {
        match self {
            Value::Object(o) => Some(o),
            _ => None,
        }
    }

    /// If the value is an object, returns a mutable reference
    pub fn as_object_mut(&mut self) -> Option<&mut HashMap<String, Value>> {
        match self {
            Value::Object(o) => Some(o),
            _ => None,
        }
    }

    /// Get a value from an object by key
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(o) => o.get(key),
            _ => None,
        }
    }

    /// Get a mutable value from an object by key
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Value> {
        match self {
            Value::Object(o) => o.get_mut(key),
            _ => None,
        }
    }

    /// Get a value from an array by index
    pub fn get_index(&self, index: usize) -> Option<&Value> {
        match self {
            Value::Array(a) => a.get(index),
            _ => None,
        }
    }

    /// Takes the value, replacing it with Null
    pub fn take(&mut self) -> Value {
        std::mem::replace(self, Value::Null)
    }
}

// Index by string key for objects
impl std::ops::Index<&str> for Value {
    type Output = Value;

    fn index(&self, key: &str) -> &Self::Output {
        static NULL: Value = Value::Null;
        self.get(key).unwrap_or(&NULL)
    }
}

// IndexMut by string key for objects - creates entry if missing
impl std::ops::IndexMut<&str> for Value {
    fn index_mut(&mut self, key: &str) -> &mut Self::Output {
        match self {
            Value::Object(o) => o.entry(key.to_string()).or_insert(Value::Null),
            _ => panic!("Cannot index non-object with string key"),
        }
    }
}

// Index by usize for arrays
impl std::ops::Index<usize> for Value {
    type Output = Value;

    fn index(&self, index: usize) -> &Self::Output {
        match self {
            Value::Array(a) => &a[index],
            _ => panic!("Cannot index non-array with usize"),
        }
    }
}

// Conversions from Rust types to Value
impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::Bool(b)
    }
}

impl From<i32> for Value {
    fn from(n: i32) -> Self {
        Value::Number(Number::Int(n as i64))
    }
}

impl From<u32> for Value {
    fn from(n: u32) -> Self {
        Value::Number(Number::Int(n as i64))
    }
}

impl From<i64> for Value {
    fn from(n: i64) -> Self {
        Value::Number(Number::Int(n))
    }
}

impl From<u64> for Value {
    fn from(n: u64) -> Self {
        Value::Number(Number::from(n))
    }
}

impl From<f64> for Value {
    fn from(n: f64) -> Self {
        Value::Number(Number::Float(n))
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Value::String(s.to_string())
    }
}

impl From<String> for Value {
    fn from(s: String) -> Self {
        Value::String(s)
    }
}

impl<T: Into<Value>> From<Vec<T>> for Value {
    fn from(v: Vec<T>) -> Self {
        Value::Array(v.into_iter().map(Into::into).collect())
    }
}

impl<T: Into<Value>> From<Option<T>> for Value {
    fn from(opt: Option<T>) -> Self {
        match opt {
            Some(v) => v.into(),
            None => Value::Null,
        }
    }
}

impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Number::Int(n) => write!(f, "{}", n),
            Number::Float(n) => {
                if n.fract() == 0.0 {
                    write!(f, "{}.0", n)
                } else {
                    write!(f, "{}", n)
                }
            }
        }
    }
}
