//! json! macro for constructing JSON values

/// Macro for constructing JSON values inline
///
/// # Examples
///
/// ```
/// use iongraph_rust_redux::json::json;
///
/// let value = json!({
///     "name": "test",
///     "count": 42,
///     "items": [1, 2, 3],
///     "nested": {
///         "flag": true
///     }
/// });
/// ```
#[macro_export]
macro_rules! json {
    // null
    (null) => {
        $crate::json::Value::Null
    };

    // boolean
    (true) => {
        $crate::json::Value::Bool(true)
    };
    (false) => {
        $crate::json::Value::Bool(false)
    };

    // array
    ([]) => {
        $crate::json::Value::Array(::std::vec::Vec::new())
    };
    ([ $($elem:tt),* $(,)? ]) => {
        $crate::json::Value::Array(::std::vec![ $( $crate::json!($elem) ),* ])
    };

    // object
    ({}) => {
        $crate::json::Value::Object(::std::collections::HashMap::new())
    };
    ({ $($key:tt : $value:tt),* $(,)? }) => {{
        let mut map = ::std::collections::HashMap::new();
        $(
            map.insert($key.to_string(), $crate::json!($value));
        )*
        $crate::json::Value::Object(map)
    }};

    // string literals
    ($s:literal) => {{
        // Try to handle both string and numeric literals
        $crate::json::Value::from($s)
    }};

    // expressions (variables, function calls, etc.)
    ($e:expr) => {
        $crate::json::Value::from($e)
    };
}

// Re-export the macro
pub use json;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::Value;

    #[test]
    fn test_null() {
        assert_eq!(json!(null), Value::Null);
    }

    #[test]
    fn test_bool() {
        assert_eq!(json!(true), Value::Bool(true));
        assert_eq!(json!(false), Value::Bool(false));
    }

    #[test]
    fn test_number() {
        assert_eq!(json!(42).as_i64(), Some(42));
        assert_eq!(json!(3.14).as_f64(), Some(3.14));
    }

    #[test]
    fn test_string() {
        assert_eq!(json!("hello").as_str(), Some("hello"));
    }

    #[test]
    fn test_array() {
        let arr = json!([1, 2, 3]);
        assert!(arr.is_array());
        assert_eq!(arr.as_array().unwrap().len(), 3);
    }

    #[test]
    fn test_empty_array() {
        let arr = json!([]);
        assert!(arr.is_array());
        assert_eq!(arr.as_array().unwrap().len(), 0);
    }

    #[test]
    fn test_object() {
        let obj = json!({
            "name": "test",
            "count": 42
        });
        assert!(obj.is_object());
        assert_eq!(obj.get("name").unwrap().as_str(), Some("test"));
        assert_eq!(obj.get("count").unwrap().as_i64(), Some(42));
    }

    #[test]
    fn test_empty_object() {
        let obj = json!({});
        assert!(obj.is_object());
        assert_eq!(obj.as_object().unwrap().len(), 0);
    }

    #[test]
    fn test_nested() {
        let value = json!({
            "items": [1, 2, 3],
            "nested": {
                "flag": true
            }
        });
        assert!(value.get("items").unwrap().is_array());
        assert!(value.get("nested").unwrap().is_object());
        assert_eq!(
            value.get("nested").unwrap().get("flag").unwrap().as_bool(),
            Some(true)
        );
    }

    #[test]
    fn test_variable() {
        let x = 42;
        let s = "hello".to_string();
        assert_eq!(json!(x).as_i64(), Some(42));
        assert_eq!(json!(s).as_str(), Some("hello"));
    }
}
