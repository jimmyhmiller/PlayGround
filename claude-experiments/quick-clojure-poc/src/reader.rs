use crate::value::Value;
use clojure_reader::edn::{Edn, read_string};
use im::{vector, hashmap, hashset, HashMap};

/// Metadata attached to a value
/// Used for ^:dynamic and other metadata annotations
#[derive(Debug, Clone, PartialEq)]
pub struct WithMetadata {
    pub value: Value,
    pub metadata: HashMap<String, Value>,
}

/// Convert clojure-reader's Edn to our Value representation
pub fn edn_to_value(edn: &Edn) -> Result<Value, String> {
    match edn {
        Edn::Nil => Ok(Value::Nil),
        Edn::Bool(b) => Ok(Value::Bool(*b)),
        Edn::Int(i) => Ok(Value::Int(*i)),
        Edn::Double(f) => Ok(Value::Float(f.into_inner())),
        Edn::Str(s) => Ok(Value::String(s.to_string())),
        Edn::Symbol(s) => Ok(Value::Symbol(s.to_string())),
        Edn::Key(k) => {
            // Keys in EDN include the ':' prefix, so strip it
            let keyword = k.strip_prefix(':').unwrap_or(k);
            Ok(Value::Keyword(keyword.to_string()))
        }

        Edn::List(items) => {
            let mut values = vector![];
            for item in items {
                values.push_back(edn_to_value(item)?);
            }
            Ok(Value::List(values))
        }

        Edn::Vector(items) => {
            let mut values = vector![];
            for item in items {
                values.push_back(edn_to_value(item)?);
            }
            Ok(Value::Vector(values))
        }

        Edn::Map(map) => {
            let mut result = hashmap!{};
            for (k, v) in map {
                let key = edn_to_value(k)?;
                let value = edn_to_value(v)?;
                result.insert(key, value);
            }
            Ok(Value::Map(result))
        }

        Edn::Set(items) => {
            let mut result = hashset!{};
            for item in items {
                result.insert(edn_to_value(item)?);
            }
            Ok(Value::Set(result))
        }

        // Handle metadata
        Edn::Meta(meta_map, inner) => {
            // Convert metadata map to HashMap<String, Value>
            let mut metadata = hashmap!{};
            for (k, v) in meta_map {
                let key = match k {
                    Edn::Key(s) => s.to_string(),
                    Edn::Symbol(s) => s.to_string(),
                    Edn::Str(s) => s.to_string(),
                    _ => continue, // Skip non-string keys
                };
                metadata.insert(key, edn_to_value(v)?);
            }

            // Convert inner value
            let inner_value = edn_to_value(inner)?;

            Ok(Value::WithMeta(metadata, Box::new(inner_value)))
        }

        // Handle other Edn types as needed
        _ => Err(format!("Unsupported EDN type: {:?}", edn)),
    }
}

/// Read a Clojure expression from a string
pub fn read(input: &str) -> Result<Value, String> {
    match read_string(input) {
        Ok(edn) => edn_to_value(&edn),
        Err(e) => Err(format!("Parse error: {:?}", e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_scalars() {
        assert_eq!(read("nil").unwrap(), Value::Nil);
        assert_eq!(read("true").unwrap(), Value::Bool(true));
        assert_eq!(read("false").unwrap(), Value::Bool(false));
        assert_eq!(read("42").unwrap(), Value::Int(42));
        assert_eq!(read("3.14").unwrap(), Value::Float(3.14));
        assert_eq!(read("\"hello\"").unwrap(), Value::String("hello".to_string()));
    }

    #[test]
    fn test_read_symbols_and_keywords() {
        match read("foo").unwrap() {
            Value::Symbol(s) => assert_eq!(s, "foo"),
            _ => panic!("Expected symbol"),
        }

        match read(":bar").unwrap() {
            Value::Keyword(k) => assert_eq!(k, "bar"),
            _ => panic!("Expected keyword"),
        }
    }

    #[test]
    fn test_read_list() {
        match read("(+ 1 2)").unwrap() {
            Value::List(items) => {
                assert_eq!(items.len(), 3);
                assert!(matches!(&items[0], Value::Symbol(s) if s == "+"));
                assert!(matches!(&items[1], Value::Int(1)));
                assert!(matches!(&items[2], Value::Int(2)));
            }
            _ => panic!("Expected list"),
        }
    }

    #[test]
    fn test_read_vector() {
        match read("[1 2 3]").unwrap() {
            Value::Vector(items) => {
                assert_eq!(items.len(), 3);
                assert!(matches!(&items[0], Value::Int(1)));
                assert!(matches!(&items[1], Value::Int(2)));
                assert!(matches!(&items[2], Value::Int(3)));
            }
            _ => panic!("Expected vector"),
        }
    }

    #[test]
    fn test_read_map() {
        match read("{:a 1 :b 2}").unwrap() {
            Value::Map(map) => {
                assert_eq!(map.len(), 2);
                let key_a = Value::Keyword("a".to_string());
                assert_eq!(map.get(&key_a), Some(&Value::Int(1)));
            }
            _ => panic!("Expected map"),
        }
    }

    #[test]
    fn test_read_nested() {
        // [1 2 {:a 3}]
        match read("[1 2 {:a 3}]").unwrap() {
            Value::Vector(items) => {
                assert_eq!(items.len(), 3);
                assert!(matches!(&items[0], Value::Int(1)));
                assert!(matches!(&items[1], Value::Int(2)));
                assert!(matches!(&items[2], Value::Map(_)));
            }
            _ => panic!("Expected vector"),
        }
    }
}
