use crate::value::Value;
use clojure_reader::edn::{Edn, read_string};
use im::{vector, hashmap, hashset};
use std::collections::HashMap;

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

/// Check if a character can be part of a keyword/symbol name
fn is_keyword_char(c: char) -> bool {
    c.is_alphanumeric() || c == '/' || c == '-' || c == '_'
        || c == '.' || c == '*' || c == '+' || c == '!' || c == '?'
        || c == '<' || c == '>' || c == '='
}

/// Pre-process input to handle reader macros not supported by clojure-reader
/// Converts #'symbol to (var symbol)
/// Converts ::foo to :current-ns/foo (auto-resolved keywords)
/// Converts ::alias/foo to :resolved-ns/foo (aliased keywords)
fn preprocess_with_context(
    input: &str,
    current_namespace: &str,
    namespace_aliases: &HashMap<String, String>,
) -> Result<String, String> {
    let mut result = String::with_capacity(input.len() * 2);
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '#' && chars.peek() == Some(&'\'') {
            // Handle #'symbol -> (var symbol)
            chars.next(); // consume the '
            result.push_str("(var ");

            // Read the symbol (including namespace-qualified symbols)
            while let Some(&next) = chars.peek() {
                if is_keyword_char(next) {
                    result.push(chars.next().unwrap());
                } else {
                    break;
                }
            }
            result.push(')');
        } else if c == ':' && chars.peek() == Some(&':') {
            // Handle :: auto-resolved keywords
            chars.next(); // consume second ':'

            // Read the keyword name (may include alias/name)
            let mut keyword_text = String::new();
            while let Some(&next) = chars.peek() {
                if is_keyword_char(next) {
                    keyword_text.push(chars.next().unwrap());
                } else {
                    break;
                }
            }

            if keyword_text.is_empty() {
                return Err("Invalid keyword: :: must be followed by a name".to_string());
            }

            // Check if it's ::alias/name or just ::name
            if let Some(slash_pos) = keyword_text.find('/') {
                let alias = &keyword_text[..slash_pos];
                let name = &keyword_text[slash_pos + 1..];

                // Look up the alias
                if let Some(resolved_ns) = namespace_aliases.get(alias) {
                    result.push(':');
                    result.push_str(resolved_ns);
                    result.push('/');
                    result.push_str(name);
                } else {
                    return Err(format!("Unknown namespace alias: {}", alias));
                }
            } else {
                // Just ::name -> :current-ns/name
                result.push(':');
                result.push_str(current_namespace);
                result.push('/');
                result.push_str(&keyword_text);
            }
        } else {
            result.push(c);
        }
    }

    Ok(result)
}

/// Pre-process input without namespace context (simple version)
/// Only handles #'symbol, not :: keywords
fn preprocess(input: &str) -> String {
    let mut result = String::with_capacity(input.len() * 2);
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '#' && chars.peek() == Some(&'\'') {
            chars.next(); // consume the '
            result.push_str("(var ");

            // Read the symbol (including namespace-qualified symbols)
            while let Some(&next) = chars.peek() {
                if is_keyword_char(next) {
                    result.push(chars.next().unwrap());
                } else {
                    break;
                }
            }
            result.push(')');
        } else {
            result.push(c);
        }
    }

    result
}

/// Read a Clojure expression from a string
pub fn read(input: &str) -> Result<Value, String> {
    let preprocessed = preprocess(input);
    match read_string(&preprocessed) {
        Ok(edn) => edn_to_value(&edn),
        Err(e) => Err(format!("Parse error: {:?}", e)),
    }
}

/// Read a Clojure expression with namespace context for :: keyword resolution
/// - current_namespace: the current namespace name (e.g., "user", "my.app")
/// - namespace_aliases: map of alias -> full namespace name (e.g., {"str" -> "clojure.string"})
pub fn read_with_context(
    input: &str,
    current_namespace: &str,
    namespace_aliases: &HashMap<String, String>,
) -> Result<Value, String> {
    let preprocessed = preprocess_with_context(input, current_namespace, namespace_aliases)?;
    match read_string(&preprocessed) {
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

    #[test]
    fn test_var_reader_macro() {
        // #'foo should become (var foo)
        match read("#'foo").unwrap() {
            Value::List(items) => {
                assert_eq!(items.len(), 2);
                assert!(matches!(&items[0], Value::Symbol(s) if s == "var"));
                assert!(matches!(&items[1], Value::Symbol(s) if s == "foo"));
            }
            _ => panic!("Expected list (var foo)"),
        }

        // #'ns/name should become (var ns/name)
        match read("#'clojure.core/*ns*").unwrap() {
            Value::List(items) => {
                assert_eq!(items.len(), 2);
                assert!(matches!(&items[0], Value::Symbol(s) if s == "var"));
                assert!(matches!(&items[1], Value::Symbol(s) if s == "clojure.core/*ns*"));
            }
            _ => panic!("Expected list (var clojure.core/*ns*)"),
        }
    }

    #[test]
    fn test_namespaced_keywords() {
        // Simple namespaced keyword :ns/name
        match read(":foo/bar").unwrap() {
            Value::Keyword(k) => assert_eq!(k, "foo/bar"),
            _ => panic!("Expected keyword"),
        }

        // Deeply namespaced keyword
        match read(":my.app.core/handler").unwrap() {
            Value::Keyword(k) => assert_eq!(k, "my.app.core/handler"),
            _ => panic!("Expected keyword"),
        }
    }

    #[test]
    fn test_auto_resolved_keywords() {
        let aliases = HashMap::new();

        // ::foo should resolve to :user/foo in user namespace
        match read_with_context("::foo", "user", &aliases).unwrap() {
            Value::Keyword(k) => assert_eq!(k, "user/foo"),
            _ => panic!("Expected keyword"),
        }

        // ::bar in my.app namespace
        match read_with_context("::bar", "my.app", &aliases).unwrap() {
            Value::Keyword(k) => assert_eq!(k, "my.app/bar"),
            _ => panic!("Expected keyword"),
        }
    }

    #[test]
    fn test_aliased_keywords() {
        let mut aliases = HashMap::new();
        aliases.insert("str".to_string(), "clojure.string".to_string());
        aliases.insert("s".to_string(), "clojure.spec.alpha".to_string());

        // ::str/join should resolve to :clojure.string/join
        match read_with_context("::str/join", "user", &aliases).unwrap() {
            Value::Keyword(k) => assert_eq!(k, "clojure.string/join"),
            _ => panic!("Expected keyword"),
        }

        // ::s/keys should resolve to :clojure.spec.alpha/keys
        match read_with_context("::s/keys", "user", &aliases).unwrap() {
            Value::Keyword(k) => assert_eq!(k, "clojure.spec.alpha/keys"),
            _ => panic!("Expected keyword"),
        }
    }

    #[test]
    fn test_unknown_alias_error() {
        let aliases = HashMap::new();

        // ::unknown/foo should error
        let result = read_with_context("::unknown/foo", "user", &aliases);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown namespace alias"));
    }

    #[test]
    fn test_keywords_in_data_structures() {
        let mut aliases = HashMap::new();
        aliases.insert("k".to_string(), "my.keys".to_string());

        // Map with auto-resolved keywords
        match read_with_context("{::name \"Alice\" ::k/id 42}", "user", &aliases).unwrap() {
            Value::Map(map) => {
                assert_eq!(map.len(), 2);
                let key_name = Value::Keyword("user/name".to_string());
                let key_id = Value::Keyword("my.keys/id".to_string());
                assert_eq!(map.get(&key_name), Some(&Value::String("Alice".to_string())));
                assert_eq!(map.get(&key_id), Some(&Value::Int(42)));
            }
            _ => panic!("Expected map"),
        }
    }
}
