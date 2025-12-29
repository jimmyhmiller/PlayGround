use crate::gc_runtime::GCRuntime;
use crate::value::Value;
use clojure_reader::edn::{Edn, read_string};
use im::{hashmap, hashset, vector};

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
            let mut result = hashmap! {};
            for (k, v) in map {
                let key = edn_to_value(k)?;
                let value = edn_to_value(v)?;
                result.insert(key, value);
            }
            Ok(Value::Map(result))
        }

        Edn::Set(items) => {
            let mut result = hashset! {};
            for item in items {
                result.insert(edn_to_value(item)?);
            }
            Ok(Value::Set(result))
        }

        // Handle metadata - meta_expr can be Map, Keyword, Symbol, etc.
        Edn::Meta(meta_expr, inner) => {
            // Convert metadata to HashMap<String, Value>
            // ^:keyword becomes {:keyword true}
            // ^{:key val} stays as is
            // ^Symbol becomes {:tag Symbol}
            let mut metadata = hashmap! {};
            match meta_expr.as_ref() {
                Edn::Map(map) => {
                    for (k, v) in map {
                        let key = match k {
                            Edn::Key(s) => s.to_string(),
                            Edn::Symbol(s) => s.to_string(),
                            Edn::Str(s) => s.to_string(),
                            _ => continue, // Skip non-string keys
                        };
                        metadata.insert(key, edn_to_value(v)?);
                    }
                }
                Edn::Key(k) => {
                    // ^:keyword -> {:keyword true}
                    metadata.insert(k.to_string(), Value::Bool(true));
                }
                Edn::Symbol(s) => {
                    // ^Type -> {:tag Type}
                    metadata.insert("tag".to_string(), Value::Symbol(s.to_string()));
                }
                _ => {} // Ignore other metadata forms
            }

            // Convert inner value
            let inner_value = edn_to_value(inner)?;

            Ok(Value::WithMeta(metadata, Box::new(inner_value)))
        }

        // Handle other Edn types as needed
        _ => Err(format!("Unsupported EDN type: {:?}", edn)),
    }
}

// ============================================================================
// Tagged Pointer Reader - produces heap-allocated reader types
// ============================================================================

/// Convert clojure-reader's Edn to tagged heap pointers using GCRuntime
/// This produces ReaderList, ReaderVector, ReaderMap, ReaderSymbol types
/// that can be directly used by macros and the analyzer.
pub fn edn_to_tagged(edn: &Edn, rt: &mut GCRuntime) -> Result<usize, String> {
    match edn {
        Edn::Nil => Ok(7), // nil tagged value

        Edn::Bool(b) => {
            // Boolean tag is 0b011, value is shifted left 3 bits
            // true = 1 << 3 | 0b011 = 0b1011 = 11
            // false = 0 << 3 | 0b011 = 0b0011 = 3
            if *b {
                Ok(0b1011) // true: 1 << 3 | 0b011
            } else {
                Ok(0b0011) // false: 0 << 3 | 0b011
            }
        }

        Edn::Int(i) => {
            // Tag as integer (shift left by 3)
            Ok((*i as usize) << 3)
        }

        Edn::Double(f) => {
            // Allocate float on heap
            rt.allocate_float(f.into_inner())
        }

        Edn::Str(s) => {
            // Allocate string on heap
            rt.allocate_string(s)
        }

        Edn::Symbol(s) => {
            // Special case: "/" is the division operator, not a qualified symbol
            if *s == "/" {
                rt.allocate_reader_symbol(None, "/")
            } else if let Some(slash_pos) = s.find('/') {
                let ns = &s[..slash_pos];
                let name = &s[slash_pos + 1..];
                // If namespace is empty (e.g., "/foo"), treat as unqualified
                if ns.is_empty() {
                    rt.allocate_reader_symbol(None, s)
                } else {
                    rt.allocate_reader_symbol(Some(ns), name)
                }
            } else {
                rt.allocate_reader_symbol(None, s)
            }
        }

        Edn::Key(k) => {
            // Keys in EDN include the ':' prefix, so strip it
            let keyword = k.strip_prefix(':').unwrap_or(k);
            rt.allocate_keyword(keyword)
        }

        Edn::List(items) => {
            // Convert each item and collect as ReaderList
            let mut tagged_items = Vec::with_capacity(items.len());
            for item in items {
                tagged_items.push(edn_to_tagged(item, rt)?);
            }
            rt.allocate_reader_list(&tagged_items)
        }

        Edn::Vector(items) => {
            // Convert each item and collect as ReaderVector
            let mut tagged_items = Vec::with_capacity(items.len());
            for item in items {
                tagged_items.push(edn_to_tagged(item, rt)?);
            }
            rt.allocate_reader_vector(&tagged_items)
        }

        Edn::Map(map) => {
            // Convert each key-value pair and collect as ReaderMap
            let mut entries = Vec::with_capacity(map.len());
            for (k, v) in map {
                let tagged_key = edn_to_tagged(k, rt)?;
                let tagged_value = edn_to_tagged(v, rt)?;
                entries.push((tagged_key, tagged_value));
            }
            rt.allocate_reader_map(&entries)
        }

        Edn::Set(items) => {
            // For now, represent sets as a ReaderVector with a special marker
            // TODO: Add proper ReaderSet type if needed
            // For bootstrap, sets are rare in macro code
            let mut tagged_items = Vec::with_capacity(items.len());
            for item in items {
                tagged_items.push(edn_to_tagged(item, rt)?);
            }
            // Just use ReaderVector for now - sets are rare in reader output
            rt.allocate_reader_vector(&tagged_items)
        }

        // Handle metadata - attach to the inner value
        // meta_expr can be Map, Keyword, Symbol, etc.
        Edn::Meta(meta_expr, inner) => {
            // Convert metadata to a ReaderMap
            // ^:keyword becomes {:keyword true}
            // ^{:key val} stays as is
            // ^Symbol becomes {:tag Symbol}
            let mut meta_entries = Vec::new();
            match meta_expr.as_ref() {
                Edn::Map(map) => {
                    meta_entries.reserve(map.len());
                    for (k, v) in map {
                        let tagged_key = edn_to_tagged(k, rt)?;
                        let tagged_value = edn_to_tagged(v, rt)?;
                        meta_entries.push((tagged_key, tagged_value));
                    }
                }
                Edn::Key(k) => {
                    // ^:keyword -> {:keyword true}
                    let tagged_key = rt.allocate_keyword(k)?;
                    let tagged_value = 0b1011; // true
                    meta_entries.push((tagged_key, tagged_value));
                }
                Edn::Symbol(s) => {
                    // ^Type -> {:tag Type}
                    let tagged_key = rt.allocate_keyword("tag")?;
                    let tagged_value = edn_to_tagged(&Edn::Symbol(s), rt)?;
                    meta_entries.push((tagged_key, tagged_value));
                }
                _ => {} // Ignore other metadata forms
            }
            let meta_ptr = rt.allocate_reader_map(&meta_entries)?;

            // Now convert the inner value and attach metadata if it's a symbol
            match inner.as_ref() {
                Edn::Symbol(s) => {
                    // Special case: "/" is the division operator, not a qualified symbol
                    if *s == "/" {
                        rt.allocate_reader_symbol_with_meta(None, "/", meta_ptr)
                    } else if let Some(slash_pos) = s.find('/') {
                        let ns = &s[..slash_pos];
                        let name = &s[slash_pos + 1..];
                        if ns.is_empty() {
                            rt.allocate_reader_symbol_with_meta(None, s, meta_ptr)
                        } else {
                            rt.allocate_reader_symbol_with_meta(Some(ns), name, meta_ptr)
                        }
                    } else {
                        rt.allocate_reader_symbol_with_meta(None, s, meta_ptr)
                    }
                }
                // For other types, just convert normally (metadata is lost for now)
                // TODO: Add metadata support to other reader types if needed
                _ => edn_to_tagged(inner, rt),
            }
        }

        // Quote: 'form -> (quote form)
        Edn::Quote(inner) => {
            let inner_tagged = edn_to_tagged(inner, rt)?;
            let quote_sym = rt.allocate_reader_symbol(None, "quote")?;
            rt.allocate_reader_list(&[quote_sym, inner_tagged])
        }

        // Syntax-quote (quasiquote): `form -> expanded code
        Edn::SyntaxQuote(inner) => {
            expand_syntax_quote(inner, rt)
        }

        // Unquote: only valid inside syntax-quote, produce a marker
        Edn::Unquote(inner) => {
            let inner_tagged = edn_to_tagged(inner, rt)?;
            let unquote_sym = rt.allocate_reader_symbol(None, "clojure.core/unquote")?;
            rt.allocate_reader_list(&[unquote_sym, inner_tagged])
        }

        // Unquote-splicing: only valid inside syntax-quote, produce a marker
        Edn::UnquoteSplicing(inner) => {
            let inner_tagged = edn_to_tagged(inner, rt)?;
            let unquote_splicing_sym = rt.allocate_reader_symbol(None, "clojure.core/unquote-splicing")?;
            rt.allocate_reader_list(&[unquote_splicing_sym, inner_tagged])
        }

        // Handle other Edn types as needed
        _ => Err(format!("Unsupported EDN type: {:?}", edn)),
    }
}

/// Expand a syntax-quoted form into list/seq/concat calls
/// This implements Clojure's syntax-quote semantics
fn expand_syntax_quote(edn: &Edn, rt: &mut GCRuntime) -> Result<usize, String> {
    match edn {
        // Unquote: ~form -> form (don't quote it)
        Edn::Unquote(inner) => edn_to_tagged(inner, rt),

        // Unquote-splicing: ~@form is only valid in a collection context
        // Here we just pass it through - it will be handled by the collection case
        Edn::UnquoteSplicing(_) => {
            Err("Unquote-splicing (~@) not valid outside of collection in syntax-quote".to_string())
        }

        // List: `(a b ~c ~@d) -> (seq (concat (list 'a) (list 'b) (list c) d))
        Edn::List(items) => {
            expand_syntax_quote_seq(items, rt, true)
        }

        // Vector: `[a b ~c] -> (vec (concat (list 'a) (list 'b) (list c)))
        Edn::Vector(items) => {
            let concat_result = expand_syntax_quote_seq(items, rt, false)?;
            // Wrap in (vec ...)
            // For now, just use apply vector - we may need to adjust
            let vec_sym = rt.allocate_reader_symbol(None, "vec")?;
            rt.allocate_reader_list(&[vec_sym, concat_result])
        }

        // Map: similar treatment
        Edn::Map(entries) => {
            // For maps, we need to expand keys and values
            // `{:a ~b} -> (apply hash-map (concat (list :a) (list b)))
            let mut concat_args = Vec::new();
            for (k, v) in entries {
                let k_expanded = syntax_quote_element(k, rt)?;
                let v_expanded = syntax_quote_element(v, rt)?;
                concat_args.push(k_expanded);
                concat_args.push(v_expanded);
            }
            let concat_sym = rt.allocate_reader_symbol(None, "concat")?;
            let mut concat_list = vec![concat_sym];
            concat_list.extend(concat_args);
            let concat_call = rt.allocate_reader_list(&concat_list)?;

            let apply_sym = rt.allocate_reader_symbol(None, "apply")?;
            let hash_map_sym = rt.allocate_reader_symbol(None, "hash-map")?;
            rt.allocate_reader_list(&[apply_sym, hash_map_sym, concat_call])
        }

        // Symbol: quote it with full namespace resolution
        Edn::Symbol(_s) => {
            // In real Clojure, syntax-quote resolves symbols to their namespaced versions
            // For now, just quote the symbol as-is
            let sym = edn_to_tagged(edn, rt)?;
            let quote_sym = rt.allocate_reader_symbol(None, "quote")?;
            rt.allocate_reader_list(&[quote_sym, sym])
        }

        // Other literals: just quote them
        _ => {
            let tagged = edn_to_tagged(edn, rt)?;
            let quote_sym = rt.allocate_reader_symbol(None, "quote")?;
            rt.allocate_reader_list(&[quote_sym, tagged])
        }
    }
}

/// Expand a sequence (list or vector) inside syntax-quote
/// Returns (seq (concat ...)) for lists, just (concat ...) for vectors
fn expand_syntax_quote_seq(items: &[Edn], rt: &mut GCRuntime, wrap_in_seq: bool) -> Result<usize, String> {
    let mut concat_args = Vec::new();

    for item in items {
        concat_args.push(syntax_quote_element(item, rt)?);
    }

    let concat_sym = rt.allocate_reader_symbol(None, "concat")?;
    let mut concat_list = vec![concat_sym];
    concat_list.extend(concat_args);
    let concat_call = rt.allocate_reader_list(&concat_list)?;

    if wrap_in_seq {
        let seq_sym = rt.allocate_reader_symbol(None, "seq")?;
        rt.allocate_reader_list(&[seq_sym, concat_call])
    } else {
        Ok(concat_call)
    }
}

/// Process a single element inside a syntax-quoted collection
/// Returns a form suitable for use in concat
fn syntax_quote_element(edn: &Edn, rt: &mut GCRuntime) -> Result<usize, String> {
    match edn {
        // ~form -> (list form)
        Edn::Unquote(inner) => {
            let inner_tagged = edn_to_tagged(inner, rt)?;
            let list_sym = rt.allocate_reader_symbol(None, "list")?;
            rt.allocate_reader_list(&[list_sym, inner_tagged])
        }

        // ~@form -> form (spliced directly into concat)
        Edn::UnquoteSplicing(inner) => {
            edn_to_tagged(inner, rt)
        }

        // Nested list: recursively syntax-quote and wrap in (list ...)
        Edn::List(_) => {
            let expanded = expand_syntax_quote(edn, rt)?;
            let list_sym = rt.allocate_reader_symbol(None, "list")?;
            rt.allocate_reader_list(&[list_sym, expanded])
        }

        // Nested vector: recursively syntax-quote and wrap in (list ...)
        Edn::Vector(_) => {
            let expanded = expand_syntax_quote(edn, rt)?;
            let list_sym = rt.allocate_reader_symbol(None, "list")?;
            rt.allocate_reader_list(&[list_sym, expanded])
        }

        // Other forms: quote and wrap in (list ...)
        _ => {
            let expanded = expand_syntax_quote(edn, rt)?;
            let list_sym = rt.allocate_reader_symbol(None, "list")?;
            rt.allocate_reader_list(&[list_sym, expanded])
        }
    }
}

/// Read a Clojure expression from a string, returning a tagged heap pointer
/// This is the new reader entry point that produces reader types directly.
pub fn read_to_tagged(input: &str, rt: &mut GCRuntime) -> Result<usize, String> {
    let preprocessed = preprocess(input);
    match read_string(&preprocessed) {
        Ok(edn) => edn_to_tagged(&edn, rt),
        Err(e) => Err(format!("Parse error: {:?}", e)),
    }
}

/// Check if a character can be part of a keyword/symbol name
fn is_keyword_char(c: char) -> bool {
    c.is_alphanumeric()
        || c == '/'
        || c == '-'
        || c == '_'
        || c == '.'
        || c == '*'
        || c == '+'
        || c == '!'
        || c == '?'
        || c == '<'
        || c == '>'
        || c == '='
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
        assert_eq!(
            read("\"hello\"").unwrap(),
            Value::String("hello".to_string())
        );
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

    // ========================================================================
    // Tests for tagged pointer reader (read_to_tagged)
    // ========================================================================

    fn make_test_runtime() -> GCRuntime {
        GCRuntime::new()
    }

    #[test]
    fn test_read_tagged_scalars() {
        let mut rt = make_test_runtime();

        // nil
        let nil = read_to_tagged("nil", &mut rt).unwrap();
        assert_eq!(nil, 7); // nil tagged value

        // booleans (tag 0b011, value shifted left 3)
        let t = read_to_tagged("true", &mut rt).unwrap();
        assert_eq!(t, 0b1011); // true: 1 << 3 | 0b011 = 11

        let f = read_to_tagged("false", &mut rt).unwrap();
        assert_eq!(f, 0b0011); // false: 0 << 3 | 0b011 = 3

        // integers
        let i = read_to_tagged("42", &mut rt).unwrap();
        assert_eq!(i >> 3, 42); // untagged integer
        assert_eq!(i & 0b111, 0); // integer tag

        let neg = read_to_tagged("-10", &mut rt).unwrap();
        assert_eq!((neg as isize) >> 3, -10);
    }

    #[test]
    fn test_read_tagged_string() {
        let mut rt = make_test_runtime();

        let s = read_to_tagged("\"hello world\"", &mut rt).unwrap();
        // Should be a string (tag 0b010)
        assert_eq!(s & 0b111, 0b010);

        // Read back the string
        let text = rt.read_string(s);
        assert_eq!(text, "hello world");
    }

    #[test]
    fn test_read_tagged_symbol() {
        let mut rt = make_test_runtime();

        // Simple symbol
        let sym = read_to_tagged("foo", &mut rt).unwrap();
        assert!(rt.is_reader_symbol(sym));
        assert_eq!(rt.reader_symbol_name(sym), "foo");
        assert!(rt.reader_symbol_namespace(sym).is_none());

        // Namespaced symbol
        let ns_sym = read_to_tagged("my.ns/bar", &mut rt).unwrap();
        assert!(rt.is_reader_symbol(ns_sym));
        assert_eq!(rt.reader_symbol_name(ns_sym), "bar");
        assert_eq!(rt.reader_symbol_namespace(ns_sym), Some("my.ns".to_string()));
    }

    #[test]
    fn test_read_tagged_keyword() {
        let mut rt = make_test_runtime();

        let kw = read_to_tagged(":foo", &mut rt).unwrap();
        // Keywords have heap object tag (0b110)
        assert_eq!(kw & 0b111, 0b110);

        // Check it's actually a keyword
        let text = rt.get_keyword_text(kw).unwrap();
        assert_eq!(text, "foo");
    }

    #[test]
    fn test_read_tagged_list() {
        let mut rt = make_test_runtime();

        let list = read_to_tagged("(+ 1 2)", &mut rt).unwrap();
        assert!(rt.is_reader_list(list));
        assert_eq!(rt.reader_list_count(list), 3);

        // Check first element is symbol +
        let first = rt.reader_list_first(list);
        assert!(rt.is_reader_symbol(first));
        assert_eq!(rt.reader_symbol_name(first), "+");

        // Check second element is 1
        let second = rt.reader_list_nth(list, 1).unwrap();
        assert_eq!(second >> 3, 1);
    }

    #[test]
    fn test_read_tagged_vector() {
        let mut rt = make_test_runtime();

        let vec = read_to_tagged("[1 2 3]", &mut rt).unwrap();
        assert!(rt.is_reader_vector(vec));
        assert_eq!(rt.reader_vector_count(vec), 3);

        // Check elements
        assert_eq!(rt.reader_vector_nth(vec, 0).unwrap() >> 3, 1);
        assert_eq!(rt.reader_vector_nth(vec, 1).unwrap() >> 3, 2);
        assert_eq!(rt.reader_vector_nth(vec, 2).unwrap() >> 3, 3);
    }

    #[test]
    fn test_read_tagged_map() {
        let mut rt = make_test_runtime();

        let map = read_to_tagged("{:a 1 :b 2}", &mut rt).unwrap();
        assert!(rt.is_reader_map(map));
        assert_eq!(rt.reader_map_count(map), 2);
    }

    #[test]
    fn test_read_tagged_nested() {
        let mut rt = make_test_runtime();

        // Nested structure: (defn foo [x] (+ x 1))
        let form = read_to_tagged("(defn foo [x] (+ x 1))", &mut rt).unwrap();
        assert!(rt.is_reader_list(form));
        assert_eq!(rt.reader_list_count(form), 4);

        // First element should be symbol 'defn'
        let first = rt.reader_list_first(form);
        assert!(rt.is_reader_symbol(first));
        assert_eq!(rt.reader_symbol_name(first), "defn");

        // Third element should be vector [x]
        let params = rt.reader_list_nth(form, 2).unwrap();
        assert!(rt.is_reader_vector(params));
        assert_eq!(rt.reader_vector_count(params), 1);
    }

    #[test]
    fn test_read_tagged_empty_collections() {
        let mut rt = make_test_runtime();

        let empty_list = read_to_tagged("()", &mut rt).unwrap();
        assert!(rt.is_reader_list(empty_list));
        assert_eq!(rt.reader_list_count(empty_list), 0);

        let empty_vec = read_to_tagged("[]", &mut rt).unwrap();
        assert!(rt.is_reader_vector(empty_vec));
        assert_eq!(rt.reader_vector_count(empty_vec), 0);

        let empty_map = read_to_tagged("{}", &mut rt).unwrap();
        assert!(rt.is_reader_map(empty_map));
        assert_eq!(rt.reader_map_count(empty_map), 0);
    }
}
