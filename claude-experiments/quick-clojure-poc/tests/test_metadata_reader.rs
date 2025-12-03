// Test to check if clojure-reader parses metadata
use clojure_reader::edn::{Edn, read_string};

#[test]
fn test_simple_symbol() {
    let result = read_string("foo").unwrap();
    println!("foo => {:?}", result);
    assert!(matches!(result, Edn::Symbol("foo")));
}

#[test]
fn test_metadata_shorthand() {
    let result = read_string("^:dynamic foo");
    println!("^:dynamic foo => {:?}", result);

    if let Ok(edn) = result {
        match &edn {
            Edn::Tagged(tag, value) => {
                println!("  Got Tagged! tag='{}', value={:?}", tag, value);
                assert_eq!(*tag, "dynamic" , "Expected tag to be 'dynamic'");
            }
            _ => {
                println!("  Not Tagged, got variant: {}", match edn {
                    Edn::Vector(_) => "Vector",
                    Edn::Set(_) => "Set",
                    Edn::Map(_) => "Map",
                    Edn::List(_) => "List",
                    Edn::Key(_) => "Key",
                    Edn::Symbol(_) => "Symbol",
                    Edn::Str(_) => "Str",
                    Edn::Int(_) => "Int",
                    Edn::Tagged(_, _) => "Tagged",
                    Edn::Char(_) => "Char",
                    Edn::Bool(_) => "Bool",
                    Edn::Nil => "Nil",
                    _ => "Other",
                });
                panic!("Expected Tagged variant, got something else");
            }
        }
    } else {
        panic!("Failed to parse ^:dynamic foo");
    }
}

#[test]
fn test_metadata_full_map() {
    let result = read_string("^{:dynamic true} foo");
    println!("^{{:dynamic true}} foo => {:?}", result);

    // This should work regardless of whether metadata is supported
    assert!(result.is_ok(), "Should parse without error");
}

#[test]
fn test_def_with_metadata() {
    let result = read_string("(def ^:dynamic x 10)");
    println!("(def ^:dynamic x 10) => {:?}", result);
    assert!(result.is_ok(), "Should parse without error");
}
