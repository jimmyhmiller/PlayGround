// Test to check if clojure-reader parses metadata
use clojure_reader::edn::{Edn, read_string};

fn main() {
    // Test 1: Simple symbol
    println!("Test 1: Simple symbol");
    let result = read_string("foo");
    println!("  foo => {:?}", result);

    // Test 2: ^:dynamic metadata (shorthand)
    println!("\nTest 2: ^:dynamic foo");
    let result = read_string("^:dynamic foo");
    println!("  ^:dynamic foo => {:?}", result);

    // Test 3: Full metadata map
    println!("\nTest 3: ^{{:dynamic true}} foo");
    let result = read_string("^{:dynamic true} foo");
    println!("  ^{{:dynamic true}} foo => {:?}", result);

    // Test 4: def with metadata
    println!("\nTest 4: (def ^:dynamic x 10)");
    let result = read_string("(def ^:dynamic x 10)");
    println!("  (def ^:dynamic x 10) => {:?}", result);

    // Test 5: Check Edn enum variants
    println!("\nTest 5: Checking for Tagged/Meta variants");
    if let Ok(edn) = read_string("^:dynamic foo") {
        match &edn {
            Edn::Tagged(tag, value) => {
                println!("  Got Tagged! tag='{}', value={:?}", tag, value);
            }
            _ => {
                println!("  Not Tagged, got: {:?}", edn);
            }
        }
    }
}
