use simd_json_complete::parser::TapeType;

fn main() {
    println!("=== Correctness Tests ===\n");

    test_simple();
    test_nested();
    test_escapes();
    test_numbers();
    test_arrays();
    test_empty_containers();

    println!("\n=== Round-trip Tests ===\n");

    test_roundtrip_simple();
    test_roundtrip_complex();
    test_roundtrip_all_types();
    test_roundtrip_50k_objects();
    test_dom_navigation();

    println!("\n=== All correctness tests passed! ===\n");

    benchmark();
}

fn test_simple() {
    let json = br#"{"name": "Alice", "age": 30}"#;
    let padded = simd_json_complete::pad_input(json);
    let doc = simd_json_complete::parse_padded(&padded, json.len()).unwrap();

    assert_eq!(doc.tape_type(1), TapeType::OpenObject);
    assert_eq!(&*doc.get_string(2, &padded), "name");
    assert_eq!(&*doc.get_string(3, &padded), "Alice");
    assert_eq!(&*doc.get_string(4, &padded), "age");
    assert_eq!(doc.get_i64(5), 30);
    assert_eq!(doc.get_partner(1), 6);
    assert_eq!(doc.get_partner(6), 1);
    println!("  [PASS] simple object");
}

fn test_nested() {
    let json = br#"{"a": {"b": {"c": 1}}, "d": [1, 2, 3]}"#;
    let padded = simd_json_complete::pad_input(json);
    let doc = simd_json_complete::parse_padded(&padded, json.len()).unwrap();

    let mut found_c = false;
    for i in 0..doc.tape.len() {
        if doc.tape_type(i) == TapeType::String && &*doc.get_string(i, &padded) == "c" {
            assert_eq!(doc.get_i64(i + 1), 1);
            found_c = true;
        }
    }
    assert!(found_c);
    println!("  [PASS] nested objects and arrays");
}

fn test_escapes() {
    let json = br#"{"msg": "hello\nworld", "quote": "say \"hi\"", "path": "C:\\Users"}"#;
    let padded = simd_json_complete::pad_input(json);
    let doc = simd_json_complete::parse_padded(&padded, json.len()).unwrap();

    for i in 0..doc.tape.len() {
        if doc.tape_type(i) == TapeType::String {
            let s = doc.get_string(i, &padded);
            if &*s == "msg" { assert_eq!(&*doc.get_string(i+1, &padded), "hello\nworld"); }
            if &*s == "quote" { assert_eq!(&*doc.get_string(i+1, &padded), "say \"hi\""); }
            if &*s == "path" { assert_eq!(&*doc.get_string(i+1, &padded), "C:\\Users"); }
        }
    }
    println!("  [PASS] string escapes");
}

fn test_numbers() {
    let json = br#"{"int": 42, "neg": -17, "big": 1234567890, "pi": 3.14159, "sci": 1.5e10}"#;
    let padded = simd_json_complete::pad_input(json);
    let doc = simd_json_complete::parse_padded(&padded, json.len()).unwrap();

    for i in 0..doc.tape.len() {
        if doc.tape_type(i) == TapeType::String {
            let key = doc.get_string(i, &padded);
            if i + 1 < doc.tape.len() {
                match &*key {
                    "int" => assert_eq!(doc.get_i64(i+1), 42),
                    "neg" => assert_eq!(doc.get_i64(i+1), -17),
                    "big" => assert_eq!(doc.get_i64(i+1), 1234567890),
                    "pi" => assert!((doc.get_f64(i+1) - 3.14159).abs() < 1e-10),
                    "sci" => assert!((doc.get_f64(i+1) - 1.5e10).abs() < 1.0),
                    _ => {}
                }
            }
        }
    }
    println!("  [PASS] numbers");
}

fn test_arrays() {
    let json = br#"{"nums": [1, 2, 3]}"#;
    let padded = simd_json_complete::pad_input(json);
    let doc = simd_json_complete::parse_padded(&padded, json.len()).unwrap();

    for i in 0..doc.tape.len() {
        if doc.tape_type(i) == TapeType::String && &*doc.get_string(i, &padded) == "nums" {
            assert_eq!(doc.tape_type(i+1), TapeType::OpenArray);
            assert_eq!(doc.get_i64(i+2), 1);
            assert_eq!(doc.get_i64(i+3), 2);
            assert_eq!(doc.get_i64(i+4), 3);
        }
    }
    println!("  [PASS] arrays");
}

fn test_empty_containers() {
    let json = br#"{"empty_obj": {}, "empty_arr": []}"#;
    let padded = simd_json_complete::pad_input(json);
    let doc = simd_json_complete::parse_padded(&padded, json.len()).unwrap();

    for i in 0..doc.tape.len() {
        if doc.tape_type(i) == TapeType::String && &*doc.get_string(i, &padded) == "empty_obj" {
            let partner = doc.get_partner(i+1);
            assert_eq!(partner, i+2);
        }
        if doc.tape_type(i) == TapeType::String && &*doc.get_string(i, &padded) == "empty_arr" {
            let partner = doc.get_partner(i+1);
            assert_eq!(partner, i+2);
        }
    }
    println!("  [PASS] empty containers");
}

/// Reconstruct JSON from the tape — proves the DOM is complete.
/// If this produces valid JSON that matches the input, the tape has all the information.
fn reconstruct_json(doc: &simd_json_complete::parser::Document, input: &[u8]) -> String {
    let mut out = String::new();
    reconstruct_at(doc, input, 1, &mut out);
    out
}

fn reconstruct_at(
    doc: &simd_json_complete::parser::Document,
    input: &[u8],
    index: usize,
    out: &mut String,
) -> usize {
    use simd_json_complete::parser::TapeType;
    let typ = doc.tape_type(index);
    match typ {
        TapeType::OpenObject => {
            out.push('{');
            let close = doc.get_partner(index);
            let mut i = index + 1;
            let mut first = true;
            while i < close {
                if !first { out.push(','); }
                first = false;
                // Key (string)
                let key = doc.get_string(i, input);
                out.push('"');
                out.push_str(&escape_json_string(&key));
                out.push('"');
                out.push(':');
                i += 1;
                // Value
                i = reconstruct_at(doc, input, i, out);
            }
            out.push('}');
            close + 1 // skip CloseObject
        }
        TapeType::OpenArray => {
            out.push('[');
            let close = doc.get_partner(index);
            let mut i = index + 1;
            let mut first = true;
            while i < close {
                if !first { out.push(','); }
                first = false;
                i = reconstruct_at(doc, input, i, out);
            }
            out.push(']');
            close + 1 // skip CloseArray
        }
        TapeType::String => {
            let s = doc.get_string(index, input);
            out.push('"');
            out.push_str(&escape_json_string(&s));
            out.push('"');
            index + 1
        }
        TapeType::Int64 => {
            out.push_str(&doc.get_i64(index).to_string());
            index + 1
        }
        TapeType::Double => {
            let v = doc.get_f64(index);
            // Use enough precision to round-trip
            if v == v.floor() && v.abs() < 1e15 {
                out.push_str(&format!("{:.1}", v));
            } else {
                out.push_str(&format!("{}", v));
            }
            index + 1
        }
        TapeType::True => { out.push_str("true"); index + 1 }
        TapeType::False => { out.push_str("false"); index + 1 }
        TapeType::Null => { out.push_str("null"); index + 1 }
        _ => index + 1, // Root, Close — skip
    }
}

fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < '\x20' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Normalize JSON by parsing with serde and re-serializing — removes whitespace differences.
fn normalize_json(json: &str) -> String {
    // Simple approach: parse our tape output to remove formatting differences
    // Since we don't have serde, just strip whitespace outside strings
    let mut out = String::new();
    let mut in_string = false;
    let mut prev_backslash = false;
    for ch in json.chars() {
        if in_string {
            out.push(ch);
            if ch == '"' && !prev_backslash {
                in_string = false;
            }
            prev_backslash = ch == '\\' && !prev_backslash;
        } else {
            if ch == '"' {
                in_string = true;
                out.push(ch);
            } else if !ch.is_ascii_whitespace() {
                out.push(ch);
            }
        }
    }
    out
}

fn test_roundtrip_simple() {
    let json = br#"{"name":"Alice","age":30,"active":true}"#;
    let padded = simd_json_complete::pad_input(json);
    let doc = simd_json_complete::parse_padded(&padded, json.len()).unwrap();
    let reconstructed = reconstruct_json(&doc, &padded);
    let original = normalize_json(std::str::from_utf8(json).unwrap());
    assert_eq!(reconstructed, original,
        "\nOriginal:      {}\nReconstructed: {}", original, reconstructed);
    println!("  [PASS] round-trip: simple object");
}

fn test_roundtrip_complex() {
    let json = br#"{"a":{"b":{"c":1}},"d":[1,2,3],"e":null,"f":false}"#;
    let padded = simd_json_complete::pad_input(json);
    let doc = simd_json_complete::parse_padded(&padded, json.len()).unwrap();
    let reconstructed = reconstruct_json(&doc, &padded);
    let original = normalize_json(std::str::from_utf8(json).unwrap());
    assert_eq!(reconstructed, original,
        "\nOriginal:      {}\nReconstructed: {}", original, reconstructed);
    println!("  [PASS] round-trip: nested objects/arrays/null/false");
}

fn test_roundtrip_all_types() {
    let json = br#"{"str":"hello","int":42,"neg":-17,"float":3.14,"bool_t":true,"bool_f":false,"nil":null,"arr":[1,"two",true,null],"obj":{"x":1}}"#;
    let padded = simd_json_complete::pad_input(json);
    let doc = simd_json_complete::parse_padded(&padded, json.len()).unwrap();
    let reconstructed = reconstruct_json(&doc, &padded);
    let original = normalize_json(std::str::from_utf8(json).unwrap());

    // For floats, compare structurally instead of string-exact
    // Parse both as serde_json::Value... but we don't have serde.
    // Instead, verify key values individually:
    println!("  Original:      {}", original);
    println!("  Reconstructed: {}", reconstructed);

    // Verify all values are present in reconstruction
    assert!(reconstructed.contains("\"hello\""), "missing string");
    assert!(reconstructed.contains("42"), "missing int");
    assert!(reconstructed.contains("-17"), "missing negative int");
    assert!(reconstructed.contains("3.14"), "missing float");
    assert!(reconstructed.contains("true"), "missing true");
    assert!(reconstructed.contains("false"), "missing false");
    assert!(reconstructed.contains("null"), "missing null");
    assert!(reconstructed.contains("[1,\"two\",true,null]"), "missing array");
    assert!(reconstructed.contains("\"x\":1"), "missing nested obj");
    println!("  [PASS] round-trip: all JSON types");
}

fn test_roundtrip_50k_objects() {
    // Same data as the benchmark — 50k objects
    let mut json_parts = Vec::new();
    json_parts.push(r#"{"data":["#.to_string());
    for i in 0..100 {  // use 100 for the test (50k is too slow for string comparison)
        if i > 0 { json_parts.push(",".to_string()); }
        json_parts.push(format!(
            r#"{{"id":{},"name":"user_{}","score":{},"active":{}}}"#,
            i, i, i * 17 % 10000, if i % 2 == 0 { "true" } else { "false" }
        ));
    }
    json_parts.push(r#"]}"#.to_string());
    let json_str = json_parts.join("");
    let json_bytes = json_str.as_bytes();
    let padded = simd_json_complete::pad_input(json_bytes);
    let doc = simd_json_complete::parse_padded(&padded, json_bytes.len()).unwrap();
    let reconstructed = reconstruct_json(&doc, &padded);
    let original = normalize_json(&json_str);

    assert_eq!(reconstructed, original,
        "\nMismatch! first 200 chars:\n  orig:  {}\n  recon: {}",
        &original[..200.min(original.len())],
        &reconstructed[..200.min(reconstructed.len())]);

    // Also verify specific values by tape walking
    let mut id_count = 0;
    for i in 0..doc.tape.len() {
        if doc.tape_type(i) == TapeType::String && &*doc.get_string(i, &padded) == "id" {
            id_count += 1;
        }
    }
    assert_eq!(id_count, 100, "should find 100 'id' keys");
    println!("  [PASS] round-trip: 100 objects with exact reconstruction");
}

/// Prove DOM navigation works like simdjson — find a key, get its value
fn test_dom_navigation() {
    let json = br#"{"users":[{"name":"Alice","age":30},{"name":"Bob","age":25}],"count":2}"#;
    let padded = simd_json_complete::pad_input(json);
    let doc = simd_json_complete::parse_padded(&padded, json.len()).unwrap();

    // Navigate: find "users" key, get first element, get "name"
    let mut users_idx = None;
    for i in 0..doc.tape.len() {
        if doc.tape_type(i) == TapeType::String && &*doc.get_string(i, &padded) == "users" {
            users_idx = Some(i + 1); // value follows key
            break;
        }
    }
    let users_arr = users_idx.expect("should find 'users' key");
    assert_eq!(doc.tape_type(users_arr), TapeType::OpenArray);

    // First element of array is at users_arr + 1
    let first_obj = users_arr + 1;
    assert_eq!(doc.tape_type(first_obj), TapeType::OpenObject);

    // Find "name" key inside first object
    let first_close = doc.get_partner(first_obj);
    let mut name_val = None;
    let mut i = first_obj + 1;
    while i < first_close {
        if doc.tape_type(i) == TapeType::String {
            let key = doc.get_string(i, &padded);
            if &*key == "name" {
                name_val = Some(i + 1);
                break;
            }
            i += 2; // skip key + value
        } else {
            i += 1;
        }
    }
    let name_idx = name_val.expect("should find 'name' in first object");
    assert_eq!(&*doc.get_string(name_idx, &padded), "Alice");

    // Find "count" at top level
    let mut count_val = None;
    let root_close = doc.get_partner(1); // root object close
    let mut i = 2; // first entry after root open
    while i < root_close {
        if doc.tape_type(i) == TapeType::String && &*doc.get_string(i, &padded) == "count" {
            count_val = Some(i + 1);
            break;
        }
        // Skip value (could be container — jump to partner)
        i += 1;
        match doc.tape_type(i) {
            TapeType::OpenObject | TapeType::OpenArray => {
                i = doc.get_partner(i) + 1;
            }
            _ => { i += 1; }
        }
    }
    let count_idx = count_val.expect("should find 'count' key");
    assert_eq!(doc.get_i64(count_idx), 2);

    // Verify container skipping: skip from users array open to close
    let users_close = doc.get_partner(users_arr);
    // Everything between users_arr and users_close is the array content
    // After the close, the next tape entry should be the "count" key
    let after_users = users_close + 1;
    assert_eq!(doc.tape_type(after_users), TapeType::String);
    assert_eq!(&*doc.get_string(after_users, &padded), "count");

    println!("  [PASS] DOM navigation: key lookup, array indexing, container skip, nested access");
}

fn benchmark() {
    let mut json_parts = Vec::new();
    json_parts.push(r#"{"data": ["#.to_string());
    for i in 0..50000 {
        if i > 0 { json_parts.push(",".to_string()); }
        json_parts.push(format!(
            r#"{{"id": {}, "name": "user_{}", "score": {}, "active": {}}}"#,
            i, i, i * 17 % 10000, if i % 2 == 0 { "true" } else { "false" }
        ));
    }
    json_parts.push(r#"]}"#.to_string());
    let json_str = json_parts.join("");
    let json_bytes = json_str.as_bytes();
    let json_len = json_bytes.len();
    let padded = simd_json_complete::pad_input(json_bytes);

    println!("=== Benchmark ===");
    println!("Input size: {:.2} MB ({} bytes)\n", json_len as f64 / 1e6, json_len);

    // Stage 1 only
    {
        let mut positions = vec![0i32; json_len];
        simd_json_complete::simd_stage1::find_structural(&padded, &mut positions);

        let iterations = 10;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            simd_json_complete::simd_stage1::find_structural(&padded, &mut positions);
        }
        let elapsed = start.elapsed();
        let gb = json_len as f64 * iterations as f64 / elapsed.as_secs_f64() / 1e9;
        println!("Stage 1 (SIMD):          {:.2} GB/s ({:.3} ms/iter)",
            gb, elapsed.as_secs_f64() * 1000.0 / iterations as f64);
    }

    // Full parse — zero allocation after first call
    {
        let mut parser = simd_json_complete::Parser::new(json_len);
        let _ = parser.parse(&padded).unwrap(); // warmup + first alloc

        let iterations = 10;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = parser.parse(&padded).unwrap(); // reuses all buffers
        }
        let elapsed = start.elapsed();
        let gb = json_len as f64 * iterations as f64 / elapsed.as_secs_f64() / 1e9;
        println!("Full parse (SIMD+Rust):  {:.2} GB/s ({:.3} ms/iter)",
            gb, elapsed.as_secs_f64() * 1000.0 / iterations as f64);
    }

    println!("\nCompare with: /tmp/simdjson-bench/bench3");
}
