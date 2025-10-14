fn main() {
    // Test defpdl-pattern parsing
    let test_defpdl = std::fs::read_to_string("/tmp/test_defpdl.lisp").unwrap();
    println!("Test defpdl file ({} bytes):", test_defpdl.len());
    println!("Content: {:?}", test_defpdl);
    match mlir_lisp::parser::parse(&test_defpdl) {
        Ok((rest, vals)) => {
            println!("  Parsed {} forms", vals.len());
            if !rest.is_empty() {
                println!("  Remaining ({} bytes): {:?}", rest.len(), &rest[..rest.len().min(50)]);
            }
        }
        Err(e) => println!("  Error: {:?}", e),
    }

    // Read the actual file
    let file_content = std::fs::read_to_string("calc_simple.lisp").unwrap();

    println!("\n=== ACTUAL FILE ===");
    println!("File content length: {} bytes", file_content.len());
    println!("File starts with: {:?}", &file_content[0..50]);

    // Check what's at position 318 (after first form)
    let pos = file_content.find("))\n").unwrap() + 2;
    println!("Position of first '))'+'\\n': {}", pos);
    println!("Bytes at that position: {:?}", &file_content[pos..pos.min(pos+30)]);

    // Test minimal case
    let test1 = "(foo (bar))";
    println!("\nTest 1: {:?}", test1);
    match mlir_lisp::parser::parse(test1) {
        Ok((rest, vals)) => println!("  Parsed {} forms, remaining: {:?}", vals.len(), rest),
        Err(e) => println!("  Error: {:?}", e),
    }

    let test2 = "(foo (bar))\n(baz)";
    println!("\nTest 2 (one newline): {:?}", test2);
    match mlir_lisp::parser::parse(test2) {
        Ok((rest, vals)) => {
            println!("  Parsed {} forms", vals.len());
            if !rest.is_empty() {
                println!("  Remaining: {:?}", rest);
            }
        }
        Err(e) => println!("  Error: {:?}", e),
    }

    let test3 = "(foo (bar))\n\n(baz)";
    println!("\nTest 3 (two newlines): {:?}", test3);
    match mlir_lisp::parser::parse(test3) {
        Ok((rest, vals)) => {
            println!("  Parsed {} forms", vals.len());
            if !rest.is_empty() {
                println!("  Remaining: {:?}", rest);
            }
        }
        Err(e) => println!("  Error: {:?}", e),
    }

    // Try parsing just the remaining part
    let remaining_start = 320;
    let remaining_part = &file_content[remaining_start..];
    println!("\n=== REMAINING PART ONLY (from byte {}) ===", remaining_start);
    println!("Content ({} bytes): {:?}...", remaining_part.len(), &remaining_part[..remaining_part.len().min(100)]);
    match mlir_lisp::parser::parse(remaining_part) {
        Ok((rest, vals)) => {
            println!("  Parsed {} forms from remaining", vals.len());
            if !rest.is_empty() {
                println!("  Still remaining: {} bytes", rest.len());
            }
        }
        Err(e) => println!("  Error: {:?}", e),
    }

    match mlir_lisp::parser::parse(&file_content) {
        Ok((rest, values)) => {
            println!("\n=== FULL FILE ===");
            println!("Parsed {} values", values.len());
            println!("Remaining: {:?} bytes", rest.len());
            if !rest.is_empty() {
                println!("Remaining content: {:?}...", &rest[..rest.len().min(100)]);
            }
            for (i, v) in values.iter().enumerate() {
                match v {
                    mlir_lisp::parser::Value::List(elements) if !elements.is_empty() => {
                        if let mlir_lisp::parser::Value::Symbol(name) = &elements[0] {
                            println!("  Form {}: ({} ...)", i, name);
                        } else {
                            println!("  Form {}: (? ...)", i);
                        }
                    }
                    _ => println!("  Form {}: {:?}", i, v),
                }
            }
        }
        Err(e) => println!("Error: {:?}", e),
    }
}
