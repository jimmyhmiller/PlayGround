use pyret_attempt::{parse_program};
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 {
        // Parse file from command line
        let filename = &args[1];
        match fs::read_to_string(filename) {
            Ok(source) => {
                println!("Parsing file: {}", filename);
                match parse_program(&source) {
                    Ok(program) => {
                        println!("✓ Successfully parsed!");
                        println!("  Location: {}:{}-{}:{}",
                            program.loc.start_line, program.loc.start_column,
                            program.loc.end_line, program.loc.end_column);
                        println!("  Imports: {}", program.imports.len());
                    },
                    Err(e) => {
                        eprintln!("✗ Parse error: {}", e);
                        std::process::exit(1);
                    }
                }
            },
            Err(e) => {
                eprintln!("Error reading file: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        // Default test
        let source = r#"
fun square(x):
    x * x
end

square(5)
        "#;

        match parse_program(source) {
            Ok(program) => {
                println!("Successfully parsed program!");
                println!("Program location: {:?}", program.loc);
                println!("Number of imports: {}", program.imports.len());
            },
            Err(e) => {
                eprintln!("Parse error: {}", e);
            }
        }
    }
}
