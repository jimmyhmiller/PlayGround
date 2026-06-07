fn main() {
    for src in ["1 + 2 * 3;", "let a = 0; a = a + 1;"] {
        println!("// SRC: {src}\n{}\n", jsir_parse::parse_to_ir_text(src).unwrap());
    }
}
