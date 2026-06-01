use jsir_ssa::codegen;
use std::io::Read;

fn main() {
    let mut src = String::new();
    std::io::stdin().read_to_string(&mut src).unwrap();
    match codegen::compile(&src) {
        Ok(c) => println!("{c}"),
        Err(e) => println!("ERR: {e}"),
    }
}
