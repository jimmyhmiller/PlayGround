use std::io::Read;
fn main() {
    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s).unwrap();
    match jsir_ssa::codegen::compile(&s) {
        Ok(c) => println!("{c}"),
        Err(e) => println!("ERR: {e}"),
    }
}
