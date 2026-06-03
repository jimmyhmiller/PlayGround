use std::io::Read;
fn main() {
    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s).unwrap();
    match jsir_ssa::lower(&s) {
        Ok(cfg) => print!("{}", jsir_ssa::print::print(&cfg)),
        Err(e) => println!("ERR: {e}"),
    }
}
