use std::io::Read;
fn main() {
    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s).unwrap();
    if std::env::var("DUMP_SCOPES").is_ok() {
        match jsir_swc::source_to_ir(&s) {
            Ok(ir) => match jsir_ssa::plan(&ir) {
                Ok(p) => {
                    println!("fn={} single_block={}", p.fn_name, p.single_block);
                    println!("{}", jsir_ssa::scopes::render_info(&p.infos));
                }
                Err(e) => println!("plan ERR: {e}"),
            },
            Err(e) => println!("parse ERR: {e}"),
        }
        return;
    }
    match jsir_ssa::codegen::compile(&s) {
        Ok(c) => println!("{c}"),
        Err(e) => println!("ERR: {e}"),
    }
}
