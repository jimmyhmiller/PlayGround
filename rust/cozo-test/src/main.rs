// Note: Cozo seems very early days. The error messages for bad queries
// are just parser::pest errors with little to no information

fn main() {
    use cozo::*;

    let db = DbInstance::new("sqlite", "test.db", Default::default()).unwrap();
    // let script = ":create Document { id: String, => path: String, title: String, }";
    // let script = "
    // ?[id, path, title] <- [['1', 'hello.pdf', 'Hello World']]
    // :put Document { id => path, title }";

    let script = "?[id, path, title] := *Document{id, path, title";
    let result = db
        .run_script(script, Default::default(), ScriptMutability::Mutable)
        .unwrap();
    println!("{:?}", result);
}
