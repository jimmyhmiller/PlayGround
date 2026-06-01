fn main() {
    let srcs = [
        "function A(props) { const x = {a: props.a}; return <div>{x.a}</div>; }",
        "function f(a,b){ let style={color:a}; let el={size:b,props:style}; return el; }",
    ];
    for s in srcs {
        println!("=== {s}");
        match jsir_ssa::codegen::compile(s) {
            Ok(c) => println!("{c}"),
            Err(e) => println!("ERR: {e}"),
        }
    }
}
