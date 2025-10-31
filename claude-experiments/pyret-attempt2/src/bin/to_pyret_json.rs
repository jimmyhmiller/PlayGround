use pyret_attempt2::{Parser, Expr};
use pyret_attempt2::tokenizer::Tokenizer;
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::io::{self, Read};

/// Convert our AST to Pyret's JSON format (no locations, specific field names)
fn expr_to_pyret_json(expr: &Expr) -> Value {
    match expr {
        Expr::SNum { n, .. } => {
            json!({
                "type": "s-num",
                "value": n.to_string()
            })
        }
        Expr::SStr { s, .. } => {
            json!({
                "type": "s-str",
                "value": s
            })
        }
        Expr::SBool { b, .. } => {
            json!({
                "type": "s-bool",
                "value": b
            })
        }
        Expr::SId { id, .. } => {
            // Names need special handling - for now just extract the string
            json!({
                "type": "s-id",
                "id": name_to_pyret_json(id)
            })
        }
        Expr::SOp { op, left, right, .. } => {
            json!({
                "type": "s-op",
                "op": op,
                "left": expr_to_pyret_json(left),
                "right": expr_to_pyret_json(right)
            })
        }
        Expr::SParen { expr, .. } => {
            json!({
                "type": "s-paren",
                "expr": expr_to_pyret_json(expr)
            })
        }
        Expr::SApp { _fun, args, .. } => {
            json!({
                "type": "s-app",
                "fun": expr_to_pyret_json(_fun),
                "args": args.iter().map(|e| expr_to_pyret_json(e.as_ref())).collect::<Vec<_>>()
            })
        }
        Expr::SArray { values, .. } => {
            json!({
                "type": "s-array",
                "values": values.iter().map(|e| expr_to_pyret_json(e.as_ref())).collect::<Vec<_>>()
            })
        }
        Expr::SDot { obj, field, .. } => {
            json!({
                "type": "s-dot",
                "obj": expr_to_pyret_json(obj),
                "field": field
            })
        }
        _ => {
            json!({
                "type": "UNSUPPORTED",
                "debug": format!("{:?}", expr)
            })
        }
    }
}

fn name_to_pyret_json(name: &pyret_attempt2::Name) -> Value {
    use pyret_attempt2::Name;
    match name {
        Name::SUnderscore { .. } => {
            json!({
                "type": "s-underscore"
            })
        }
        Name::SName { s, .. } => {
            json!({
                "type": "s-name",
                "name": s
            })
        }
        Name::SGlobal { s } => {
            json!({
                "type": "s-global",
                "name": s
            })
        }
        Name::SModuleGlobal { s } => {
            json!({
                "type": "s-module-global",
                "name": s
            })
        }
        Name::STypeGlobal { s } => {
            json!({
                "type": "s-type-global",
                "name": s
            })
        }
        Name::SAtom { base, serial } => {
            json!({
                "type": "s-atom",
                "base": base,
                "serial": serial
            })
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    let input = if args.len() > 1 {
        // Read from file
        fs::read_to_string(&args[1])?
    } else {
        // Read from stdin
        let mut buffer = String::new();
        io::stdin().read_to_string(&mut buffer)?;
        buffer
    };

    let mut tokenizer = Tokenizer::new(&input);
    let tokens = tokenizer.tokenize();
    let mut parser = Parser::new(tokens, "input.arr".to_string());
    let expr = parser.parse_expr()?;

    let json = expr_to_pyret_json(&expr);
    println!("{}", serde_json::to_string_pretty(&json)?);

    Ok(())
}
