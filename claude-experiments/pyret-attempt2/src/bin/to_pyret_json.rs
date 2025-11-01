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
        Expr::SConstruct { modifier, constructor, values, .. } => {
            json!({
                "type": "s-construct",
                "modifier": modifier_to_pyret_json(modifier),
                "constructor": expr_to_pyret_json(constructor),
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
        Expr::SBracket { obj, field, .. } => {
            json!({
                "type": "s-bracket",
                "obj": expr_to_pyret_json(obj),
                "field": expr_to_pyret_json(field)
            })
        }
        Expr::SCheckTest { op, refinement, left, right, cause, .. } => {
            json!({
                "type": "s-check-test",
                "op": check_op_to_pyret_json(op),
                "refinement": refinement.as_ref().map(|e| expr_to_pyret_json(e)),
                "left": expr_to_pyret_json(left),
                "right": right.as_ref().map(|e| expr_to_pyret_json(e)),
                "cause": cause.as_ref().map(|e| expr_to_pyret_json(e))
            })
        }
        Expr::SObj { fields, .. } => {
            json!({
                "type": "s-obj",
                "fields": fields.iter().map(|f| member_to_pyret_json(f)).collect::<Vec<_>>()
            })
        }
        Expr::SLam { name, params, args, ann, doc, body, check, check_loc, blocky, .. } => {
            json!({
                "type": "s-lam",
                "name": name,
                "params": params.iter().map(|p| name_to_pyret_json(p)).collect::<Vec<_>>(),
                "args": args.iter().map(|a| bind_to_pyret_json(a)).collect::<Vec<_>>(),
                "ann": ann_to_pyret_json(ann),
                "doc": doc,
                "body": expr_to_pyret_json(body),
                "check": check.as_ref().map(|c| expr_to_pyret_json(c)),
                "check-loc": check_loc,
                "blocky": blocky
            })
        }
        Expr::SBlock { stmts, .. } => {
            json!({
                "type": "s-block",
                "stmts": stmts.iter().map(|s| expr_to_pyret_json(s)).collect::<Vec<_>>()
            })
        }
        Expr::SUserBlock { body, .. } => {
            json!({
                "type": "s-user-block",
                "body": expr_to_pyret_json(body)
            })
        }
        Expr::STuple { fields, .. } => {
            json!({
                "type": "s-tuple",
                "fields": fields.iter().map(|f| expr_to_pyret_json(f.as_ref())).collect::<Vec<_>>()
            })
        }
        Expr::STupleGet { tup, index, .. } => {
            json!({
                "type": "s-tuple-get",
                "tup": expr_to_pyret_json(tup),
                "index": index
            })
        }
        Expr::SIf { branches, blocky, .. } => {
            json!({
                "type": "s-if",
                "branches": branches.iter().map(|b| if_branch_to_pyret_json(b)).collect::<Vec<_>>(),
                "blocky": blocky
            })
        }
        Expr::SIfElse { branches, _else, blocky, .. } => {
            json!({
                "type": "s-if-else",
                "branches": branches.iter().map(|b| if_branch_to_pyret_json(b)).collect::<Vec<_>>(),
                "else": expr_to_pyret_json(_else),
                "blocky": blocky
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

fn member_to_pyret_json(member: &pyret_attempt2::Member) -> Value {
    use pyret_attempt2::Member;
    match member {
        Member::SDataField { name, value, .. } => {
            json!({
                "type": "s-data-field",
                "name": name,
                "value": expr_to_pyret_json(value)
            })
        }
        Member::SMutableField { name, ann, value, .. } => {
            json!({
                "type": "s-mutable-field",
                "name": name,
                "ann": ann_to_pyret_json(ann),
                "value": expr_to_pyret_json(value)
            })
        }
        Member::SMethodField { .. } => {
            json!({
                "type": "UNSUPPORTED",
                "debug": "Method fields not yet implemented"
            })
        }
    }
}

fn ann_to_pyret_json(ann: &pyret_attempt2::Ann) -> Value {
    use pyret_attempt2::Ann;
    match ann {
        Ann::ABlank => json!({"type": "a-blank"}),
        _ => json!({
            "type": "UNSUPPORTED",
            "debug": format!("{:?}", ann)
        })
    }
}

fn check_op_to_pyret_json(op: &pyret_attempt2::CheckOp) -> Value {
    use pyret_attempt2::CheckOp;
    match op {
        CheckOp::SOpIs { .. } => json!({"type": "s-op-is"}),
        CheckOp::SOpIsRoughly { .. } => json!({"type": "s-op-is-roughly"}),
        CheckOp::SOpIsNot { .. } => json!({"type": "s-op-is-not"}),
        CheckOp::SOpIsNotRoughly { .. } => json!({"type": "s-op-is-not-roughly"}),
        CheckOp::SOpIsOp { op, .. } => json!({"type": "s-op-is-op", "op": op}),
        CheckOp::SOpIsNotOp { op, .. } => json!({"type": "s-op-is-not-op", "op": op}),
        CheckOp::SOpSatisfies { .. } => json!({"type": "s-op-satisfies"}),
        CheckOp::SOpSatisfiesNot { .. } => json!({"type": "s-op-satisfies-not"}),
        CheckOp::SOpRaises { .. } => json!({"type": "s-op-raises"}),
        CheckOp::SOpRaisesOther { .. } => json!({"type": "s-op-raises-other"}),
        CheckOp::SOpRaisesNot { .. } => json!({"type": "s-op-raises-not"}),
        CheckOp::SOpRaisesSatisfies { .. } => json!({"type": "s-op-raises-satisfies"}),
        CheckOp::SOpRaisesViolates { .. } => json!({"type": "s-op-raises-violates"}),
    }
}

fn modifier_to_pyret_json(modifier: &pyret_attempt2::ConstructModifier) -> Value {
    use pyret_attempt2::ConstructModifier;
    match modifier {
        ConstructModifier::SConstructNormal => json!("s-construct-normal"),
        ConstructModifier::SConstructLazy => json!("s-construct-lazy"),
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

fn bind_to_pyret_json(bind: &pyret_attempt2::Bind) -> Value {
    use pyret_attempt2::Bind;
    match bind {
        Bind::SBind { id, ann, shadows, .. } => {
            json!({
                "type": "s-bind",
                "name": name_to_pyret_json(id),
                "ann": ann_to_pyret_json(ann),
                "shadows": shadows
            })
        }
        Bind::STupleBind { fields, as_name, .. } => {
            json!({
                "type": "s-tuple-bind",
                "fields": fields.iter().map(|f| bind_to_pyret_json(f)).collect::<Vec<_>>(),
                "as-name": as_name.as_ref().map(|n| bind_to_pyret_json(n))
            })
        }
    }
}

fn if_branch_to_pyret_json(branch: &pyret_attempt2::IfBranch) -> Value {
    json!({
        "type": "s-if-branch",
        "test": expr_to_pyret_json(&branch.test),
        "body": expr_to_pyret_json(&branch.body)
    })
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
