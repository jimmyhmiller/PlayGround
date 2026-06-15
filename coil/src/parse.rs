//! Surface s-expressions -> core AST.

use std::collections::HashMap;

use crate::ast::*;
use crate::convention::{Convention, Lowering, NativeCc};
use crate::reader::Sexp;

pub fn parse_program(forms: &[Sexp]) -> Result<Program, String> {
    let mut conventions = HashMap::new();
    conventions.insert("c".to_string(), Convention::default_c());
    let mut funcs = Vec::new();

    for form in forms {
        let items = as_list(form, "top-level form must be a list")?;
        let head = head_sym(items)?;
        match head.as_str() {
            "defcc" => {
                let c = parse_defcc(&items[1..])?;
                if conventions.insert(c.name.clone(), c.clone()).is_some() && c.name != "c" {
                    return Err(format!("convention '{}' defined twice", c.name));
                }
            }
            "defn" => {
                let f = parse_defn(&items[1..])?;
                if funcs.iter().any(|g: &Func| g.name == f.name) {
                    return Err(format!("function '{}' defined twice", f.name));
                }
                funcs.push(f);
            }
            other => return Err(format!("unknown top-level form '{other}'")),
        }
    }

    Ok(Program { conventions, funcs })
}

fn parse_defcc(rest: &[Sexp]) -> Result<Convention, String> {
    let name = sym(rest.first().ok_or("defcc: missing name")?, "defcc name")?;
    let mut params = vec![];
    let mut ret = None;
    let mut clobber = vec![];
    let mut preserve = vec![];
    let mut native: Option<NativeCc> = None;
    let mut shim = false;

    let mut i = 1;
    while i < rest.len() {
        let kw = keyword(&rest[i], "defcc option")?;
        let val = rest
            .get(i + 1)
            .ok_or_else(|| format!("defcc: option ':{kw}' missing value"))?;
        match kw.as_str() {
            "params" => params = sym_vec(val, "params")?,
            "ret" => ret = Some(sym(val, "ret")?),
            "clobber" => clobber = sym_vec(val, "clobber")?,
            "preserve" => preserve = sym_vec(val, "preserve")?,
            "native" => {
                let n = sym(val, "native")?;
                native = Some(
                    NativeCc::parse(&n)
                        .ok_or_else(|| format!("defcc: unknown native cc '{n}'"))?,
                );
            }
            "lower" => {
                // :lower shim  — declares an exotic convention (M2)
                if sym(val, "lower")? == "shim" {
                    shim = true;
                }
            }
            other => return Err(format!("defcc: unknown option ':{other}'")),
        }
        i += 2;
    }

    let lowering = match (native, shim) {
        (Some(cc), _) => Lowering::Native(cc),
        (None, true) => Lowering::Shim,
        (None, false) => {
            return Err(format!(
                "defcc '{name}': needs a lowering (:native <c|fast|cold> or :lower shim)"
            ))
        }
    };

    Ok(Convention {
        name,
        params,
        ret,
        clobber,
        preserve,
        lowering,
    })
}

fn parse_defn(rest: &[Sexp]) -> Result<Func, String> {
    let mut i = 0;
    let name = sym(rest.get(i).ok_or("defn: missing name")?, "defn name")?;
    i += 1;

    // optional `:cc <name>`
    let mut cc = "c".to_string();
    if let Some(Sexp::Keyword(k)) = rest.get(i) {
        if k == "cc" {
            cc = sym(
                rest.get(i + 1).ok_or("defn: ':cc' missing convention name")?,
                "cc name",
            )?;
            i += 2;
        }
    }

    // params vector
    let params_v = match rest.get(i) {
        Some(Sexp::Vector(v)) => v,
        _ => return Err(format!("defn '{name}': expected parameter vector")),
    };
    i += 1;
    let mut params = Vec::new();
    for p in params_v {
        let pl = as_list(p, "parameter must be (name :type)")?;
        let pname = sym(&pl[0], "param name")?;
        let pty = parse_type(pl.get(1).ok_or("param missing type")?)?;
        params.push(Param { name: pname, ty: pty });
    }

    // return type `(-> :T)`
    let ret_l = as_list(
        rest.get(i).ok_or("defn: missing (-> :type)")?,
        "expected (-> :type)",
    )?;
    if head_sym(ret_l)? != "->" {
        return Err("defn: expected (-> :type) after params".to_string());
    }
    let ret = parse_type(ret_l.get(1).ok_or("(-> ) missing type")?)?;
    i += 1;

    // body
    let body: Vec<Expr> = rest[i..]
        .iter()
        .map(parse_expr)
        .collect::<Result<_, _>>()?;
    if body.is_empty() {
        return Err(format!("defn '{name}': empty body"));
    }

    Ok(Func {
        name,
        cc,
        params,
        ret,
        body,
    })
}

fn parse_type(s: &Sexp) -> Result<Type, String> {
    match s {
        Sexp::Keyword(k) if k == "i64" => Ok(Type::I64),
        other => Err(format!("unsupported type: {other:?} (only :i64 in M1)")),
    }
}

fn parse_expr(s: &Sexp) -> Result<Expr, String> {
    match s {
        Sexp::Int(n) => Ok(Expr::Int(*n)),
        Sexp::Sym(name) => Ok(Expr::Var(name.clone())),
        Sexp::Keyword(k) => Err(format!("unexpected keyword :{k} in expression")),
        Sexp::Vector(_) => Err("unexpected vector in expression".to_string()),
        Sexp::List(items) => parse_list_expr(items),
    }
}

fn parse_list_expr(items: &[Sexp]) -> Result<Expr, String> {
    let head = head_sym(items)?;
    let args = &items[1..];
    let bin = |op: BinOp| -> Result<Expr, String> {
        let (l, r) = two(args, &head)?;
        Ok(Expr::Bin {
            op,
            lhs: Box::new(parse_expr(l)?),
            rhs: Box::new(parse_expr(r)?),
        })
    };
    let cmp = |op: CmpOp| -> Result<Expr, String> {
        let (l, r) = two(args, &head)?;
        Ok(Expr::Cmp {
            op,
            lhs: Box::new(parse_expr(l)?),
            rhs: Box::new(parse_expr(r)?),
        })
    };
    match head.as_str() {
        "iadd" => bin(BinOp::Add),
        "isub" => bin(BinOp::Sub),
        "imul" => bin(BinOp::Mul),
        "idiv" => bin(BinOp::Div),
        "icmp-lt" => cmp(CmpOp::Lt),
        "icmp-le" => cmp(CmpOp::Le),
        "icmp-gt" => cmp(CmpOp::Gt),
        "icmp-ge" => cmp(CmpOp::Ge),
        "icmp-eq" => cmp(CmpOp::Eq),
        "icmp-ne" => cmp(CmpOp::Ne),
        "if" => {
            if args.len() != 3 {
                return Err("if: expects (if cond then else)".to_string());
            }
            Ok(Expr::If {
                cond: Box::new(parse_expr(&args[0])?),
                then: Box::new(parse_expr(&args[1])?),
                els: Box::new(parse_expr(&args[2])?),
            })
        }
        "do" => Ok(Expr::Do(
            args.iter().map(parse_expr).collect::<Result<_, _>>()?,
        )),
        "let" => {
            let binds_v = match args.first() {
                Some(Sexp::Vector(v)) => v,
                _ => return Err("let: expected binding vector".to_string()),
            };
            if binds_v.len() % 2 != 0 {
                return Err("let: bindings must be name/value pairs".to_string());
            }
            let mut binds = Vec::new();
            for pair in binds_v.chunks(2) {
                let name = sym(&pair[0], "let binding name")?;
                binds.push((name, parse_expr(&pair[1])?));
            }
            let body: Vec<Expr> = args[1..]
                .iter()
                .map(parse_expr)
                .collect::<Result<_, _>>()?;
            if body.is_empty() {
                return Err("let: empty body".to_string());
            }
            Ok(Expr::Let { binds, body })
        }
        "call" => {
            let f = sym(args.first().ok_or("call: missing function")?, "call target")?;
            let cargs = args[1..]
                .iter()
                .map(parse_expr)
                .collect::<Result<_, _>>()?;
            Ok(Expr::Call { func: f, args: cargs })
        }
        // direct application: (fib n) == (call fib n)
        other => {
            let cargs = args.iter().map(parse_expr).collect::<Result<_, _>>()?;
            Ok(Expr::Call {
                func: other.to_string(),
                args: cargs,
            })
        }
    }
}

// ---- small helpers -------------------------------------------------------

fn as_list<'a>(s: &'a Sexp, msg: &str) -> Result<&'a [Sexp], String> {
    match s {
        Sexp::List(items) => Ok(items),
        _ => Err(msg.to_string()),
    }
}

fn head_sym(items: &[Sexp]) -> Result<String, String> {
    match items.first() {
        Some(Sexp::Sym(s)) => Ok(s.clone()),
        Some(other) => Err(format!("expected a symbol head, got {other:?}")),
        None => Err("empty list".to_string()),
    }
}

fn sym(s: &Sexp, what: &str) -> Result<String, String> {
    match s {
        Sexp::Sym(s) => Ok(s.clone()),
        other => Err(format!("{what}: expected symbol, got {other:?}")),
    }
}

fn keyword(s: &Sexp, what: &str) -> Result<String, String> {
    match s {
        Sexp::Keyword(k) => Ok(k.clone()),
        other => Err(format!("{what}: expected keyword, got {other:?}")),
    }
}

fn sym_vec(s: &Sexp, what: &str) -> Result<Vec<String>, String> {
    match s {
        Sexp::Vector(v) => v.iter().map(|x| sym(x, what)).collect(),
        other => Err(format!("{what}: expected vector, got {other:?}")),
    }
}

fn two<'a>(args: &'a [Sexp], head: &str) -> Result<(&'a Sexp, &'a Sexp), String> {
    if args.len() != 2 {
        return Err(format!("{head}: expects exactly 2 arguments"));
    }
    Ok((&args[0], &args[1]))
}
