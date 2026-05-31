//! A compact MLIR-ish textual dump of a CFG (for debugging / golden tests).

use crate::cfg::{BinOp, Cfg, Const, MemberKey, Op, PropKey, Term, UnOp};

pub fn print(cfg: &Cfg) -> String {
    let mut s = String::new();
    let params: Vec<String> = cfg.params.iter().map(|v| format!("%{}", v.0)).collect();
    s.push_str(&format!("func({}) {{\n", params.join(", ")));
    for b in &cfg.blocks {
        let bp: Vec<String> = b.params.iter().map(|v| format!("%{}", v.0)).collect();
        if bp.is_empty() {
            s.push_str(&format!("  ^bb{}:\n", b.id.0));
        } else {
            s.push_str(&format!("  ^bb{}({}):\n", b.id.0, bp.join(", ")));
        }
        for ins in &b.instrs {
            let lhs = match ins.result {
                Some(v) => format!("%{} = ", v.0),
                None => String::new(),
            };
            s.push_str(&format!("    {lhs}{}\n", op(&ins.op)));
        }
        s.push_str(&format!("    {}\n", term(&b.term)));
    }
    s.push_str("}\n");
    s
}

fn op(o: &Op) -> String {
    match o {
        Op::Const(c) => format!("const {}", konst(c)),
        Op::ReadVar(v) => format!("read @{}", v.0),
        Op::WriteVar(v, x) => format!("write @{} <- %{}", v.0, x.0),
        Op::Bin(b, x, y) => format!("{} %{}, %{}", binop(*b), x.0, y.0),
        Op::Un(u, x) => format!("{} %{}", unop(*u), x.0),
        Op::Global(n) => format!("global {n:?}"),
        Op::Call { callee, args } => {
            let a: Vec<String> = args.iter().map(|v| format!("%{}", v.0)).collect();
            format!("call %{}({})", callee.0, a.join(", "))
        }
        Op::Member { obj, prop } => match prop {
            MemberKey::Static(s) => format!("member %{}.{s}", obj.0),
            MemberKey::Computed(c) => format!("member %{}[%{}]", obj.0, c.0),
        },
        Op::StoreMember { obj, prop, value } => match prop {
            MemberKey::Static(s) => format!("store %{}.{s} <- %{}", obj.0, value.0),
            MemberKey::Computed(c) => format!("store %{}[%{}] <- %{}", obj.0, c.0, value.0),
        },
        Op::MakeArray(e) => {
            let a: Vec<String> = e.iter().map(|v| format!("%{}", v.0)).collect();
            format!("array [{}]", a.join(", "))
        }
        Op::MakeObject(p) => {
            let a: Vec<String> = p
                .iter()
                .map(|(k, v)| {
                    let key = match k {
                        PropKey::Ident(s) => s.clone(),
                        PropKey::Computed(c) => format!("[%{}]", c.0),
                    };
                    format!("{key}: %{}", v.0)
                })
                .collect();
            format!("object {{{}}}", a.join(", "))
        }
    }
}

fn konst(c: &Const) -> String {
    match c {
        Const::Undef => "undefined".into(),
        Const::Null => "null".into(),
        Const::Bool(b) => b.to_string(),
        Const::Num(bits) => crate::interp::js_num_to_string(f64::from_bits(*bits)),
        Const::Str(s) => format!("{s:?}"),
    }
}

fn term(t: &Term) -> String {
    let args = |a: &[crate::cfg::Value]| {
        if a.is_empty() {
            String::new()
        } else {
            format!("({})", a.iter().map(|v| format!("%{}", v.0)).collect::<Vec<_>>().join(", "))
        }
    };
    match t {
        Term::Br(b, a) => format!("br ^bb{}{}", b.0, args(a)),
        Term::CondBr { cond, then_block, then_args, else_block, else_args } => format!(
            "cond_br %{}, ^bb{}{}, ^bb{}{}",
            cond.0,
            then_block.0,
            args(then_args),
            else_block.0,
            args(else_args)
        ),
        Term::Ret(Some(v)) => format!("ret %{}", v.0),
        Term::Ret(None) => "ret".into(),
        Term::Unreachable => "unreachable".into(),
    }
}

fn binop(b: BinOp) -> &'static str {
    use BinOp::*;
    match b {
        Add => "add", Sub => "sub", Mul => "mul", Div => "div", Mod => "mod", Pow => "pow",
        Eq => "eq", Ne => "ne", StrictEq => "seq", StrictNe => "sne",
        Lt => "lt", Le => "le", Gt => "gt", Ge => "ge",
        BitAnd => "band", BitOr => "bor", BitXor => "bxor", Shl => "shl", Shr => "shr", UShr => "ushr",
    }
}
fn unop(u: UnOp) -> &'static str {
    use UnOp::*;
    match u {
        Neg => "neg", Pos => "pos", Not => "not", BitNot => "bnot", TypeOf => "typeof", Void => "void",
    }
}
