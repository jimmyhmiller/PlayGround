//! A small interpreter over the CFG, used as the **executable oracle** for
//! lowering and SSA construction. It runs both the pre-SSA form (variables via
//! `ReadVar`/`WriteVar` over an environment) and the post-SSA form (values via
//! block arguments), so a bug in either phase shows up as a value that disagrees
//! with Node running the original JavaScript.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::rc::Rc;

use crate::cfg::{BinOp, BlockId, Cfg, Const, MemberKey, Op, PropKey, Term, UnOp, Value};

/// A runtime value (the subset the oracle exercises).
#[derive(Debug, Clone)]
pub enum Val {
    Undef,
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Obj(Rc<RefCell<BTreeMap<String, Val>>>),
    Arr(Rc<RefCell<Vec<Val>>>),
}

const STEP_LIMIT: usize = 1_000_000;

/// Run the CFG with the given argument values. Host globals (e.g. `Math`) can be
/// supplied via `globals` for member/call support.
pub fn run(cfg: &Cfg, args: &[Val]) -> Result<Val, String> {
    let mut vals: HashMap<Value, Val> = HashMap::new();
    let mut env: HashMap<u32, Val> = HashMap::new();

    // Bind parameters.
    for (i, p) in cfg.params.iter().enumerate() {
        vals.insert(*p, args.get(i).cloned().unwrap_or(Val::Undef));
    }

    let mut block = cfg.entry;
    let mut steps = 0;
    loop {
        let b = cfg.block(block);
        for instr in &b.instrs {
            steps += 1;
            if steps > STEP_LIMIT {
                return Err("step limit exceeded".into());
            }
            let v = eval_op(&instr.op, &vals, &mut env)?;
            if let Some(r) = instr.result {
                vals.insert(r, v);
            }
        }
        match &b.term {
            Term::Ret(v) => {
                return Ok(match v {
                    Some(v) => vals.get(v).cloned().unwrap_or(Val::Undef),
                    None => Val::Undef,
                });
            }
            Term::Unreachable => return Ok(Val::Undef),
            Term::Br(target, args) => {
                pass_args(cfg, *target, args, &mut vals)?;
                block = *target;
            }
            Term::CondBr { cond, then_block, then_args, else_block, else_args } => {
                let c = truthy(vals.get(cond).ok_or("cond unset")?);
                let (target, a) = if c { (*then_block, then_args) } else { (*else_block, else_args) };
                pass_args(cfg, target, a, &mut vals)?;
                block = target;
            }
        }
        steps += 1;
        if steps > STEP_LIMIT {
            return Err("step limit exceeded".into());
        }
    }
}

/// Bind a successor block's parameters to the branch operands (MLIR phi).
fn pass_args(cfg: &Cfg, target: BlockId, args: &[Value], vals: &mut HashMap<Value, Val>) -> Result<(), String> {
    let params = &cfg.block(target).params;
    if !args.is_empty() || !params.is_empty() {
        if args.len() != params.len() {
            return Err(format!("block {target:?}: {} args for {} params", args.len(), params.len()));
        }
        // Snapshot first (params may alias args across a self-loop edge).
        let snapshot: Vec<Val> = args.iter().map(|a| vals.get(a).cloned().unwrap_or(Val::Undef)).collect();
        for (p, v) in params.iter().zip(snapshot) {
            vals.insert(*p, v);
        }
    }
    Ok(())
}

fn eval_op(op: &Op, vals: &HashMap<Value, Val>, env: &mut HashMap<u32, Val>) -> Result<Val, String> {
    let get = |v: &Value| vals.get(v).cloned().unwrap_or(Val::Undef);
    Ok(match op {
        Op::Const(c) => from_const(c),
        Op::ReadVar(var) => env.get(&var.0).cloned().unwrap_or(Val::Undef),
        Op::WriteVar(var, v) => {
            env.insert(var.0, get(v));
            Val::Undef
        }
        Op::Bin(b, x, y) => bin(*b, &get(x), &get(y))?,
        Op::Un(u, x) => un(*u, &get(x))?,
        Op::Global(name) => return Err(format!("interp: free global `{name}` unsupported")),
        Op::Call { .. } => return Err("interp: call unsupported".into()),
        Op::Member { obj, prop } => {
            let o = get(obj);
            member(&o, prop, vals)?
        }
        Op::StoreMember { obj, prop, value } => {
            let o = get(obj);
            let v = get(value);
            let key = match prop {
                MemberKey::Static(s) => s.clone(),
                MemberKey::Computed(c) => to_js_string(&get(c)),
            };
            match &o {
                Val::Obj(m) => {
                    m.borrow_mut().insert(key, v.clone());
                }
                Val::Arr(a) => {
                    if let Ok(i) = key.parse::<usize>() {
                        let mut arr = a.borrow_mut();
                        if i >= arr.len() {
                            arr.resize(i + 1, Val::Undef);
                        }
                        arr[i] = v.clone();
                    }
                }
                _ => {}
            }
            v
        }
        Op::MakeArray(elems) => Val::Arr(Rc::new(RefCell::new(elems.iter().map(|e| get(e)).collect()))),
        Op::MakeObject(props) => {
            let mut m = BTreeMap::new();
            for (k, v) in props {
                let key = match k {
                    PropKey::Ident(s) => s.clone(),
                    PropKey::Computed(c) => to_js_string(&get(c)),
                };
                m.insert(key, get(v));
            }
            Val::Obj(Rc::new(RefCell::new(m)))
        }
    })
}

fn member(o: &Val, prop: &MemberKey, vals: &HashMap<Value, Val>) -> Result<Val, String> {
    let key = match prop {
        MemberKey::Static(s) => s.clone(),
        MemberKey::Computed(c) => to_js_string(vals.get(c).unwrap_or(&Val::Undef)),
    };
    Ok(match o {
        Val::Obj(m) => m.borrow().get(&key).cloned().unwrap_or(Val::Undef),
        Val::Arr(a) => {
            if key == "length" {
                Val::Num(a.borrow().len() as f64)
            } else if let Ok(i) = key.parse::<usize>() {
                a.borrow().get(i).cloned().unwrap_or(Val::Undef)
            } else {
                Val::Undef
            }
        }
        _ => Val::Undef,
    })
}

fn from_const(c: &Const) -> Val {
    match c {
        Const::Undef => Val::Undef,
        Const::Null => Val::Null,
        Const::Bool(b) => Val::Bool(*b),
        Const::Num(bits) => Val::Num(f64::from_bits(*bits)),
        Const::Str(s) => Val::Str(s.clone()),
    }
}

pub fn truthy(v: &Val) -> bool {
    match v {
        Val::Undef | Val::Null => false,
        Val::Bool(b) => *b,
        Val::Num(n) => *n != 0.0 && !n.is_nan(),
        Val::Str(s) => !s.is_empty(),
        Val::Obj(_) | Val::Arr(_) => true,
    }
}

fn to_number(v: &Val) -> f64 {
    match v {
        Val::Undef => f64::NAN,
        Val::Null => 0.0,
        Val::Bool(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        Val::Num(n) => *n,
        Val::Str(s) => {
            let t = s.trim();
            if t.is_empty() {
                0.0
            } else {
                t.parse::<f64>().unwrap_or(f64::NAN)
            }
        }
        _ => f64::NAN,
    }
}

pub fn to_js_string(v: &Val) -> String {
    match v {
        Val::Undef => "undefined".into(),
        Val::Null => "null".into(),
        Val::Bool(b) => b.to_string(),
        Val::Num(n) => js_num_to_string(*n),
        Val::Str(s) => s.clone(),
        Val::Arr(a) => a.borrow().iter().map(to_js_string).collect::<Vec<_>>().join(","),
        Val::Obj(_) => "[object Object]".into(),
    }
}

fn bin(b: BinOp, x: &Val, y: &Val) -> Result<Val, String> {
    use BinOp::*;
    Ok(match b {
        Add => {
            // string concat if either operand is a string, else numeric
            if matches!(x, Val::Str(_)) || matches!(y, Val::Str(_)) {
                Val::Str(format!("{}{}", to_js_string(x), to_js_string(y)))
            } else {
                Val::Num(to_number(x) + to_number(y))
            }
        }
        Sub => Val::Num(to_number(x) - to_number(y)),
        Mul => Val::Num(to_number(x) * to_number(y)),
        Div => Val::Num(to_number(x) / to_number(y)),
        Mod => Val::Num(js_mod(to_number(x), to_number(y))),
        Pow => Val::Num(to_number(x).powf(to_number(y))),
        Lt => Val::Bool(rel(x, y, |a, b| a < b, |a, b| a < b)),
        Le => Val::Bool(rel(x, y, |a, b| a <= b, |a, b| a <= b)),
        Gt => Val::Bool(rel(x, y, |a, b| a > b, |a, b| a > b)),
        Ge => Val::Bool(rel(x, y, |a, b| a >= b, |a, b| a >= b)),
        StrictEq => Val::Bool(strict_eq(x, y)),
        StrictNe => Val::Bool(!strict_eq(x, y)),
        Eq => Val::Bool(loose_eq(x, y)),
        Ne => Val::Bool(!loose_eq(x, y)),
        BitAnd => Val::Num(((to_i32(x)) & (to_i32(y))) as f64),
        BitOr => Val::Num(((to_i32(x)) | (to_i32(y))) as f64),
        BitXor => Val::Num(((to_i32(x)) ^ (to_i32(y))) as f64),
        Shl => Val::Num((to_i32(x).wrapping_shl(to_u32(y) & 31)) as f64),
        Shr => Val::Num((to_i32(x).wrapping_shr(to_u32(y) & 31)) as f64),
        UShr => Val::Num((to_u32(x).wrapping_shr(to_u32(y) & 31)) as f64),
    })
}

fn un(u: UnOp, x: &Val) -> Result<Val, String> {
    Ok(match u {
        UnOp::Neg => Val::Num(-to_number(x)),
        UnOp::Pos => Val::Num(to_number(x)),
        UnOp::Not => Val::Bool(!truthy(x)),
        UnOp::BitNot => Val::Num(!(to_i32(x)) as f64),
        UnOp::Void => Val::Undef,
        UnOp::TypeOf => Val::Str(
            match x {
                Val::Undef => "undefined",
                Val::Null => "object",
                Val::Bool(_) => "boolean",
                Val::Num(_) => "number",
                Val::Str(_) => "string",
                Val::Obj(_) | Val::Arr(_) => "object",
            }
            .into(),
        ),
    })
}

/// JS relational: both strings compare lexicographically, else numeric.
fn rel(x: &Val, y: &Val, sf: fn(&str, &str) -> bool, nf: fn(f64, f64) -> bool) -> bool {
    if let (Val::Str(a), Val::Str(b)) = (x, y) {
        sf(a, b)
    } else {
        let (a, b) = (to_number(x), to_number(y));
        if a.is_nan() || b.is_nan() {
            false
        } else {
            nf(a, b)
        }
    }
}

fn strict_eq(x: &Val, y: &Val) -> bool {
    match (x, y) {
        (Val::Undef, Val::Undef) => true,
        (Val::Null, Val::Null) => true,
        (Val::Bool(a), Val::Bool(b)) => a == b,
        (Val::Num(a), Val::Num(b)) => a == b,
        (Val::Str(a), Val::Str(b)) => a == b,
        (Val::Obj(a), Val::Obj(b)) => Rc::ptr_eq(a, b),
        (Val::Arr(a), Val::Arr(b)) => Rc::ptr_eq(a, b),
        _ => false,
    }
}

fn loose_eq(x: &Val, y: &Val) -> bool {
    match (x, y) {
        (Val::Null | Val::Undef, Val::Null | Val::Undef) => true,
        (Val::Null | Val::Undef, _) | (_, Val::Null | Val::Undef) => false,
        _ if std::mem::discriminant(x) == std::mem::discriminant(y) => strict_eq(x, y),
        _ => to_number(x) == to_number(y),
    }
}

fn js_mod(a: f64, b: f64) -> f64 {
    if b == 0.0 || a.is_nan() || b.is_nan() || a.is_infinite() {
        f64::NAN
    } else if b.is_infinite() {
        a
    } else {
        a % b
    }
}

fn to_i32(v: &Val) -> i32 {
    let n = to_number(v);
    if !n.is_finite() {
        return 0;
    }
    (n.trunc() as i64 as u32) as i32
}
fn to_u32(v: &Val) -> u32 {
    to_i32(v) as u32
}

/// JS `Number.prototype.toString` for finite values (matches Node for the cases
/// the oracle exercises).
pub fn js_num_to_string(n: f64) -> String {
    if n.is_nan() {
        "NaN".into()
    } else if n == 0.0 {
        "0".into()
    } else if n.is_infinite() {
        if n > 0.0 {
            "Infinity".into()
        } else {
            "-Infinity".into()
        }
    } else if n.fract() == 0.0 && n.abs() < 1e21 {
        format!("{}", n as i64)
    } else {
        let s = format!("{n}");
        s
    }
}

/// A tagged canonical form for oracle comparison (mirrored on the JS side).
pub fn tag(v: &Val) -> String {
    match v {
        Val::Undef => "u:".into(),
        Val::Null => "l:".into(),
        Val::Bool(b) => format!("b:{b}"),
        Val::Num(n) => format!("n:{}", js_num_to_string(*n)),
        Val::Str(s) => format!("s:{s}"),
        Val::Arr(a) => format!("a:[{}]", a.borrow().iter().map(tag).collect::<Vec<_>>().join(",")),
        Val::Obj(m) => format!(
            "o:{{{}}}",
            m.borrow().iter().map(|(k, v)| format!("{k}={}", tag(v))).collect::<Vec<_>>().join(",")
        ),
    }
}
