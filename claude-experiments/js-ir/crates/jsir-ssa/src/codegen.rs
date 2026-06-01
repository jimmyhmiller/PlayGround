//! Memoization codegen — the React Compiler payoff.
//!
//! For a straight-line function (single basic block: the common component
//! shape) we emit JavaScript that wraps each pure reactive scope in a
//! `useMemoCache`-style cache check (`$[i] !== dep ? recompute : reuse`),
//! exactly as the React Compiler does. The emitted code is semantically
//! identical to the input but recomputes each memoized value only when its
//! dependencies change.
//!
//! Only single-block functions are handled here (no loops/branches); those are
//! where memoization matters most and where the emit is unambiguous.

use std::collections::HashSet;

use crate::cfg::{BinOp, Cfg, Const, MemberKey, Op, PropKey, Term, UnOp, Value};
use crate::scopes::ScopeInfo;

/// Emit a memoized version of a single-block function. Returns the JS source for
/// a `function f(...) { ... }`, or an error if the CFG has control flow.
pub fn emit_memoized(cfg: &Cfg, infos: &[ScopeInfo], fn_name: &str) -> Result<String, String> {
    if cfg.blocks.len() != 1 {
        return Err("codegen: only single-block (straight-line) functions supported".into());
    }
    let block = &cfg.blocks[0];
    let ret = match &block.term {
        Term::Ret(v) => *v,
        _ => return Err("codegen: function does not end in return".into()),
    };

    let namer = Namer::new(cfg);
    let instrs = &block.instrs;

    // Only scopes that actually produce an output are emitted.
    let emitted: Vec<&ScopeInfo> = infos.iter().filter(|i| !i.outputs.is_empty()).collect();

    // Which instructions each emitted scope owns: those defining a scope value,
    // plus member-stores that mutate a scope value.
    let value_set: Vec<HashSet<Value>> = emitted.iter().map(|i| i.scope.values.iter().copied().collect()).collect();
    let mut instr_owner: Vec<Option<usize>> = vec![None; instrs.len()];
    let mut owned: Vec<Vec<usize>> = vec![Vec::new(); emitted.len()];
    for (ix, ins) in instrs.iter().enumerate() {
        let owner = value_set.iter().position(|vs| match (&ins.op, ins.result) {
            (_, Some(r)) => vs.contains(&r),
            (Op::StoreMember { obj, .. }, None) => vs.contains(obj),
            _ => false,
        });
        if let Some(s) = owner {
            instr_owner[ix] = Some(s);
            owned[s].push(ix);
        }
    }
    let last_owned: Vec<usize> = owned.iter().map(|v| v.last().copied().unwrap_or(0)).collect();

    // Outputs declared `let` so they survive the cache-check blocks.
    let mut scope_outputs: HashSet<Value> = HashSet::new();
    for i in &emitted {
        scope_outputs.extend(i.outputs.iter().copied());
    }

    let mut out = String::new();
    let params: Vec<String> = cfg.param_names.clone();
    out.push_str(&format!("function {fn_name}({}) {{\n", params.join(", ")));

    let slot_count: usize = emitted.iter().map(|i| i.deps.len() + i.outputs.len()).sum();
    out.push_str(&format!("  const $ = _c({slot_count});\n"));
    if !scope_outputs.is_empty() {
        let mut d: Vec<String> = scope_outputs.iter().map(|v| namer.name(*v)).collect();
        d.sort();
        out.push_str(&format!("  let {};\n", d.join(", ")));
    }

    // Walk instructions; emit a scope's memo block at the position of its *last*
    // owned instruction (so every plain dependency before it is already emitted).
    let mut slot = 0usize;
    for (ix, ins) in instrs.iter().enumerate() {
        match instr_owner[ix] {
            Some(s) if ix == last_owned[s] => {
                emit_scope(&mut out, instrs, &owned[s], emitted[s], &scope_outputs, &namer, &mut slot)?;
            }
            Some(_) => {} // owned but not the last; recomputed inside the block
            None => {
                if let Some(r) = ins.result {
                    out.push_str(&format!("  const {} = {};\n", namer.name(r), expr(&ins.op, &namer)?));
                } else if let Op::StoreMember { .. } = &ins.op {
                    out.push_str(&format!("  {};\n", expr(&ins.op, &namer)?));
                }
            }
        }
    }

    match ret {
        Some(v) => out.push_str(&format!("  return {};\n", namer.name(v))),
        None => out.push_str("  return;\n"),
    }
    out.push_str("}\n");
    Ok(out)
}

fn emit_scope(
    out: &mut String,
    instrs: &[crate::cfg::Instr],
    owned: &[usize],
    info: &ScopeInfo,
    scope_outputs: &HashSet<Value>,
    namer: &Namer,
    slot: &mut usize,
) -> Result<(), String> {
    let dep_base = *slot;
    let n_dep_slots = info.deps.len();
    let out_base = dep_base + n_dep_slots;
    *slot = out_base + info.outputs.len();

    let cond = if info.deps.is_empty() {
        format!("$[{out_base}] === _e")
    } else {
        info.deps
            .iter()
            .enumerate()
            .map(|(i, d)| format!("$[{}] !== {}", dep_base + i, namer.name(*d)))
            .collect::<Vec<_>>()
            .join(" || ")
    };
    out.push_str(&format!("  if ({cond}) {{\n"));
    for (i, d) in info.deps.iter().enumerate() {
        out.push_str(&format!("    $[{}] = {};\n", dep_base + i, namer.name(*d)));
    }
    // Recompute the scope's owned instructions in order: outputs assign the outer
    // `let`; scope-internal values are local `const`; stores are statements.
    for &ix in owned {
        let ins = &instrs[ix];
        if let Some(r) = ins.result {
            let kw = if scope_outputs.contains(&r) { "" } else { "const " };
            out.push_str(&format!("    {kw}{} = {};\n", namer.name(r), expr(&ins.op, namer)?));
        } else if let Op::StoreMember { .. } = &ins.op {
            out.push_str(&format!("    {};\n", expr(&ins.op, namer)?));
        }
    }
    for (j, o) in info.outputs.iter().enumerate() {
        out.push_str(&format!("    $[{}] = {};\n", out_base + j, namer.name(*o)));
    }
    out.push_str("  } else {\n");
    for (j, o) in info.outputs.iter().enumerate() {
        out.push_str(&format!("    {} = $[{}];\n", namer.name(*o), out_base + j));
    }
    out.push_str("  }\n");
    Ok(())
}

struct Namer {
    params: std::collections::HashMap<Value, String>,
}
impl Namer {
    fn new(cfg: &Cfg) -> Namer {
        let mut params = std::collections::HashMap::new();
        for (i, p) in cfg.params.iter().enumerate() {
            if let Some(n) = cfg.param_names.get(i) {
                params.insert(*p, n.clone());
            }
        }
        Namer { params }
    }
    fn name(&self, v: Value) -> String {
        self.params.get(&v).cloned().unwrap_or_else(|| format!("_v{}", v.0))
    }
}

fn expr(op: &Op, n: &Namer) -> Result<String, String> {
    Ok(match op {
        Op::Const(c) => konst(c),
        Op::Bin(b, x, y) => format!("({} {} {})", n.name(*x), bin_str(*b), n.name(*y)),
        Op::Un(u, x) => {
            let (pre, sp) = un_str(*u);
            format!("({pre}{sp}{})", n.name(*x))
        }
        Op::Global(g) => g.clone(),
        Op::Member { obj, prop } => match prop {
            MemberKey::Static(s) => format!("{}.{s}", n.name(*obj)),
            MemberKey::Computed(c) => format!("{}[{}]", n.name(*obj), n.name(*c)),
        },
        Op::StoreMember { obj, prop, value } => match prop {
            MemberKey::Static(s) => format!("{}.{s} = {}", n.name(*obj), n.name(*value)),
            MemberKey::Computed(c) => format!("{}[{}] = {}", n.name(*obj), n.name(*c), n.name(*value)),
        },
        Op::Call { callee, args } => {
            let a: Vec<String> = args.iter().map(|x| n.name(*x)).collect();
            format!("{}({})", n.name(*callee), a.join(", "))
        }
        Op::MakeArray(e) => {
            let a: Vec<String> = e.iter().map(|x| n.name(*x)).collect();
            format!("[{}]", a.join(", "))
        }
        Op::MakeObject(props) => {
            let a: Vec<String> = props
                .iter()
                .map(|(k, v)| match k {
                    PropKey::Ident(s) => format!("{s}: {}", n.name(*v)),
                    PropKey::Computed(c) => format!("[{}]: {}", n.name(*c), n.name(*v)),
                })
                .collect();
            format!("{{{}}}", a.join(", "))
        }
        Op::ReadVar(_) | Op::WriteVar(_, _) => return Err("codegen: residual var op (run SSA first)".into()),
    })
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

fn bin_str(b: BinOp) -> &'static str {
    use BinOp::*;
    match b {
        Add => "+", Sub => "-", Mul => "*", Div => "/", Mod => "%", Pow => "**",
        Eq => "==", Ne => "!=", StrictEq => "===", StrictNe => "!==",
        Lt => "<", Le => "<=", Gt => ">", Ge => ">=",
        BitAnd => "&", BitOr => "|", BitXor => "^", Shl => "<<", Shr => ">>", UShr => ">>>",
    }
}
fn un_str(u: UnOp) -> (&'static str, &'static str) {
    use UnOp::*;
    match u {
        Neg => ("-", ""), Pos => ("+", ""), Not => ("!", ""), BitNot => ("~", ""),
        TypeOf => ("typeof", " "), Void => ("void", " "),
    }
}

/// The runtime prelude the emitted code needs (`_c` cache + sentinel) for
/// standalone execution / tests.
pub const RUNTIME: &str = "const _e = Symbol('empty');\nfunction _c(n){ return new Array(n).fill(_e); }\n";

/// React's actual import for the memo cache.
pub const REACT_IMPORT: &str = "import { c as _c } from \"react/compiler-runtime\";\n";

/// A function is a memoization target if it is a component (Capitalized name) or
/// a hook (`use` + Capital / non-alpha), exactly like the React Compiler.
pub fn is_component_or_hook(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_uppercase() => true, // component
        Some('u') if name.starts_with("use") => {
            // `use` followed by an uppercase letter (or end) is a hook.
            name[3..].chars().next().map(|c| c.is_uppercase()).unwrap_or(true)
        }
        _ => false,
    }
}

/// Compile a source snippet's function the way the React Compiler would: if it
/// is a component/hook, emit the memoized version (the `react/compiler-runtime`
/// import is prepended by the IR-rewrite pass); otherwise return the source
/// unchanged.
///
/// MEMOIZED branch: the production path is now a JSIR -> JSIR transform
/// ([`crate::memoize_plan::build_layout`] + [`jsir_transforms::memoize`]),
/// printed through the reversible IR path. The frozen string emitter
/// ([`emit_memoized`]) is retained only as the parity reference (and is still
/// exercised by the `tests/codegen.rs` node-semantic tests, which call it
/// directly).
pub fn compile(src: &str) -> Result<String, String> {
    let ir = jsir_swc::source_to_ir(src)?;
    let mut cfg = crate::lower::lower_function(&ir)?;
    crate::ssa::construct(&mut cfg);
    let name = cfg.fn_name.clone().unwrap_or_default();
    if !is_component_or_hook(&name) {
        // Not a component/hook: pass through (round-tripped from the IR).
        return jsir_swc::ir_to_source(&ir);
    }
    let r = crate::mutability::analyze(&cfg);
    let infos = crate::scopes::analyze(&cfg, &r);
    let layout = crate::memoize_plan::build_layout(&cfg, &infos, &r)?;
    // Nothing actually memoized (no surviving escaping scope): emit the function
    // unchanged rather than a degenerate `const $ = _c(0)` scaffold + runtime
    // import. React leaves such functions un-memoized, so emitting the empty
    // cache shell is a pure over-memoization (it shows up as `ours_only`).
    if layout.cache_size == 0 {
        return jsir_swc::ir_to_source(&ir);
    }
    let result = jsir_transforms::memoize::memoize_file(&ir, &layout)?;
    jsir_swc::ir_to_source(&result.file)
}
