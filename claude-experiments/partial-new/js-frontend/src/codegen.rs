//! Emit valid JavaScript from a residual program.
//!
//! A fully-specialized residual is a single straight-line block, emitted as a
//! clean `function main(v0) { ...; return ...; }`. A residual with control flow
//! (loops / branches) is emitted as a `switch`-on-program-counter trampoline,
//! which is unambiguously correct for any (even irreducible) control-flow graph.
//! Both forms are real JavaScript that runs under any JS engine.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use partial::js::{
    render_call, render_fnref, render_get, render_new, render_opaque, residual_cap_var_id,
    residual_param_var_id, rexpr_is_ref_like, Bop, Cond, Js, Op, RExpr,
};
use partial::residual::{Program, Terminator};

/// Render an object expression as a member/call base, parenthesizing it only
/// when it is not already a reference-like chain (so `v100000.k`, `Math.x`, but
/// `((a + b)).k`).
fn base_js(e: &RExpr) -> String {
    if rexpr_is_ref_like(e) {
        rexpr_to_js(e)
    } else {
        format!("({})", rexpr_to_js(e))
    }
}

/// Render the whole residual program: every generated residual function (for
/// escaped closures) plus `main`.
pub fn program_to_js(vm: &Js, main_prog: &Program<Op, Cond>, input_var: usize) -> String {
    let mut s = String::new();
    for (rfid, rf) in vm.residual_fns().iter().enumerate() {
        let mut params: Vec<usize> = (0..rf.ncaptured).map(|i| residual_cap_var_id(rfid, i)).collect();
        params.extend((0..rf.nparams).map(|j| residual_param_var_id(rfid, j)));
        let _ = writeln!(s, "{}", render_function(&format!("__rf{rfid}"), &params, &rf.body));
    }
    let _ = write!(s, "{}", render_function("main", &[input_var], main_prog));
    s
}

/// Collect every residual variable that needs a `let` declaration, recursing
/// into nested `try`/`catch` programs (their locals are hoisted to the enclosing
/// function; the catch's exception binding is a `catch` parameter, not a `let`).
fn collect_decls(prog: &Program<Op, Cond>, decls: &mut BTreeSet<usize>) {
    for b in &prog.blocks {
        for op in &b.ops {
            match op {
                Op::Assign { var, .. } => {
                    decls.insert(*var);
                }
                Op::NewObject { dst, .. } | Op::NewArray { dst, .. } | Op::Eval { dst, .. } => {
                    decls.insert(*dst);
                }
                Op::Try { body, catch_slot, catch_body } => {
                    collect_decls(body, decls);
                    collect_decls(catch_body, decls);
                    if let Some(cs) = catch_slot {
                        decls.remove(cs); // bound by `catch (vN)`, not `let`
                    }
                }
                Op::PushOp { .. }
                | Op::SetIndex { .. }
                | Op::SetProp { .. }
                | Op::DeleteProp { .. }
                | Op::DeleteIndex { .. }
                | Op::AssignGlobal { .. }
                | Op::Return(_)
                | Op::Throw(_) => {}
            }
        }
    }
}

/// Fresh trampoline labels so a `try`/`catch` sub-program can break out of its
/// own loop without colliding with an enclosing one.
static LABEL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Render a residual program's blocks (no function wrapper, no declarations):
/// a single-block `Halt` program emits straight-line statements, anything with
/// control flow emits a `switch`-on-program-counter trampoline. Used for the
/// function body and, recursively, for `try`/`catch` bodies. `ind` is the
/// leading indentation.
///
/// `in_try` distinguishes the two uses on a normal-completion (`Halt`)
/// terminator: a function body returns from the function (`return undefined`),
/// but a `try`/`catch` sub-program must merely *end the block* (break out of its
/// trampoline) so control continues after the `try`. Rendering the latter as a
/// `return` would return from the enclosing function instead of finishing the
/// handler, e.g. a `catch` that should fall through and loop again.
fn render_blocks(prog: &Program<Op, Cond>, ind: &str, in_try: bool) -> String {
    let mut s = String::new();
    let straight_line =
        prog.blocks.len() == 1 && matches!(prog.blocks[0].term, Terminator::Halt);
    if straight_line {
        // A single Halt block: emit the ops and fall through (the block ends
        // naturally; a function then returns undefined, a try block continues).
        for op in &prog.blocks[0].ops {
            let _ = writeln!(s, "{ind}{}", op_to_js(op, ind));
        }
        return s;
    }
    let label = if in_try {
        Some(format!("__t{}", LABEL.fetch_add(1, std::sync::atomic::Ordering::Relaxed)))
    } else {
        None
    };
    let halt_stmt = match &label {
        Some(l) => format!("break {l};"),
        None => "return undefined;".to_string(),
    };
    let _ = writeln!(s, "{ind}let __pc = {};", prog.entry.0);
    if let Some(l) = &label {
        let _ = writeln!(s, "{ind}{l}: for (;;) {{");
    } else {
        let _ = writeln!(s, "{ind}for (;;) {{");
    }
    let _ = writeln!(s, "{ind}  switch (__pc) {{");
    let inner = format!("{ind}      ");
    for (i, b) in prog.blocks.iter().enumerate() {
        let _ = writeln!(s, "{ind}    case {i}: {{");
        for op in &b.ops {
            let _ = writeln!(s, "{inner}{}", op_to_js(op, &inner));
        }
        match &b.term {
            Terminator::Halt => {
                if !ends_with_return(b) {
                    let _ = writeln!(s, "{inner}{halt_stmt}");
                }
            }
            Terminator::Br(t) => {
                let _ = writeln!(s, "{inner}__pc = {}; break;", t.0);
            }
            Terminator::Cond { cond: Cond::Falsy(e), t, f } => {
                let _ = writeln!(
                    s,
                    "{inner}if (!({})) {{ __pc = {}; }} else {{ __pc = {}; }} break;",
                    rexpr_to_js(e),
                    t.0,
                    f.0
                );
            }
            Terminator::Unset => {
                let _ = writeln!(s, "{inner}throw new Error(\"unset terminator\");");
            }
        }
        let _ = writeln!(s, "{ind}    }}");
    }
    let _ = writeln!(s, "{ind}  }}");
    let _ = writeln!(s, "{ind}}}");
    s
}

/// Render one residual program as `function {name}(params...) { body }`.
fn render_function(name: &str, params: &[usize], prog: &Program<Op, Cond>) -> String {
    let mut decls: BTreeSet<usize> = BTreeSet::new();
    collect_decls(prog, &mut decls);
    for p in params {
        decls.remove(p); // parameters are not `let`-declared locals
    }

    let mut s = String::new();
    let param_list: Vec<String> = params.iter().map(|v| format!("v{v}")).collect();
    let _ = writeln!(s, "function {name}({}) {{", param_list.join(", "));
    if !decls.is_empty() {
        let names: Vec<String> = decls.iter().map(|v| format!("v{v}")).collect();
        let _ = writeln!(s, "  let {};", names.join(", "));
    }
    s.push_str(&render_blocks(prog, "  ", false));
    let _ = writeln!(s, "}}");
    s
}

fn ends_with_return(b: &partial::residual::Block<Op, Cond>) -> bool {
    matches!(b.ops.last(), Some(Op::Return(_) | Op::Throw(_)))
}

fn op_to_js(op: &Op, ind: &str) -> String {
    match op {
        Op::Assign { var, expr } => format!("v{var} = {};", rexpr_to_js(expr)),
        Op::Return(e) => format!("return {};", rexpr_to_js(e)),
        Op::NewObject { dst, fields } => {
            let fs: Vec<String> = fields
                .iter()
                .map(|(k, e)| format!("{}: {}", js_string(k), rexpr_to_js(e)))
                .collect();
            format!("v{dst} = {{{}}};", fs.join(", "))
        }
        Op::NewArray { dst, elems } => {
            let es: Vec<String> = elems.iter().map(rexpr_to_js).collect();
            format!("v{dst} = [{}];", es.join(", "))
        }
        Op::PushOp { arr, val } => {
            format!("{}.push({});", base_js(arr), rexpr_to_js(val))
        }
        Op::SetIndex { arr, index, val } => {
            format!("{}[{}] = {};", base_js(arr), rexpr_to_js(index), rexpr_to_js(val))
        }
        Op::Eval { dst, expr } => format!("v{dst} = {};", rexpr_to_js(expr)),
        Op::SetProp { obj, key, val } => {
            format!("{}.{key} = {};", base_js(obj), rexpr_to_js(val))
        }
        Op::DeleteProp { obj, key } => format!("delete {}.{key};", base_js(obj)),
        Op::DeleteIndex { arr, index } => {
            format!("delete {}[{}];", base_js(arr), rexpr_to_js(index))
        }
        Op::Throw(e) => format!("throw {};", rexpr_to_js(e)),
        Op::AssignGlobal { name, expr } => format!("{name} = {};", rexpr_to_js(expr)),
        Op::Try { body, catch_slot, catch_body } => {
            let cs = catch_slot.map(|v| format!("v{v}")).unwrap_or_else(|| "__e".to_string());
            let inner = format!("{ind}  ");
            format!(
                "try {{\n{}{ind}}} catch ({cs}) {{\n{}{ind}}}",
                render_blocks(body, &inner, true),
                render_blocks(catch_body, &inner, true),
            )
        }
    }
}

fn rexpr_to_js(e: &RExpr) -> String {
    match e {
        RExpr::Num(n) => n.to_string(),
        RExpr::Str(s) => js_string(s),
        RExpr::Bool(b) => b.to_string(),
        RExpr::Undef => "undefined".to_string(),
        RExpr::Null => "null".to_string(),
        RExpr::This => "this".to_string(),
        RExpr::Var(v) if *v == partial::js::ARGUMENTS_VAR_ID => "arguments".to_string(),
        RExpr::Var(v) => format!("v{v}"),
        RExpr::Bin(op, a, b) => {
            format!("({} {} {})", rexpr_to_js(a), bop(*op), rexpr_to_js(b))
        }
        RExpr::Index(a, i) => format!("{}[{}]", rexpr_to_js(a), rexpr_to_js(i)),
        RExpr::Opaque(op, args) => {
            let parts: Vec<String> = args.iter().map(rexpr_to_js).collect();
            render_opaque(op, &parts)
        }
        RExpr::Global(name) => name.clone(),
        RExpr::Get(o, k) => render_get(o, rexpr_to_js(o), k),
        RExpr::Call(callee, args) => {
            let a: Vec<String> = args.iter().map(rexpr_to_js).collect();
            render_call(callee, rexpr_to_js(callee), &a)
        }
        RExpr::New(callee, args) => {
            let a: Vec<String> = args.iter().map(rexpr_to_js).collect();
            render_new(callee, rexpr_to_js(callee), &a)
        }
        RExpr::FnRef { rfid, caps } => {
            let cs: Vec<String> = caps.iter().map(rexpr_to_js).collect();
            render_fnref(*rfid, &cs)
        }
    }
}

fn bop(op: Bop) -> &'static str {
    match op {
        Bop::Add => "+",
        Bop::Sub => "-",
        Bop::Mul => "*",
        Bop::Lt => "<",
        Bop::Le => "<=",
        Bop::Gt => ">",
        Bop::Ge => ">=",
        Bop::Eq => "===",
        Bop::Ne => "!==",
        Bop::BitAnd => "&",
        Bop::BitOr => "|",
        Bop::BitXor => "^",
        Bop::Shl => "<<",
        Bop::Shr => ">>",
        Bop::UShr => ">>>",
    }
}

/// A JSON-style string literal (valid as a JS string and as an object key).
fn js_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
