//! Validation passes that turn type/effect-inference results into bail
//! decisions, matching the React Compiler's `react_compiler_validation` passes.
//! A function that would error in React is passed through un-memoized (it shows
//! up as an `error.*` fixture we must not memoize).
//!
//! Each pass is gated against the corpus: it may only move `error.*` fixtures
//! out of `ours_only`, never regress an agreeing/mismatching fixture.

use std::collections::HashMap;

use crate::cfg::{Cfg, Op, Term, Value};
use crate::infer_types::Types;
use crate::types::{is_hook_name, ShapeRegistry, Type};

/// Whether `v` is a *reference to a hook* (the hook function itself, not its
/// result). Faithful to how `ValidateHooksUsage` identifies hooks: by hook-like
/// name or by a resolved hook function type.
fn is_hook_value(
    v: Value,
    def_op: &HashMap<Value, Op>,
    types: &Types,
    shapes: &ShapeRegistry,
) -> bool {
    if let Some(Op::Global(name)) = def_op.get(&v) {
        if is_hook_name(name) {
            return true;
        }
    }
    if let Type::Function { shape_id: Some(id), .. } = types.get(v) {
        if let Some(shape) = shapes.get(id) {
            if let Some(ft) = &shape.function_type {
                return ft.hook_kind.is_some();
            }
        }
    }
    false
}

/// Returns true if the function misuses a hook: calls it conditionally (a hook
/// reference used as a callee outside the entry block) or passes/uses a hook as
/// a value (anywhere other than as the callee of a call). Both are always
/// errors under the Rules of Hooks, so bailing here never regresses a valid
/// fixture.
///
/// Faithful in spirit to `validate_hooks_usage`; our SSA form lets us check it
/// structurally. A hook value is permitted *only* as the callee of a call in
/// the entry block (an unconditional hook call).
pub fn invalid_hook_usage(cfg: &Cfg, types: &Types, shapes: &ShapeRegistry) -> bool {
    let mut def_op: HashMap<Value, Op> = HashMap::new();
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if let Some(r) = ins.result {
                def_op.insert(r, ins.op.clone());
            }
        }
    }
    let is_hook = |v: Value| is_hook_value(v, &def_op, types, shapes);

    for b in &cfg.blocks {
        for ins in &b.instrs {
            match &ins.op {
                Op::Call { callee, args } => {
                    // A hook called conditionally (not in the entry block).
                    if is_hook(*callee) && b.id != cfg.entry {
                        return true;
                    }
                    // A hook passed as an argument (used as a value).
                    if args.iter().any(|a| is_hook(*a)) {
                        return true;
                    }
                    // The callee position in the entry block is the one allowed
                    // use; any other operand of this call (none besides callee)
                    // is already covered above.
                }
                other => {
                    // Any other op using a hook reference as an operand is a
                    // hook-as-value misuse (object/array/member/store/etc.).
                    if other.operands().iter().any(|v| is_hook(*v)) {
                        return true;
                    }
                }
            }
        }
        // Terminator operands: a hook reference flowing into a branch (e.g. a
        // ternary `cond ? useA : useB`) or returned is a hook-as-value misuse.
        let term_ops: Vec<Value> = match &b.term {
            Term::Br(_, args) => args.clone(),
            Term::CondBr { cond, then_args, else_args, .. } => {
                let mut v = vec![*cond];
                v.extend(then_args.iter().copied());
                v.extend(else_args.iter().copied());
                v
            }
            Term::Ret(Some(v)) => vec![*v],
            Term::Ret(None) | Term::Unreachable => vec![],
        };
        if term_ops.iter().any(|v| is_hook(*v)) {
            return true;
        }
    }
    false
}

/// Whether the source carries a `@pragma` directive (React's test fixtures gate
/// several validations behind these). Scans the leading comments verbatim.
pub fn has_pragma(src: &str, pragma: &str) -> bool {
    // Pragmas appear as `// @validateX` / `/* @validateX */`. A plain substring
    // search for `@<pragma>` as a token is sufficient and matches the oracle's
    // `parseConfigPragmaForTests` recognition.
    let needle = format!("@{pragma}");
    src.lines().take_while(|l| {
        let t = l.trim_start();
        t.starts_with("//") || t.starts_with("/*") || t.starts_with('*') || t.is_empty() || t.starts_with("import") || t.starts_with("'use")
    }).any(|l| l.contains(&needle))
        || src.contains(&needle) // fall back: pragma anywhere (fixtures keep them at top)
}

/// Returns true if a state setter (`useState`/`useReducer` dispatch) is *called*
/// during render. Gated by `@validateNoSetStateInRender`. Because closures are
/// opaque in our CFG, every setter call we can see is in render — exactly the
/// set React flags. Faithful in spirit to `validate_no_set_state_in_render`.
pub fn calls_set_state_in_render(cfg: &Cfg, types: &Types) -> bool {
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if let Op::Call { callee, .. } = &ins.op {
                if types.is_set_state(*callee) {
                    return true;
                }
            }
        }
    }
    false
}

/// Built-in capitalized callables that are *not* components (allowed to be
/// called directly even under `@validateNoCapitalizedCalls`). Mirrors the
/// default allowlist spirit of `validate_no_capitalized_calls`.
fn is_allowed_capitalized(name: &str) -> bool {
    matches!(
        name,
        "Boolean" | "Number" | "String" | "Array" | "Object" | "BigInt" | "Symbol"
            | "Date" | "Error" | "RegExp" | "Map" | "Set" | "WeakMap" | "WeakSet" | "Promise"
            | "Math" | "JSON"
    )
}

/// Returns true if the function calls a capitalized identifier as a plain
/// function (`Foo()` / `obj.Foo()`) — React reads capitalized callees as
/// component constructors, which must not be invoked directly. Gated by
/// `@validateNoCapitalizedCalls`. Faithful in spirit to
/// `validate_no_capitalized_calls`.
pub fn calls_capitalized(cfg: &Cfg) -> bool {
    let mut def_op: HashMap<Value, Op> = HashMap::new();
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if let Some(r) = ins.result {
                def_op.insert(r, ins.op.clone());
            }
        }
    }
    let capitalized = |name: &str| {
        name.chars().next().is_some_and(|c| c.is_ascii_uppercase()) && !is_allowed_capitalized(name)
    };
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if let Op::Call { callee, .. } = &ins.op {
                match def_op.get(callee) {
                    Some(Op::Global(name)) if capitalized(name) => return true,
                    Some(Op::Member { prop: crate::cfg::MemberKey::Static(name), .. }) if capitalized(name) => return true,
                    _ => {}
                }
            }
        }
    }
    false
}

/// Returns true if the function calls `eval`, which the compiler cannot analyze
/// (always a bail). No pragma needed.
pub fn calls_eval(cfg: &Cfg) -> bool {
    let mut def_op: HashMap<Value, Op> = HashMap::new();
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if let Some(r) = ins.result {
                def_op.insert(r, ins.op.clone());
            }
        }
    }
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if let Op::Call { callee, .. } = &ins.op {
                if matches!(def_op.get(callee), Some(Op::Global(n)) if n == "eval") {
                    return true;
                }
            }
        }
    }
    false
}

/// Returns true if the function calls a known impure function during render
/// (`Date.now`, `Math.random`, `performance.now`). Gated by
/// `@validateNoImpureFunctionsInRender`. Faithful to the impure-function set
/// recognized by `validate_no_impure_functions_in_render`.
pub fn calls_impure_in_render(cfg: &Cfg) -> bool {
    let mut def_op: HashMap<Value, Op> = HashMap::new();
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if let Some(r) = ins.result {
                def_op.insert(r, ins.op.clone());
            }
        }
    }
    // (object, method) pairs that are impure.
    let impure = |obj: Value, method: &str| -> bool {
        if let Some(Op::Global(o)) = def_op.get(&obj) {
            matches!(
                (o.as_str(), method),
                ("Date", "now") | ("Math", "random") | ("performance", "now")
            )
        } else {
            false
        }
    };
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if let Op::Call { callee, .. } = &ins.op {
                if let Some(Op::Member { obj, prop: crate::cfg::MemberKey::Static(m) }) = def_op.get(callee) {
                    if impure(*obj, m) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{build_builtin_shapes, build_default_globals};

    fn invalid(src: &str) -> bool {
        let mut shapes = build_builtin_shapes();
        let globals = build_default_globals(&mut shapes);
        let cfg = crate::compile_ssa(src).expect("compile");
        let types = crate::infer_types::infer(&cfg, &shapes, &globals, true, Default::default());
        invalid_hook_usage(&cfg, &types, &shapes)
    }

    #[test]
    fn conditional_hook_call_is_invalid() {
        assert!(invalid("function C(props) { let s; if (props.c) { s = useState(); } return s; }"));
    }

    #[test]
    fn hook_passed_as_arg_is_invalid() {
        assert!(invalid("function C(props) { return foo(useFoo); }"));
    }

    #[test]
    fn normal_unconditional_hook_call_is_valid() {
        assert!(!invalid("function C(props) { const x = useState(0); return x; }"));
    }

    #[test]
    fn no_hooks_is_valid() {
        assert!(!invalid("function C(props) { const x = props.a; return x; }"));
    }

    fn setstate_called(src: &str) -> bool {
        let mut shapes = build_builtin_shapes();
        let globals = build_default_globals(&mut shapes);
        let cfg = crate::compile_ssa(src).expect("compile");
        let types = crate::infer_types::infer(&cfg, &shapes, &globals, true, Default::default());
        calls_set_state_in_render(&cfg, &types)
    }

    #[test]
    fn setstate_in_render_detected() {
        assert!(setstate_called("function C() { const [t, setT] = useState(0); setT(42); return t; }"));
    }

    #[test]
    fn no_setstate_call_is_clean() {
        assert!(!setstate_called("function C() { const [t, setT] = useState(0); return t; }"));
    }

    #[test]
    fn pragma_detection() {
        assert!(has_pragma("// @validateNoSetStateInRender\nfunction C(){}", "validateNoSetStateInRender"));
        assert!(!has_pragma("function C(){}", "validateNoSetStateInRender"));
    }
}
