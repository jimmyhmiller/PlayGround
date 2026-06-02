//! React component / hook detection — a port of `getReactFunctionType` /
//! `getComponentOrHookLike` from the React Compiler Rust implementation
//! (`react_compiler/src/entrypoint/program.rs`, facebook/react PR #36173).
//!
//! Used to honor `@compilationMode:"infer"`: in infer mode React compiles ONLY
//! functions it proves are components or hooks; a plain function is left
//! untouched. We mirror that so we don't over-memoize where infer-mode React
//! skips (the `ours_only` over-memoization bucket).
//!
//! We operate on the jsir IR (the AST the Rust port traverses), not the CFG: the
//! CFG lowers `new X()` to a plain call and conflates closures with array
//! literals, which would break `returns_non_node`. JSX is already desugared to
//! `createElement`/`jsx` calls by the converter, so "creates JSX" is detected as
//! such a call rather than a JSX node.

use jsir_ir::{Attr, Op as IrOp};

/// `compilationMode` requested by the fixture's config pragma. Only `infer`
/// changes our behavior (it gates compilation on component/hook detection);
/// `all` (the default) compiles everything as we already do.
fn is_infer_mode(src: &str) -> bool {
    src.contains("@compilationMode:\"infer\"") || src.contains("@compilationMode(infer)")
}

/// `use` followed by an uppercase letter or digit (`useState`, `useFoo`, `use2`).
/// Mirrors `is_hook_name` in program.rs.
fn is_hook_name(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() >= 4
        && b[0] == b'u'
        && b[1] == b's'
        && b[2] == b'e'
        && b.get(3).map_or(false, |c| c.is_ascii_uppercase() || c.is_ascii_digit())
}

/// A name starting with an uppercase letter. Mirrors `is_component_name`.
fn is_component_name(name: &str) -> bool {
    name.chars().next().map_or(false, |c| c.is_ascii_uppercase())
}

/// The string value of a `name`/`value` `Str` attribute.
fn str_attr<'a>(op: &'a IrOp, key: &str) -> Option<&'a str> {
    op.attrs.iter().find_map(|(k, v)| match v {
        Attr::Str(s) if k == key => Some(s.as_str()),
        _ => None,
    })
}

/// The `name` of an `Identifier` attribute (e.g. a member's `literal_property`,
/// a function's `id`).
fn ident_attr_name<'a>(op: &'a IrOp, key: &str) -> Option<&'a str> {
    op.attrs.iter().find_map(|(k, v)| match v {
        Attr::Identifier(id) if k == key => Some(id.name.as_str()),
        _ => None,
    })
}

/// Op names of expressions that are NOT React nodes — a function returning one
/// is not a component. Mirrors `is_non_node`.
fn is_non_node_op(name: &str) -> bool {
    matches!(
        name,
        "jsir.object_expression"
            | "jsir.arrow_function_expression"
            | "jsir.function_expression"
            | "jsir.big_int_literal"
            | "jsir.class_expression"
            | "jsir.new_expression"
    )
}

/// Collect every op in a function body's statement list into `out`, recursing
/// through structural/control-flow regions but NOT into nested functions (their
/// hooks/returns are their own). Builds the flat op set the detection scans.
fn collect_body_ops<'a>(ops: &'a [IrOp], out: &mut Vec<&'a IrOp>) {
    for op in ops {
        out.push(op);
        if matches!(
            op.name.as_str(),
            "jsir.function_declaration"
                | "jsir.function_expression"
                | "jsir.arrow_function_expression"
        ) {
            continue;
        }
        for r in &op.regions {
            for b in &r.blocks {
                collect_body_ops(&b.ops, out);
            }
        }
    }
}

/// Resolve a value id to the op that produced it (results-only), within `ops`.
fn def_of<'a>(ops: &[&'a IrOp], v: jsir_ir::ValueId) -> Option<&'a IrOp> {
    ops.iter().copied().find(|o| o.results.contains(&v))
}

/// True if the callee op denotes JSX creation (`createElement`/`jsx*`).
fn callee_creates_jsx(callee: &IrOp) -> bool {
    match callee.name.as_str() {
        "jsir.member_expression" | "jsir.optional_member_expression" => matches!(
            ident_attr_name(callee, "literal_property"),
            Some("createElement" | "jsx" | "jsxs" | "jsxDEV")
        ),
        "jsir.identifier" => matches!(str_attr(callee, "name"), Some("jsx" | "jsxs" | "_jsx" | "_jsxs" | "jsxDEV")),
        _ => false,
    }
}

/// True if the callee op denotes a hook call: a bare `useX` identifier, or a
/// `PascalCase.useX` member (matches `is_hook` / `expr_is_hook`).
fn callee_is_hook(callee: &IrOp, ops: &[&IrOp]) -> bool {
    match callee.name.as_str() {
        "jsir.identifier" => str_attr(callee, "name").map_or(false, is_hook_name),
        "jsir.member_expression" | "jsir.optional_member_expression" => {
            let hook_prop = ident_attr_name(callee, "literal_property").map_or(false, is_hook_name);
            if !hook_prop {
                return false;
            }
            // The namespace object must be a PascalCase identifier (`React.useX`).
            callee
                .operands
                .first()
                .and_then(|obj| def_of(ops, *obj))
                .and_then(|obj| match obj.name.as_str() {
                    "jsir.identifier" => str_attr(obj, "name"),
                    _ => None,
                })
                .map_or(false, is_component_name)
        }
        _ => false,
    }
}

/// Does the body call a hook or create JSX? Mirrors `calls_hooks_or_creates_jsx`.
fn calls_hooks_or_creates_jsx(body: &[&IrOp]) -> bool {
    for op in body {
        if matches!(op.name.as_str(), "jsir.call_expression" | "jsir.optional_call_expression") {
            if let Some(callee) = op.operands.first().and_then(|c| def_of(body, *c)) {
                if callee_creates_jsx(callee) || callee_is_hook(callee, body) {
                    return true;
                }
            }
        }
    }
    false
}

/// Does any (non-nested-function) return statement return a non-node value?
/// Mirrors `returns_non_node_fn` for block bodies.
fn returns_non_node(body: &[&IrOp]) -> bool {
    for op in body {
        if op.name == "jsir.return_statement" {
            if let Some(arg) = op.operands.first().and_then(|a| def_of(body, *a)) {
                if is_non_node_op(&arg.name) {
                    return true;
                }
            }
        }
    }
    false
}

/// React's `is_valid_component_params`: 0 params, or 1 (non-rest) param, or 2
/// params whose second is a ref-like identifier. (Type annotations don't occur
/// in the JS fixtures, so `is_valid_props_annotation` is always true here.)
fn is_valid_component_params(params: &[&IrOp]) -> bool {
    if params.is_empty() {
        return true;
    }
    if params.len() > 2 {
        return false;
    }
    if params[0].name == "jsir.rest_element_ref" {
        return false;
    }
    if params.len() == 1 {
        return true;
    }
    if params[1].name == "jsir.identifier_ref" {
        let n = str_attr(params[1], "name").unwrap_or("");
        n.contains("ref") || n.contains("Ref")
    } else {
        false
    }
}

/// The parameter ops of a `jsir.function_declaration` (region 0), excluding the
/// region terminator.
fn function_params(func: &IrOp) -> Vec<&IrOp> {
    func.regions
        .first()
        .and_then(|r| r.blocks.first())
        .map(|b| {
            b.ops
                .iter()
                .filter(|o| o.name != "jsir.exprs_region_end" && o.name != "jsir.expr_region_end")
                .collect()
        })
        .unwrap_or_default()
}

/// The flattened statement ops of a function's body (region 1), excluding nested
/// functions.
fn function_body_ops(func: &IrOp) -> Vec<&IrOp> {
    let mut out = Vec::new();
    if let Some(b) = func.regions.get(1).and_then(|r| r.blocks.first()) {
        collect_body_ops(&b.ops, &mut out);
    }
    out
}

/// The first top-level `jsir.function_declaration` (descending into `export`
/// wrappers), the function our pipeline compiles.
fn first_function_declaration(file: &IrOp) -> Option<&IrOp> {
    fn program_stmts(file: &IrOp) -> &[IrOp] {
        file.regions
            .first()
            .and_then(|r| r.blocks.first())
            .and_then(|b| b.ops.first()) // the program op
            .and_then(|p| p.regions.first())
            .and_then(|r| r.blocks.first())
            .map(|b| b.ops.as_slice())
            .unwrap_or(&[])
    }
    fn find(op: &IrOp) -> Option<&IrOp> {
        if op.name == "jsir.function_declaration" {
            return Some(op);
        }
        if op.name == "jsir.export_named_declaration" || op.name == "jsir.export_default_declaration" {
            return op
                .regions
                .first()
                .and_then(|r| r.blocks.first())
                .and_then(|b| b.ops.iter().find(|o| o.name == "jsir.function_declaration"));
        }
        None
    }
    program_stmts(file).iter().find_map(find)
}

/// React's `get_component_or_hook_like` for a named function declaration: returns
/// whether infer-mode React would compile it (it is a component or a hook).
fn is_component_or_hook(func: &IrOp) -> bool {
    let name = ident_attr_name(func, "id").unwrap_or("");
    let body = function_body_ops(func);
    if is_component_name(name) {
        let params = function_params(func);
        calls_hooks_or_creates_jsx(&body) && is_valid_component_params(&params) && !returns_non_node(&body)
    } else if is_hook_name(name) {
        calls_hooks_or_creates_jsx(&body)
    } else {
        false
    }
}

/// Does the function body carry an opt-in memoization directive (`'use memo'` /
/// `'use forget'`)? Such a function is compiled regardless of detection.
fn has_opt_in_directive(func: &IrOp) -> bool {
    let body = function_body_ops(func);
    body.iter().any(|op| {
        op.name == "jsir.directive_literal"
            && matches!(str_attr(op, "value"), Some("use memo") | Some("use forget"))
    })
}

/// True when the fixture is in `@compilationMode:"infer"` and the function we
/// would compile is NOT a component or hook (and has no opt-in directive) — so
/// React leaves it untouched and we must pass it through rather than memoize it.
/// Returns false for anonymous/assigned-expression components our
/// single-declaration model can't see (we don't risk wrongly skipping them).
pub fn infer_mode_skips(file: &IrOp, src: &str) -> bool {
    if !is_infer_mode(src) {
        return false;
    }
    let Some(func) = first_function_declaration(file) else {
        return false;
    };
    if has_opt_in_directive(func) {
        return false;
    }
    !is_component_or_hook(func)
}
